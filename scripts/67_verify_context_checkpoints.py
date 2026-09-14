"""Fresh GPU inference of saved CNN checkpoints, independent of training loop."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion',ROOT/'scripts/59_train_context_fusion.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class VerifySingle(m.ContextFusion):
    def forward(self,x):
        # Match the control's operation order: rotate first, expand inside forward.
        return super().forward(x[:,:1].expand(-1,3,-1,-1,-1))
def main():
    torch.backends.cudnn.benchmark=False
    center=torch.tensor([.485,.456,.406],device='cuda')[None,None,:,None,None]
    scale=torch.tensor([.229,.224,.225],device='cuda')[None,None,:,None,None]
    source=[np.load(m.common.CACHE/f'C73_D1_{s}_patches.npy',mmap_mode='r')[:32] for s in [224,896,1792]]
    # DataLoader stacks Dataset samples into contiguous N,S,C,H,W tensors.
    # Preserve that layout: channels-last BF16 convolution can round differently.
    x=torch.from_numpy(np.stack(source,1).copy()).permute(0,1,4,2,3).contiguous().cuda().float()/255
    x=(x-center)/scale;results={}
    cases=[('liver_context_fusion_20260913','imagenet',False),('liver_context_fusion_20260913','scratch',False),
      ('liver_single_view_control_20260913','imagenet',True),('liver_context_fusion_seed17_20260913','imagenet',False),('liver_context_fusion_seed83_20260913','imagenet',False)]
    for directory,arm,single in cases:
        print('verify',directory,arm,flush=True)
        base=ROOT/'results'/directory;ck=base/f'{arm}_best.pt';saved=torch.load(ck,map_location='cuda',weights_only=True)
        model=(VerifySingle(False) if single else m.ContextFusion(False)).cuda();model.load_state_dict(saved['model']);model.eval()
        selection=json.loads((base/f'{arm}_selection.json').read_text());image=x
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            if selection['tta']:pred=torch.stack([model(torch.rot90(image,k,[-2,-1])) for k in range(4)]).float().mean(0)
            else:pred=model(image).float()
        pred=(pred*saved['ys']+saved['ym']).cpu().numpy()
        old=np.load(base/f'{arm}_D1.npz')['predicted'][:32]
        delta=float(np.max(np.abs(pred-old)))
        # Archive inference used BF16 + cuDNN benchmarking. Audit numerical
        # stability against FP32 and FULL-slide PCC, not unjustified bit identity.
        with torch.inference_mode():
            if selection['tta']:reference=torch.stack([model(torch.rot90(image,k,[-2,-1])) for k in range(4)]).mean(0)
            else:reference=model(image)
        reference=(reference*saved['ys']+saved['ym']).cpu().numpy()
        with ck.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        item=dict(sample_spots=32,sample_max_abs_prediction_difference=delta,checkpoint_sha256=digest,epoch=saved['epoch'],
                  archived_BF16_vs_FP32_RMSE=float(np.sqrt(np.mean((old-reference)**2))),
                  fresh_BF16_vs_FP32_RMSE=float(np.sqrt(np.mean((pred-reference)**2))))
        results[directory+'/'+arm]=item
        # Always re-infer ALL 2265 test spots. Also measure train learnability
        # for the main recipe controls, without any random augmentation.
        train_true=[];train_pred=[]
        slides=['A1','B1','D1'] if directory in ['liver_context_fusion_20260913','liver_single_view_control_20260913'] else ['D1']
        for slide in slides:
                arrays=np.load(ROOT/f'data/processed/gse240429_heg/arrays/C73_{slide}.npz')
                true=m.common.norm(arrays['raw_counts']);rows=[]
                patches=[np.load(m.common.CACHE/f'C73_{slide}_{s}_patches.npy',mmap_mode='r') for s in [224,896,1792]]
                for start in range(0,len(patches[0]),32):
                    batch=torch.from_numpy(np.stack([p[start:start+32] for p in patches],1).copy()).permute(0,1,4,2,3).contiguous().cuda().float()/255
                    batch=(batch-center)/scale
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        if selection['tta']:z=torch.stack([model(torch.rot90(batch,k,[-2,-1])) for k in range(4)]).float().mean(0)
                        else:z=model(batch).float()
                    rows.append((z*saved['ys']+saved['ym']).cpu().numpy())
                prediction=np.concatenate(rows)
                if slide!='D1':train_true.append(true);train_pred.append(prediction)
                else:
                    archive=np.load(base/f'{arm}_D1.npz')
                    np.testing.assert_array_equal(archive['genes'],arrays['genes'])
                    np.testing.assert_array_equal(archive['barcodes'],arrays['barcodes'])
                    np.testing.assert_array_equal(archive['coordinates_xy'],arrays['coordinates_xy'])
                    np.testing.assert_array_equal(archive['truth'],true)
                    before=m.common.score(true,archive['predicted']);after=m.common.score(true,prediction)
                    item.update(checked_test_spots=len(true),archived_metrics=before,fresh_metrics=after,
                        full_slide_prediction_RMSE=float(np.sqrt(np.mean((archive['predicted']-prediction)**2))),
                        full_slide_max_abs_difference=float(np.max(np.abs(archive['predicted']-prediction))),
                        absolute_PCC_difference={k:abs(before[k]-after[k]) for k in ['HEG200','HEG50']})
                    # 0.001 PCC is far below the ~0.08 / 0.12 reported gains.
                    assert max(item['absolute_PCC_difference'].values())<.001,item
                    assert item['full_slide_prediction_RMSE']<.003,item
                    np.savez_compressed(ROOT/'results/liver_context_20260913'/f'fresh_{directory}_{arm}_D1.npz',predicted=prediction)
        if train_true:item['train_metrics']=m.common.score(np.concatenate(train_true),np.concatenate(train_pred))
        item['validation_metrics']=selection['validation']
        item['verification']='Full-slide metric reproduction; BF16 numerical differences allowed and measured, not bitwise identity.'
        print(json.dumps(item),flush=True)
        target=ROOT/'results/liver_context_20260913/fresh_checkpoint_verification.json';target.write_text(json.dumps(results,indent=2))
        del model,saved
    target=ROOT/'results/liver_context_20260913/fresh_checkpoint_verification.json';target.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
