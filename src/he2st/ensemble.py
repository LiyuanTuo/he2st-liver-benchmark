"""Fixed, validation-selected grouped averaging; image-only checkpoint inference."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SlideDataset, read_panel, gene_digest
from .engine import environment, infer, predict
from .models import ContextFusion


def grouped_average(predictions, groups):
    weights = np.asarray([g['weight'] for g in groups], dtype=float)
    if not len(groups) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Group weights must be finite, nonnegative and sum to one')
    values=[]
    for group in groups:
        members=group['members']
        if not members or len(set(members)) != len(members):
            raise ValueError('Empty or duplicated group members')
        arrays=[np.asarray(predictions[name]) for name in members]
        if any(a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid component prediction')
        values.append(group['weight']*np.mean(arrays,axis=0))
    if any(v.shape!=values[0].shape for v in values):
        raise ValueError('Group shapes differ')
    return sum(values)


def predict_ensemble(root, manifest_path, slides, output, device='cuda'):
    root,output=Path(root),Path(output)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    genes,mask=read_panel(root/manifest['data']['panel'])
    if gene_digest(genes)!=manifest['genes_sha256']:
        raise ValueError('Panel hash differs from frozen ensemble')
    # Verify every required weight before allocating GPU memory or writing predictions.
    for entry in manifest['components']:
        path=root/entry['checkpoint']
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=entry['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {entry["name"]}')
    environment(device)
    component_dir=output.parent/(output.stem+'_components')
    component_dir.mkdir(parents=True,exist_ok=True)
    values={}; metadata=None
    for entry in manifest['components']:
        path=root/entry['checkpoint']
        if entry['format']=='core':
            pred_path=predict(root,path,slides,component_dir/f'{entry["name"]}.npz',device)
            with np.load(pred_path) as d:
                current={k:d[k] for k in ['genes','barcodes','coordinates_xy','heg50_mask']}
                value=d['predicted']
        elif entry['format']=='legacy_contextfusion':
            saved=torch.load(path,map_location=device,weights_only=True)
            model=ContextFusion().to(device); model.load_state_dict(saved['model'])
            cfg=manifest['data']
            data=SlideDataset(root,slides,cfg['arrays'],cfg['images'],cfg['scales'],genes,labels=False)
            loader=DataLoader(data,batch_size=32,num_workers=2,pin_memory=device=='cuda')
            value=infer(model,loader,saved['ym'].to(device),saved['ys'].to(device),device,entry['tta'])
            current=dict(genes=genes,heg50_mask=mask,barcodes=np.concatenate([m['barcodes'] for m in data.metadata]),coordinates_xy=np.concatenate(data.xy))
            np.savez_compressed(component_dir/f'{entry["name"]}.npz',predicted=value,**current)
            del model,saved
        else:
            raise ValueError('Unsupported checkpoint format')
        if metadata is None: metadata=current
        for key in metadata:
            np.testing.assert_array_equal(current[key],metadata[key],err_msg=f'Misaligned ensemble {key}')
        values[entry['name']]=value
        print('Inferred ensemble component:',entry['name'],flush=True)
    result=grouped_average(values,manifest['groups'])
    np.savez_compressed(output,predicted=result,**metadata)
    return output
