"""Small independent check of cached ImageNet features from actual cached pixels."""
from pathlib import Path
import json
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
model.fc = torch.nn.Identity()
mean = torch.tensor([.485, .456, .406])[None, :, None, None]
std = torch.tensor([.229, .224, .225])[None, :, None, None]
report = {}
with torch.inference_mode():
    for slide in ['C73_A1', 'C73_B1', 'C73_C1', 'C73_D1']:
        directory = ROOT / 'data/processed/gse240429/image'
        patches = np.load(directory / f'{slide}_patches_uint8.npy', mmap_mode='r')
        features = np.load(directory / f'{slide}_resnet18.npy')
        ids = np.linspace(0, len(patches)-1, 8, dtype=int)
        pixels = torch.from_numpy(patches[ids].copy()).permute(0, 3, 1, 2).float()/255
        actual = model((pixels-mean)/std).numpy()
        diff = float(abs(actual-features[ids]).max())
        target = features[ids]
        relative_l2 = float(np.linalg.norm(actual-target)/np.linalg.norm(target))
        cosine = float(np.min((actual*target).sum(1)/(np.linalg.norm(actual, axis=1)*np.linalg.norm(target, axis=1))))
        modes = {}
        model.cuda()
        for tf32 in [False, True]:
            torch.backends.cudnn.allow_tf32 = tf32
            gpu = model(((pixels-mean)/std).cuda()).cpu().numpy()
            modes[str(tf32)] = dict(max_error=float(abs(gpu-target).max()),
                relative_l2=float(np.linalg.norm(gpu-target)/np.linalg.norm(target)))
        model.cpu()
        # CPU/CUDA kernels (including TF32) need not reproduce bits. Check
        # matching feature direction and magnitude, and retain actual errors.
        assert relative_l2 < .005 and cosine > .99999
        report[slide] = dict(sampled_spots=8, cpu_recomputed_vs_cached_max_error=diff,
                            cpu_relative_l2=relative_l2, cpu_min_cosine=cosine,
                            cuda_cudnn_tf32_modes=modes,
                            numerical_feature_match=True, bitwise_identity_claimed=False)
        print(slide, report[slide], flush=True)
(ROOT / 'results/verified_20260913/cached_feature_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
