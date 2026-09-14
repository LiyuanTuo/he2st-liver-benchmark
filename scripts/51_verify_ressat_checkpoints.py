"""Re-run fixed SA/SP checkpoints; compare fresh predictions to archived arrays."""
from pathlib import Path
import hashlib
import json
import pickle
import random
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/ResSAT'))
sys.path.insert(0, str(ROOT / 'src'))
from ressat.models import ResSAT
from ressat.utils import back_project
from he2st.metrics import per_gene_pearson


def main():
    torch.set_num_threads(6)
    out = ROOT / 'results/verified_20260913'
    out.mkdir(exist_ok=True, parents=True)
    report = {}
    for name, data_dir, exp_dir, checkpoint_pattern, archive in [
        ('SA', 'data/official_ressat_example', 'results/ressat_official',
         'official_seed42/lightning_logs/version_2/checkpoints/*.ckpt', 'official_batch32_predictions.npz'),
        ('SP', 'data/processed/ressat_original_rebuilt/SP', 'results/ressat_sp_rebuilt',
         'sp_section2_to_section1_seed42/lightning_logs/version_0/checkpoints/*.ckpt', 'SP1_predictions.npz')]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        directory = ROOT / data_dir
        with open(directory / 'Section_1/dataset.pkl', 'rb') as f:
            patches = pickle.load(f)
        patches = [(p.astype(np.uint8), e) for p, e in patches]
        with open(directory / 'Section_1/locations.pkl', 'rb') as f:
            coords = pickle.load(f)
        with open(directory / 'pca_info.pkl', 'rb') as f:
            pca = pickle.load(f)
        ckpts = list((ROOT / exp_dir).glob(checkpoint_pattern))
        assert len(ckpts) == 1
        # Dedicated output avoids touching the archived experiment directories.
        model = ResSAT(test_sections=[dict(data=patches, locs=coords, section_id=1)],
                       data_dir=str(directory), result_dir=str(out), patch_size=224,
                       num_fourier=128, sigma=1, dropout=.3, num_workers=2,
                       exp_name=f'verify_{name}', gene_names=pca['gene_names'])
        model.load_checkpoint(str(ckpts[0]))
        pred_pc, true_pc = model.predict(batch_size=32)
        pred, truth = back_project(pred_pc, pca).numpy(), back_project(true_pc, pca).numpy()
        with np.load(ROOT / exp_dir / archive) as d:
            previous = d['predicted']
        delta = float(np.abs(pred - previous).max())
        np.testing.assert_allclose(pred, previous, atol=2e-4, rtol=2e-4)
        pcc = per_gene_pearson(truth, pred)
        high = np.argsort(truth.mean(0))[-50:]
        with open(ckpts[0], 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        report[name] = dict(checkpoint=str(ckpts[0].relative_to(ROOT)), checkpoint_sha256=digest,
                            maximum_absolute_difference_to_archive=delta,
                            HVG2000=float(np.nanmean(pcc)), HEG50_paper_diagnostic=float(np.nanmean(pcc[high])),
                            match=True, action='Fresh inference from existing trained checkpoint; no new training')
        (out / 'fresh_checkpoint_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(name, report[name], flush=True)
        del model, patches, pred_pc, true_pc
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
