"""Repeat unchanged ContextFusion ImageNet training for seeds 17 and 83."""
from pathlib import Path
import importlib.util,json,time
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion',ROOT/'scripts/59_train_context_fusion.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
if __name__=='__main__':
    fusion.ARMS=['imagenet']
    for seed in [17,83]:
        fusion.SEED=seed;fusion.OUT=ROOT/f'results/liver_context_fusion_seed{seed}_20260913'
        fusion.OUT.mkdir(parents=True,exist_ok=True);fusion.main()
