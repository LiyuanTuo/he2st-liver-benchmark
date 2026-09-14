"""C1 chooses equal seed ensemble vs individual seed; no D1-driven weights."""
from pathlib import Path
import importlib.util,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('common',ROOT/'scripts/58_liver_context_experiments.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
OUT=ROOT/'results/liver_context_ensemble_20260913';OUT.mkdir(parents=True,exist_ok=True)
DIRS=[ROOT/'results/liver_context_fusion_20260913']+[ROOT/f'results/liver_context_fusion_seed{s}_20260913' for s in [17,83]]
def main():
    vp=[dict(np.load(d/'imagenet_C1.npz')) for d in DIRS]
    for p in vp[1:]:np.testing.assert_array_equal(p['genes'],vp[0]['genes']);np.testing.assert_array_equal(p['truth'],vp[0]['truth'])
    candidates={f'seed{s}':p['predicted'] for s,p in zip([42,17,83],vp)}
    candidates['equal_ensemble']=np.mean([p['predicted'] for p in vp],0)
    scores={n:common.score(vp[0]['truth'],p) for n,p in candidates.items()}
    selected=max(scores,key=lambda n:scores[n]['HEG200'])
    (OUT/'selection_locked.json').write_text(json.dumps(dict(selected=selected,validation=scores,unix=time.time()),indent=2))
    dp=[dict(np.load(d/'imagenet_D1.npz')) for d in DIRS]
    predictions={f'seed{s}':p['predicted'] for s,p in zip([42,17,83],dp)}
    predictions['equal_ensemble']=np.mean([p['predicted'] for p in dp],0)
    for p in dp[1:]:
        for key in ['genes','barcodes','coordinates_xy','truth']:np.testing.assert_array_equal(p[key],dp[0][key])
    results={n:common.score(dp[0]['truth'],p) for n,p in predictions.items()}
    (OUT/'test_metrics.json').write_text(json.dumps(dict(selected=selected,models=results),indent=2))
    np.savez_compressed(OUT/'selected_D1.npz',predicted=predictions[selected],**{k:dp[0][k] for k in ['genes','barcodes','coordinates_xy','truth']})
    np.savez_compressed(OUT/'selected_C1.npz',predicted=candidates[selected],truth=vp[0]['truth'],genes=vp[0]['genes'])
    print(selected,results,flush=True)
if __name__=='__main__':main()
