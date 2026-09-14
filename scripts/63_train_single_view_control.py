"""Exact CNN training-recipe control: repeat local view instead of 3 scales.

Same parameter count, loss, augmentation, optimizer, epochs and C1 selection.
This isolates availability of broad image context in the new training recipe.
"""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion',ROOT/'scripts/59_train_context_fusion.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
Base=fusion.ContextFusion
class SingleViewControl(Base):
    def forward(self,x):
        return super().forward(x[:,:1].expand(-1,3,-1,-1,-1))
if __name__=='__main__':
    fusion.ContextFusion=SingleViewControl
    fusion.ARMS=['imagenet']
    fusion.OUT=ROOT/'results/liver_single_view_control_20260913';fusion.OUT.mkdir(parents=True,exist_ok=True)
    (fusion.OUT/'input_override.json').write_text(json.dumps(dict(input='224px local patch repeated three times',
      reason='same model parameter count and training recipe, isolate broad context',test_selected=False),indent=2))
    fusion.main()
