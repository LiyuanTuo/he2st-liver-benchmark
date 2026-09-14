import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/ResSAT"))

from ressat.data_loader import load_sections
from torchvision import transforms

train, val, test = load_sections(str(ROOT / "data/official_ressat_example"), section_num=2)
p, e = test[0]["data"][0]
print("raw patch min/max/dtype:", float(p.min()), float(p.max()), p.dtype)
t = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
out = t(p)
print("transformed min/max/mean:", float(out.min()), float(out.max()), float(out.mean()))
