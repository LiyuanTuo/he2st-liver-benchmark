import sys

sys.path.insert(0, "third_party/ResSAT")

from ressat.data_loader import load_sections, load_gene_names
from ressat.dataset import build_spot_records, HEPatchesDataset

train, val, test = load_sections("data/processed/ressat_gse240429", section_num=4)
print("sections:", len(train), "train,", len(val), "val,", len(test), "test")
print("genes:", len(load_gene_names("data/processed/ressat_gse240429")))

records = build_spot_records(test)
print("test records:", len(records))
ds = HEPatchesDataset(records, patch_size=224)
image, expr, coord, sec_id, spot_id = ds[0]
print("item:", tuple(image.shape), tuple(expr.shape), tuple(coord.shape))
print("image stats:", float(image.min()), float(image.max()), float(image.mean()))
print("expr stats:", float(expr.min()), float(expr.max()))
print("coord:", coord.tolist(), "sec:", int(sec_id), "spot:", int(spot_id))
