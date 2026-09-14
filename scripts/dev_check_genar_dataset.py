import sys

sys.path.insert(0, "third_party/GenAR/src")

import configs
import main as genar_main

configs.DATASETS["gse240429"] = {
    "dir_name": "gse240429",
    "val_slides": "C73_C1",
    "test_slides": "C73_D1",
    "recommended_encoder": "resnet18",
}

argv = [
    "--dataset", "gse240429", "--data-root", "data/processed",
    "--encoder", "resnet18", "--gpus", "1", "--global-batch-size", "16",
    "--epochs", "1", "--lr", "1e-4", "--seed", "2021",
    "--max-gene-count", "2000", "--precision", "16-mixed",
    "--allow-nondeterministic",
]
cfg = genar_main.build_config_from_args(genar_main.get_parse().parse_args(argv))
print("config ok:", cfg.DATA.global_batch_size, cfg.MODEL.scale_dims, cfg.MODEL.embed_dim)

from dataset.hest_dataset import STDataset

ds = STDataset(
    mode="train", data_path=cfg.data_path, expr_name="gse240429",
    slide_val="C73_C1", slide_test="C73_D1", encoder_name="resnet18",
    max_gene_count=2000, prediction_mode="discrete", library_scale=10000.0,
    grouping_mode="kmeans", grouping_seed=42,
)
print("train spots:", len(ds), "| genes:", len(ds.genes))
item = ds[0]
print("item keys:", sorted(item.keys()))
print("img", tuple(item["img"].shape), "target", tuple(item["target_genes"].shape),
      "pos", tuple(item["positions"].shape))
print("target dtype", item["target_genes"].dtype, "max", int(item["target_genes"].max()),
      "slide", item["slide_id"])
