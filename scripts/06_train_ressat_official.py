"""在 ResSAT 官方示例数据上运行官方模型主体。

脚本固定随机种子并写死全部配置，便于重复执行。模型、数据集、优化器、
损失、保存逻辑均直接调用 `third_party/ResSAT`，不在第三方仓库中改代码。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/ResSAT"))

from ressat.data_loader import load_gene_names, load_sections
from ressat.models import ResSAT


def as_uint8_patches(sections):
    """把官方示例的 float32(0-255) patch 转成 uint8。

    现代 torchvision 的 ToPILImage 会把 float32 数组当作 [0,1] 缩放：
    官方示例的 0-255 float patch 会被 *255 后按 uint8 回绕，图像被静默损坏。
    转换为 uint8 后 ToPILImage 原样通过，恢复官方流程的真实输入。
    """

    for section in sections:
        section["data"] = [
            (patch.astype(np.uint8, copy=False), embedding)
            for patch, embedding in section["data"]
        ]
    return sections


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_dir = ROOT / "data/official_ressat_example"
    result_dir = ROOT / "results/ressat_official"
    train, validation, _ = load_sections(str(data_dir), section_num=2)
    train = as_uint8_patches(train)
    validation = as_uint8_patches(validation) if validation is not None else None
    model = ResSAT(
        train_sections=train,
        val_sections=validation,
        data_dir=str(data_dir),
        result_dir=str(result_dir),
        patch_size=224,
        num_fourier=128,
        sigma=1,
        dropout=0.3,
        num_workers=4,
        exp_name="official_seed42",
        gene_names=load_gene_names(str(data_dir)),
    )
    model.fit(
        num_epochs=100,
        lr=5e-4,
        batch_size=32,
        patience=10,
        weight_decay=1e-6,
    )


if __name__ == "__main__":
    main()
