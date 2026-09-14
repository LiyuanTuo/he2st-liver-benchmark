"""Stem 统一协议适配：训练（复用官方模型/扩散代码，替换数据管线）。

与官方仓库的差异（均为显式适配，不改动 third_party）：
- 条件特征：本项目统一 ResNet18 512 维特征（官方用 UNI+CONCH 拼接），
  并用训练切片统计量 z-score；
- 目标：本项目统一 log1p(1e4 归一化) 200 基因面板（官方用 log2(count+1)）；
- 划分：train=A1+B1 / val=C1（早停）/ test=D1；
- 单 GPU 训练循环（官方脚本硬编码 DDP 与 cuda:6），保留官方 EMA 策略；
- DiT 12 层 × 384 隐层 × 6 头（仓库默认），batch 128，lr 1e-4。
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/Stem"))

from Stem.diffusion import create_diffusion
from Stem.models import Stem_models


ARRAYS = ROOT / "data/processed/gse240429/arrays"
IMAGES = ROOT / "data/processed/gse240429/image"
OUTPUT = ROOT / "results/stem_adapted"
TRAIN_SLIDES = ("C73_A1", "C73_B1")
VALIDATION_SLIDE = "C73_C1"

BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-4
VALIDATE_EVERY = 5
PATIENCE = 100
HIDDEN_SIZE = 384
DEPTH = 12
NUM_HEADS = 6
EMA_DECAY = 0.9999
USE_AMP = True  # 8 GB 适配：fp32 前向图在 batch>32 时溢出显存并显著变慢（实测见 dev_stem_*）


def load_slide(slide: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(ARRAYS / f"{slide}.npz") as source:
        expression = source["log_normalized"].astype(np.float32)
    features = np.load(IMAGES / f"{slide}_resnet18.npy", mmap_mode="r").copy()
    if len(features) != len(expression):
        raise ValueError(f"{slide} 特征与表达行数不一致")
    return expression, features


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float = EMA_DECAY) -> None:
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    train_parts = [load_slide(slide) for slide in TRAIN_SLIDES]
    validation = load_slide(VALIDATION_SLIDE)
    train_x = np.concatenate([part[0] for part in train_parts]).astype(np.float32)
    train_y = np.concatenate([part[1] for part in train_parts]).astype(np.float32)
    validation_x, validation_y = (t.astype(np.float32) for t in validation)

    feature_mean = train_y.mean(axis=0, keepdims=True)
    feature_std = np.maximum(train_y.std(axis=0, keepdims=True), 1e-6)
    train_y = (train_y - feature_mean) / feature_std
    validation_y = (validation_y - feature_mean) / feature_std

    model = Stem_models["Stem"](
        input_size=train_x.shape[1],
        depth=DEPTH,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        label_size=train_y.shape[1],
    ).to(device)
    ema = deepcopy(model)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    diffusion = create_diffusion("")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    validation_x_t = torch.from_numpy(validation_x).to(device)
    validation_y_t = torch.from_numpy(validation_y).to(device)

    def diffusion_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
        return diffusion.training_losses(model, x, t, dict(y=y))["loss"].mean()

    best_loss, stale = float("inf"), 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for x, y in loader:
            x = x.unsqueeze(1).to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                loss = diffusion_loss(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model)
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        record = {"epoch": epoch, "train_loss": train_loss}
        if epoch % VALIDATE_EVERY == 0 or epoch == 1:
            model.eval()
            # Compare checkpoints with identical validation noise/timesteps;
            # restore the training RNG state when validation finishes.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), torch.inference_mode():
                torch.manual_seed(20260913)
                val_loss = 0.0
                for start in range(0, len(validation_x_t), BATCH_SIZE):
                    stop = min(start + BATCH_SIZE, len(validation_x_t))
                    with autocast:
                        val_loss += float(
                            diffusion_loss(
                                validation_x_t[start:stop].unsqueeze(1),
                                validation_y_t[start:stop],
                            ).item()
                        )
                val_loss /= math.ceil(len(validation_x_t) / BATCH_SIZE)
            record["validation_loss"] = val_loss
            if val_loss < best_loss - 1e-7:
                best_loss, stale = val_loss, 0
                torch.save(
                    {"model": model.state_dict(), "ema": ema.state_dict(), "epoch": epoch,
                     "ema_decay": EMA_DECAY, "updates": epoch * len(loader)},
                    OUTPUT / "best.pt",
                )
            else:
                stale += VALIDATE_EVERY
                if stale >= PATIENCE:
                    history.append(record)
                    print(f"epoch={epoch} train={train_loss:.5f} val={val_loss:.5f} "
                          f"(early stop)")
                    break
        history.append(record)
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} train={train_loss:.5f} "
                  f"val={record.get('validation_loss', float('nan')):.5f} "
                  f"elapsed={time.perf_counter() - started:.0f}s")

    summary = {
        "protocol": {
            "train_slides": list(TRAIN_SLIDES),
            "validation_slide": VALIDATION_SLIDE,
            "target": "unified log1p(1e4-normalized) 200-gene panel",
            "condition": "z-scored unified ResNet18-512 features (stats from train)",
            "model": f"Stem DiT depth={DEPTH} hidden={HIDDEN_SIZE} heads={NUM_HEADS}",
            "diffusion": "linear 1000 steps, learned-sigma range, MSE (official defaults)",
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "ema_decay": EMA_DECAY,
            "amp": USE_AMP,
            "early_stopping": f"validation loss, patience {PATIENCE} (every {VALIDATE_EVERY} epochs)",
        },
        "best_validation_loss": best_loss if best_loss != float("inf") else None,
        "epochs_run": len(history),
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "training.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"完成：{OUTPUT / 'training.json'}")


if __name__ == "__main__":
    main()
