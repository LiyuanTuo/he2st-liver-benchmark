# ContextFusion

以每个 spot 为中心裁取边长 224、896、1792 像素的图块，均缩放为 224×224。三幅图共享 CNN 编码器，特征拼接后预测 200 个基因的相对表达。

```text
H&E + spot 坐标
  → 三尺度图像块
  → 共享 ResNet18 / EfficientNet-B0
  → 特征拼接 → LayerNorm → MLP（256 维隐藏层）
  → HEG200 log 表达
```

## 训练

按 A1+B1 的逐基因均值和标准差对表达标签标准化。损失为 MSE + 0.2 × 批内基因相关性损失。训练使用 AdamW、学习率预热和余弦衰减、梯度裁剪，以及颜色、旋转和翻转增强。

[configs](../configs)中的三组配置均采用 ImageNet 初始化，编码器和预测头一起训练。在 C1 比较训练参数与指数移动平均参数（EMA），以及单次预测与四种旋转角度的预测平均（TTA）。

`resnet18_spatial` 将训练标签与同切片最近 6 个其他 spots 的平均表达按 0.8/0.2 混合。验证和测试目标保持实测表达。随机初始化和单视野对照见[多尺度实验](06_LIVER_CONTEXT_IMPROVEMENT.md)。

## 代码

| 文件 | 功能 |
|---|---|
| [data.py](../src/he2st/data.py) | 样本对齐、图像读取和标签 |
| [models.py](../src/he2st/models.py) | 编码器与回归头 |
| [engine.py](../src/he2st/engine.py) | 训练、验证与推理 |
| [ensemble.py](../src/he2st/ensemble.py) | 多模型预测平均 |
| [metrics.py](../src/he2st/metrics.py) | PCC、MAE 等指标 |
| [cli.py](../src/he2st/cli.py) | 命令行入口 |
