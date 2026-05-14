# ResNet Image Classification and Interpretability

This module implements a ResNet-18 image classification pipeline with custom normalization experiments and Grad-CAM visualizations.

## Features

- ResNet-18 implementation in PyTorch
- Normalization variants: built-in BatchNorm, custom BatchNorm, InstanceNorm, Batch-InstanceNorm, LayerNorm, GroupNorm, and no normalization
- Training metrics: loss, accuracy, micro-F1, and macro-F1
- Checkpointing, resume support, early stopping, and learning-curve plotting
- Augmentation options including Cutout, MixUp, AutoAugment-style policies, and PCA color augmentation
- Grad-CAM generation for model interpretability

## Dataset Layout

```text
DATA_ROOT/
  train/
    class_a/
      image_001.jpeg
      ...
  val/
    class_a/
      image_101.jpeg
      ...
```

Folder names are used as class labels. Uppercase image extensions such as `.JPEG` are supported.

## Training

```bash
python train.py \
  --data-root /path/to/dataset \
  --output-dir outputs/resnet18_gn \
  --norm gn \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.05 \
  --amp
```

Useful switches:

- `--norm baseline|bn|in|bin|ln|gn|nn`
- `--use-cutout --cutout-size 56`
- `--mixup-alpha 0.2`
- `--autoaugment-policy imagenet`
- `--train-resize-mode alexnet_square|resnet_paper|resnet_strict`

## Evaluation

```bash
python evaluate.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/dataset \
  --split val
```

## Grad-CAM

```bash
python generate_gradcam.py \
  --checkpoint outputs/resnet18_gn/checkpoint_best.pt \
  --data-root /path/to/dataset \
  --output-dir outputs/gradcam \
  --split val
```

Outputs are grouped by class and prediction correctness, with a `gradcam_summary.json` file for downstream analysis.

