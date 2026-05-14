# Deep Learning Systems: Vision and Neural Machine Translation

This repository collects two production-style deep learning pipelines built in PyTorch:

- **Image classification and interpretability**: a from-scratch ResNet-18 training stack with custom normalization layers, modern augmentation options, checkpointing, metrics, and Grad-CAM visual explanations.
- **Neural machine translation**: an encoder-decoder NMT system for English-to-Indic translation experiments with attention, BPE target tokenization, GloVe and BERT encoder variants, scheduled teacher forcing, beam search, and BLEU/chrF evaluation.

The code is organized as a research engineering project: reusable modules, JSON-configured experiments, command-line entry points, cached preprocessing, saved tokenizer artifacts, and training/evaluation utilities.

## Repository Layout

```text
.
├── vision_resnet/          # ResNet-18 image classification experiments
│   ├── vision/             # datasets, transforms, model, metrics, training engine
│   ├── train.py            # training entry point
│   ├── evaluate.py         # checkpoint evaluation
│   └── generate_gradcam.py # Grad-CAM visualizations
├── nmt_seq2seq/            # English-to-Indic neural machine translation
│   ├── nmt/                # data, model, decoding, metrics, training utilities
│   ├── configs/            # experiment configs for GloVe/BERT/attention variants
│   ├── scripts/            # training, evaluation, and data inspection CLIs
│   └── inference.py        # batch translation entry point
└── requirements.txt
```

## Highlights

- Implemented ResNet-18 without relying on torchvision model definitions.
- Added interchangeable normalization strategies: BatchNorm, InstanceNorm, Batch-InstanceNorm, LayerNorm, GroupNorm, and no-normalization baselines.
- Built an image preprocessing stack with random crops, horizontal flips, Cutout, MixUp, AutoAugment-style policies, and optional PCA color augmentation.
- Added Grad-CAM tooling to inspect successful and failed predictions.
- Implemented an attention-based seq2seq translation model with configurable GloVe or BERT encoders.
- Built tokenizer/vocabulary pipelines, teacher-forcing schedules, beam search decoding, and generation metrics.

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a vision model:

```bash
cd vision_resnet
python train.py \
  --data-root /path/to/imagenet-style-data \
  --output-dir outputs/resnet18_gn \
  --norm gn \
  --epochs 100 \
  --batch-size 128 \
  --amp
```

Train a translation model:

```bash
cd nmt_seq2seq
python scripts/train.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json
```

Run translation inference with a trained checkpoint bundle:

```bash
cd nmt_seq2seq
python inference.py \
  --input /path/to/input.csv \
  --output predictions.csv \
  --checkpoint /path/to/checkpoints \
  --mode 4 \
  --decoding_strategy beam
```

## CV-Friendly Summary

**Deep Learning Systems: Vision and Neural Machine Translation**  
Built PyTorch pipelines for ResNet-based image classification with custom normalization/Grad-CAM interpretability and attention-based English-to-Indic neural machine translation with BERT/GloVe encoders, BPE tokenization, beam search, and BLEU/chrF evaluation.
