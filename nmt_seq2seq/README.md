# Neural Machine Translation

This module implements configurable encoder-decoder translation systems for English-to-Indic translation experiments.

## Features

- Word-level source encoder with optional GloVe initialization
- BERT encoder variants with frozen or fine-tuned parameters
- BPE target tokenizer built with Hugging Face tokenizers
- Attention-based LSTM decoder
- Constant and scheduled teacher forcing
- Greedy and beam-search decoding
- BLEU and chrF evaluation
- JSON configs for reproducible experiment runs

## Training

```bash
python scripts/train.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json
```

The config controls dataset paths, encoder type, decoder settings, optimization, teacher forcing, and evaluation behavior. Relative paths are resolved from the module root or the config directory.

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/en_hi_bert_finetune_attn_sched_final40.json \
  --checkpoint outputs/en_hi_bert_finetune_attn_sched/checkpoint_best.pt
```

## Inference

```bash
python inference.py \
  --input /path/to/input.csv \
  --output predictions.csv \
  --checkpoint /path/to/checkpoints \
  --mode 4 \
  --decoding_strategy beam
```

The input CSV should contain a `source` column. The output CSV preserves the input columns and adds `translated`.

