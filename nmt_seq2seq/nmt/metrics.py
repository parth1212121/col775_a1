from __future__ import annotations

import sacrebleu


def compute_mt_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if not predictions:
        return {"bleu": 0.0, "chrf": 0.0, "ter": 0.0}

    return {
        "bleu": float(sacrebleu.corpus_bleu(predictions, [references], force=True).score),
        "chrf": float(sacrebleu.corpus_chrf(predictions, [references]).score),
        "ter": float(sacrebleu.corpus_ter(predictions, [references], normalized=True).score),
    }
