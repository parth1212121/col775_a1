#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nmt.config import load_config, save_config
from nmt.data import Vocabulary, load_or_build_glove_embedding_matrix, prepare_data_bundle
from nmt.model import build_model
from nmt.training import train_model
from nmt.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural machine translation models.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--output-dir", default=None, help="Optional override for config.output_dir.")
    return parser.parse_args()


def maybe_override_output_dir(config: dict, output_dir_override: str | None) -> dict:
    if output_dir_override:
        config["output_dir"] = str(Path(output_dir_override).resolve())
    return config


def prepare_source_embedding_matrix(config: dict, data_bundle, output_dir: Path):
    if config["source"]["type"] != "word":
        return None

    assert isinstance(data_bundle.source_processor, Vocabulary)
    return load_or_build_glove_embedding_matrix(
        vocab=data_bundle.source_processor,
        embedding_dim=config["source"]["embedding_dim"],
        glove_path=config["source"]["glove_path"],
        cache_path=output_dir / "artifacts" / "source_embedding_matrix.pt",
    )


def build_run_summary(results: dict, data_bundle, device: str) -> dict:
    return {
        "device": str(device),
        "best_checkpoint_path": results["best_checkpoint_path"],
        "best_checkpoint_metric": results["best_checkpoint_metric"],
        "best_checkpoint_score": results["best_checkpoint_score"],
        "best_checkpoint_epoch": results["best_checkpoint_epoch"],
        "parameter_summary": results["parameter_summary"],
        "num_train_examples": len(data_bundle.train_examples),
        "num_dev_examples": len(data_bundle.dev_examples),
        "num_test_examples": len(data_bundle.test_examples),
        "source_type": data_bundle.source_type,
        "source_vocab_path": data_bundle.source_vocab_path,
        "target_tokenizer_path": data_bundle.target_tokenizer_path,
    }


def main() -> int:
    args = parse_args()
    config = maybe_override_output_dir(load_config(args.config), args.output_dir)

    output_dir = ensure_dir(config["output_dir"])
    save_config(config, output_dir / "resolved_config.json")

    set_seed(config["seed"])
    device = get_device(args.device)
    data_bundle = prepare_data_bundle(config, output_dir)
    embedding_matrix = prepare_source_embedding_matrix(config, data_bundle, Path(output_dir))

    model = build_model(
        config=config,
        source_processor=data_bundle.source_processor,
        target_vocab_size=data_bundle.target_tokenizer.vocab_size,
        source_pad_id=data_bundle.source_pad_id,
        target_pad_id=data_bundle.target_pad_id,
        embedding_matrix=embedding_matrix,
    )

    results = train_model(
        model=model,
        data_bundle=data_bundle,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    run_summary = build_run_summary(results, data_bundle, str(device))
    save_json(run_summary, Path(output_dir) / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
