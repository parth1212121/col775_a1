#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nmt.config import load_config
from nmt.data import Vocabulary, load_or_build_glove_embedding_matrix, prepare_data_bundle
from nmt.model import build_model
from nmt.training import evaluate_generation
from nmt.utils import get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained translation checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding.")
    parser.add_argument("--device", default="auto", help="Device string, for example 'auto' or 'cuda'.")
    parser.add_argument("--output-json", default=None, help="Optional file path for saving evaluation results.")
    return parser.parse_args()


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


def select_evaluation_loader(data_bundle, split_name: str):
    if split_name == "dev":
        return data_bundle.dev_loader
    return data_bundle.test_loader


def build_evaluation_result(
    *,
    split_name: str,
    beam_size: int,
    metrics: dict,
    avg_decode_time: float,
    prediction_rows: list[dict],
    num_sample_predictions: int,
) -> dict:
    return {
        "split": split_name,
        "beam_size": beam_size,
        "metrics": metrics,
        "avg_decode_time_per_sentence_sec": avg_decode_time,
        "samples": prediction_rows[:num_sample_predictions],
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    device = get_device(args.device)

    data_bundle = prepare_data_bundle(config, output_dir)
    embedding_matrix = prepare_source_embedding_matrix(config, data_bundle, output_dir)

    model = build_model(
        config=config,
        source_processor=data_bundle.source_processor,
        target_vocab_size=data_bundle.target_tokenizer.vocab_size,
        source_pad_id=data_bundle.source_pad_id,
        target_pad_id=data_bundle.target_pad_id,
        embedding_matrix=embedding_matrix,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    evaluation_loader = select_evaluation_loader(data_bundle, args.split)
    metrics, prediction_rows, avg_decode_time = evaluate_generation(
        model,
        loader=evaluation_loader,
        device=device,
        tokenizer=data_bundle.target_tokenizer,
        beam_size=args.beam_size,
        max_decode_steps=config["evaluation"]["max_decode_steps"],
        length_penalty_alpha=config["evaluation"]["length_penalty_alpha"],
    )

    result = build_evaluation_result(
        split_name=args.split,
        beam_size=args.beam_size,
        metrics=metrics,
        avg_decode_time=avg_decode_time,
        prediction_rows=prediction_rows,
        num_sample_predictions=config["evaluation"]["num_sample_predictions"],
    )

    if args.output_json:
        save_json(result, args.output_json)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
