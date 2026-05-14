from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "translation_experiment",
    "seed": 42,
    "output_dir": "outputs/translation_experiment",
    "data": {
        "train_path": "",
        "dev_path": "",
        "test_path": "",
        "source_column": None,
        "target_column": None,
        "drop_duplicates": False,
        "csv_sep": ",",
        "csv_encoding": "utf-8",
        "cleanup": {
            "strip_translate_prompt": True,
            "normalize_whitespace": True,
            "lowercase_source": False,
            "lowercase_target": False,
        },
    },
    "source": {
        "type": "word",
        "max_length": 128,
        "min_freq": 1,
        "max_vocab_size": 40000,
        "embedding_dim": 100,
        "glove_path": "",
        "lowercase": True,
    },
    "target": {
        "type": "bpe",
        "max_length": 128,
        "vocab_size": 5000,
        "reuse_tokenizer_path": None,
    },
    "model": {
        "encoder_type": "glove_lstm",
        "bert_model_name": "bert-base-cased",
        "freeze_encoder": False,
        "use_attention": True,
        "encoder_hidden_size": 256,
        "encoder_layers": 1,
        "decoder_hidden_size": 512,
        "target_embedding_dim": 256,
        "dropout": 0.2,
    },
    "train": {
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 1e-3,
        "encoder_learning_rate": 2e-5,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
        "num_workers": 0,
        "checkpoint_metric": "bleu",
        "teacher_forcing": {
            "type": "constant",
            "value": 0.6,
            "k": 8,
        },
        "eval_beam_size": 1,
        "save_every_epoch": False,
    },
    "evaluation": {
        "beam_sizes": [1],
        "max_decode_steps": 128,
        "length_penalty_alpha": 0.7,
        "num_sample_predictions": 20,
    },
    "initialization": {
        "pretrained_checkpoint_path": None,
        "skip_prefixes": [],
        "load_optimizer_state": False,
    },
}


# ------------------------------
# Merge / path resolution helpers
# ------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_value_from_roots(key_name: str, raw_value: Any, candidate_roots: list[Path]) -> Any:
    if not isinstance(raw_value, str) or raw_value == "":
        return raw_value

    candidate_path = Path(raw_value)
    if candidate_path.is_absolute():
        return str(candidate_path)

    looks_like_path = "path" in key_name or "dir" in key_name or raw_value.endswith(
        (".csv", ".json", ".pt", ".txt")
    )
    if not looks_like_path:
        return raw_value

    for root_dir in candidate_roots:
        resolved_candidate = (root_dir / candidate_path).resolve()
        if resolved_candidate.exists():
            return str(resolved_candidate)

    return str((candidate_roots[0] / candidate_path).resolve())


def _resolve_section_paths(section_config: dict[str, Any], candidate_roots: list[Path]) -> dict[str, Any]:
    resolved_section = deepcopy(section_config)
    for key_name, raw_value in resolved_section.items():
        if isinstance(raw_value, dict):
            for nested_key_name, nested_raw_value in raw_value.items():
                raw_value[nested_key_name] = _resolve_value_from_roots(
                    nested_key_name,
                    nested_raw_value,
                    candidate_roots,
                )
        else:
            resolved_section[key_name] = _resolve_value_from_roots(
                key_name,
                raw_value,
                candidate_roots,
            )
    return resolved_section


def _resolve_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    config_dir = config_path.parent
    cwd = Path.cwd()
    candidate_roots = [cwd, config_dir]

    if config_dir.name == "configs":
        candidate_roots.insert(0, config_dir.parent)

    for section in ("data", "source", "target", "initialization"):
        resolved[section] = _resolve_section_paths(resolved.get(section, {}), candidate_roots)

    output_dir = Path(resolved["output_dir"])
    if not output_dir.is_absolute():
        resolved["output_dir"] = str((candidate_roots[0] / output_dir).resolve())

    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    return _resolve_paths(merged, config_path)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
