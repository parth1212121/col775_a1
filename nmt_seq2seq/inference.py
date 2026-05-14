from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedTokenizerFast

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nmt.config import load_config
from nmt.data import (
    Vocabulary,
    build_collate_fn,
    clean_text,
    tokenize_words,
)
from nmt.decoding import generate_text_predictions
from nmt.model import EncoderState, LSTMDecoder, Seq2SeqModel, build_model
from nmt.utils import get_device


MODE_TO_CONFIG = {
    1: "en_hi_glove_seq2seq_final40.json",
    2: "en_hi_glove_attn_final40.json",
    3: "en_hi_bert_frozen_attn_final40.json",
    4: "en_hi_bert_finetune_attn_final40.json",
}

BERT_ASSET_DIR = ROOT_DIR / "hf_assets" / "bert-base-cased"


@dataclass
class InferenceExample:
    source: str


class InferenceDataset(Dataset):
    def __init__(
        self,
        examples: list[InferenceExample],
        *,
        source_type: str,
        source_processor: Vocabulary | PreTrainedTokenizerFast,
        source_max_length: int,
        source_lowercase: bool,
    ) -> None:
        self.examples = examples
        self.source_type = source_type
        self.source_processor = source_processor
        self.source_max_length = source_max_length
        self.source_lowercase = source_lowercase

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if self.source_type == "word":
            assert isinstance(self.source_processor, Vocabulary)
            source_tokens = tokenize_words(example.source, lowercase=self.source_lowercase)
            source_ids = self.source_processor.encode(source_tokens[: self.source_max_length])
            source_attention_mask = [1] * len(source_ids)
        else:
            assert isinstance(self.source_processor, PreTrainedTokenizerFast)
            source_encoded = self.source_processor(
                example.source,
                add_special_tokens=True,
                truncation=True,
                max_length=self.source_max_length,
            )
            source_ids = source_encoded["input_ids"]
            source_attention_mask = source_encoded["attention_mask"]

        return {
            "source_ids": source_ids,
            "source_attention_mask": source_attention_mask,
            "target_ids": [0],
            "source_text": example.source,
            "target_text": "",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural machine translation inference entry point.")
    parser.add_argument("--input", required=True, help="CSV input file with a 'source' column.")
    parser.add_argument("--output", required=True, help="CSV output file path.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing model_1.pth ... model_4.pth.")
    parser.add_argument("--mode", type=int, required=True, choices=[1, 2, 3, 4], help="Model id to use.")
    parser.add_argument(
        "--decoding_strategy",
        required=True,
        choices=["greedy", "beam"],
        help="Use greedy decoding or beam search (beam size 5).",
    )
    parser.add_argument("--device", default="auto", help="Optional device override.")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    return parser.parse_args()


def load_target_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))


def resolve_artifacts(mode: int, checkpoint_dir: Path) -> dict[str, Path | None]:
    artifact_candidates = [
        checkpoint_dir / "artifacts" / f"model_{mode}",
        checkpoint_dir / f"model_{mode}",
        checkpoint_dir / "artifacts",
        ROOT_DIR / "artifacts" / f"model_{mode}",
    ]
    artifact_root = next(
        (candidate for candidate in artifact_candidates if (candidate / "target_tokenizer").exists()),
        artifact_candidates[0],
    )
    paths: dict[str, Path | None] = {
        "artifact_root": artifact_root,
        "target_tokenizer_dir": artifact_root / "target_tokenizer",
        "source_vocab_path": artifact_root / "source_vocab.json",
        "source_embedding_matrix_path": artifact_root / "source_embedding_matrix.pt",
    }
    if not paths["source_vocab_path"].exists():
        paths["source_vocab_path"] = None
    if not paths["source_embedding_matrix_path"].exists():
        paths["source_embedding_matrix_path"] = None
    return paths


def load_source_processor(config: dict, artifact_paths: dict[str, Path | None]):
    source_config = config["source"]
    if source_config["type"] == "word":
        source_vocab_path = artifact_paths["source_vocab_path"]
        if source_vocab_path is None:
            raise FileNotFoundError("Bundled source vocabulary is required for word-based models.")
        source_vocab = Vocabulary.load(source_vocab_path)
        return source_vocab, source_vocab.pad_id

    tokenizer_name = str(BERT_ASSET_DIR) if BERT_ASSET_DIR.exists() else config["model"]["bert_model_name"]
    source_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return source_tokenizer, source_tokenizer.pad_token_id


def load_embedding_matrix_if_needed(config: dict, artifact_paths: dict[str, Path | None]):
    if config["source"]["type"] != "word":
        return None
    matrix_path = artifact_paths["source_embedding_matrix_path"]
    if matrix_path is not None and matrix_path.exists():
        return torch.load(matrix_path, map_location="cpu")

    source_vocab_path = artifact_paths["source_vocab_path"]
    if source_vocab_path is None:
        raise FileNotFoundError("Bundled source vocabulary is required for GloVe models.")

    source_vocab = Vocabulary.load(source_vocab_path)
    embedding_dim = int(config["source"]["embedding_dim"])
    return torch.zeros((len(source_vocab), embedding_dim), dtype=torch.float32)


class OfflineBertEncoder(torch.nn.Module):
    def __init__(self, asset_dir: Path, freeze: bool) -> None:
        super().__init__()
        bert_config = BertConfig.from_pretrained(str(asset_dir))
        self.model = BertModel(bert_config)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.output_size = self.model.config.hidden_size

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> EncoderState:
        del source_lengths
        encoder_outputs = self.model(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
        ).last_hidden_state
        mask = source_attention_mask.bool()
        masked_outputs = encoder_outputs * source_attention_mask.unsqueeze(-1)
        pooled_context = masked_outputs.sum(dim=1) / source_attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return EncoderState(
            outputs=encoder_outputs,
            mask=mask,
            summary_hidden=pooled_context,
            summary_cell=pooled_context,
            fixed_context=pooled_context,
        )


def build_inference_model(
    *,
    config: dict[str, Any],
    source_processor: Vocabulary | PreTrainedTokenizerFast,
    target_tokenizer: PreTrainedTokenizerFast,
    source_pad_id: int,
    embedding_matrix: torch.Tensor | None,
) -> Seq2SeqModel:
    if config["source"]["type"] == "word":
        return build_model(
            config=config,
            source_processor=source_processor,
            target_vocab_size=target_tokenizer.vocab_size,
            source_pad_id=source_pad_id,
            target_pad_id=target_tokenizer.pad_token_id,
            embedding_matrix=embedding_matrix,
        )

    model_config = config["model"]
    if not BERT_ASSET_DIR.exists():
        return build_model(
            config=config,
            source_processor=source_processor,
            target_vocab_size=target_tokenizer.vocab_size,
            source_pad_id=source_pad_id,
            target_pad_id=target_tokenizer.pad_token_id,
            embedding_matrix=None,
        )

    encoder = OfflineBertEncoder(BERT_ASSET_DIR, freeze=model_config["freeze_encoder"])
    decoder = LSTMDecoder(
        vocab_size=target_tokenizer.vocab_size,
        embedding_dim=model_config["target_embedding_dim"],
        hidden_size=model_config["decoder_hidden_size"],
        context_dim=encoder.output_size,
        use_attention=model_config["use_attention"],
        pad_id=target_tokenizer.pad_token_id,
        dropout=model_config["dropout"],
    )
    return Seq2SeqModel(
        encoder=encoder,
        decoder=decoder,
        encoder_state_dim=encoder.output_size,
        decoder_hidden_size=model_config["decoder_hidden_size"],
        target_pad_id=target_tokenizer.pad_token_id,
    )


def resolve_checkpoint_path(checkpoint_dir: Path, mode: int) -> Path:
    candidates = [
        checkpoint_dir / f"model_{mode}.pth",
        checkpoint_dir / f"model_{mode}.pt",
        checkpoint_dir / "checkpoint_best.pt",
        checkpoint_dir / "checkpoint_last.pt",
        checkpoint_dir / f"model_{mode}" / f"model_{mode}.pth",
        checkpoint_dir / f"model_{mode}" / f"model_{mode}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find model_{mode}.pth or model_{mode}.pt inside {checkpoint_dir}."
    )


def build_examples(frame: pd.DataFrame, cleanup_config: dict[str, Any]) -> list[InferenceExample]:
    if "source" not in frame.columns:
        raise ValueError("Input CSV must contain a column named 'source'.")
    examples: list[InferenceExample] = []
    for source_text in frame["source"].astype(str).tolist():
        cleaned_source = clean_text(source_text, cleanup_config, is_source=True)
        examples.append(InferenceExample(source=cleaned_source))
    return examples


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    device_batch = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            device_batch[key] = value.to(device)
        else:
            device_batch[key] = value
    return device_batch


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    config_path = ROOT_DIR / "configs" / MODE_TO_CONFIG[args.mode]
    config = load_config(config_path)
    artifact_paths = resolve_artifacts(args.mode, checkpoint_dir)
    target_tokenizer = load_target_tokenizer(artifact_paths["target_tokenizer_dir"])
    source_processor, source_pad_id = load_source_processor(config, artifact_paths)
    embedding_matrix = load_embedding_matrix_if_needed(config, artifact_paths)

    model = build_inference_model(
        config=config,
        source_processor=source_processor,
        source_pad_id=source_pad_id,
        target_tokenizer=target_tokenizer,
        embedding_matrix=embedding_matrix,
    )

    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.mode)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(model_state)

    device = get_device(args.device)
    model.to(device)
    model.eval()

    input_frame = pd.read_csv(args.input)
    examples = build_examples(input_frame, config["data"]["cleanup"])
    dataset = InferenceDataset(
        examples,
        source_type=config["source"]["type"],
        source_processor=source_processor,
        source_max_length=config["source"]["max_length"],
        source_lowercase=config["source"].get("lowercase", False),
    )
    collate_fn = build_collate_fn(source_pad_id=source_pad_id, target_pad_id=0)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    beam_size = 5 if args.decoding_strategy == "beam" else 1
    predicted_texts: list[str] = []
    for batch in data_loader:
        prediction_batch, _ = generate_text_predictions(
            model=model,
            batch=move_batch_to_device(batch, device),
            tokenizer=target_tokenizer,
            beam_size=beam_size,
            max_decode_steps=config["evaluation"]["max_decode_steps"],
            length_penalty_alpha=config["evaluation"]["length_penalty_alpha"],
        )
        predicted_texts.extend(prediction_batch)

    output_frame = input_frame.copy()
    output_frame["translated"] = predicted_texts
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
