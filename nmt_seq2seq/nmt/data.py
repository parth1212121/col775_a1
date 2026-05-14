from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import zipfile
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .utils import ensure_dir


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


@dataclass
class TextExample:
    source: str
    target: str


class Vocabulary:
    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for token, index in stoi.items():
            self.itos[index] = token

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.itos, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            tokens = json.load(handle)
        stoi = {token: index for index, token in enumerate(tokens)}
        return cls(stoi)

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
        lowercase: bool = False,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize_words(text, lowercase=lowercase)
            counter.update(tokens)

        ordered_tokens = [
            token
            for token, count in counter.most_common()
            if count >= min_freq and token not in SPECIAL_TOKENS
        ]
        if max_vocab_size is not None:
            ordered_tokens = ordered_tokens[: max(0, max_vocab_size - len(SPECIAL_TOKENS))]

        stoi = {token: index for index, token in enumerate(SPECIAL_TOKENS + ordered_tokens)}
        return cls(stoi)


def tokenize_words(text: str, lowercase: bool = False) -> list[str]:
    normalized_text = text.lower() if lowercase else text
    return normalized_text.strip().split()


def clean_text(text: str, cleanup_config: dict[str, Any], is_source: bool) -> str:
    cleaned_text = str(text).replace("\u200b", " ").replace("\ufeff", " ").strip()
    if cleanup_config.get("strip_translate_prompt", False) and is_source:
        cleaned_text = re.sub(
            r"^\s*translate to [^:]+:\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE,
        )
    if cleanup_config.get("normalize_whitespace", True):
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    if is_source and cleanup_config.get("lowercase_source", False):
        cleaned_text = cleaned_text.lower()
    if not is_source and cleanup_config.get("lowercase_target", False):
        cleaned_text = cleaned_text.lower()
    return cleaned_text


def detect_parallel_columns(
    frame: pd.DataFrame,
    source_column: str | None,
    target_column: str | None,
) -> tuple[str, str]:
    if source_column and target_column:
        return source_column, target_column

    columns = list(frame.columns)
    normalized_column_lookup = {column.lower(): column for column in columns}

    target_candidates = ["translated", "target", "tgt", "label", "reference", "output"]
    source_candidates = ["source", "src", "input", "sentence", "text", "prompt", "english"]

    detected_target = target_column
    detected_source = source_column

    if detected_target is None:
        for candidate in target_candidates:
            if candidate in normalized_column_lookup:
                detected_target = normalized_column_lookup[candidate]
                break

    if detected_source is None:
        for candidate in source_candidates:
            if candidate in normalized_column_lookup:
                detected_source = normalized_column_lookup[candidate]
                break

    if detected_target is None and len(columns) >= 2:
        detected_target = columns[1]
    if detected_source is None and len(columns) >= 1:
        fallback_candidates = [column for column in columns if column != detected_target]
        if fallback_candidates:
            detected_source = fallback_candidates[0]

    if detected_source is None or detected_target is None:
        raise ValueError(
            "Could not detect source/target columns automatically. "
            "Please set data.source_column and data.target_column in the config."
        )

    return detected_source, detected_target


def load_parallel_data(
    path: str | Path,
    source_column: str | None,
    target_column: str | None,
    cleanup_config: dict[str, Any],
    csv_sep: str = ",",
    csv_encoding: str = "utf-8",
    drop_duplicates: bool = False,
) -> list[TextExample]:
    data_frame = pd.read_csv(
        path,
        sep=csv_sep,
        encoding=csv_encoding,
        keep_default_na=False,
        engine="python",
        on_bad_lines="warn",
    )
    source_column, target_column = detect_parallel_columns(data_frame, source_column, target_column)

    parallel_frame = data_frame[[source_column, target_column]].copy()
    parallel_frame.columns = ["source", "target"]
    parallel_frame["source"] = parallel_frame["source"].astype(str)
    parallel_frame["target"] = parallel_frame["target"].astype(str)
    if drop_duplicates:
        parallel_frame = parallel_frame.drop_duplicates()

    examples: list[TextExample] = []
    for _, row in parallel_frame.iterrows():
        source_text = clean_text(row["source"], cleanup_config, is_source=True)
        target_text = clean_text(row["target"], cleanup_config, is_source=False)
        if source_text and target_text:
            examples.append(TextExample(source=source_text, target=target_text))
    return examples


def train_or_load_target_tokenizer(
    train_targets: list[str],
    output_dir: str | Path,
    vocab_size: int,
    reuse_tokenizer_path: str | None = None,
) -> PreTrainedTokenizerFast:
    if reuse_tokenizer_path:
        return PreTrainedTokenizerFast.from_pretrained(reuse_tokenizer_path)

    output_path = Path(output_dir)
    if (output_path / "tokenizer.json").exists():
        return PreTrainedTokenizerFast.from_pretrained(output_path)

    ensure_dir(output_path)
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS, vocab_size=vocab_size)
    tokenizer.train_from_iterator(train_targets, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        bos_token=SOS_TOKEN,
        eos_token=EOS_TOKEN,
    )
    fast_tokenizer.save_pretrained(output_path)
    return fast_tokenizer


def build_or_load_source_vocab(
    train_sources: list[str],
    vocab_path: str | Path,
    min_freq: int,
    max_vocab_size: int | None,
    lowercase: bool,
) -> Vocabulary:
    vocab_file = Path(vocab_path)
    if vocab_file.exists():
        return Vocabulary.load(vocab_file)

    vocab = Vocabulary.build(
        train_sources,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        lowercase=lowercase,
    )
    vocab.save(vocab_file)
    return vocab


def build_glove_embedding_matrix(
    vocab: Vocabulary,
    embedding_dim: int,
    glove_path: str | Path,
) -> torch.Tensor:
    if not glove_path:
        raise ValueError("source.glove_path must be set for GloVe-based experiments.")

    glove_path = Path(glove_path)
    glove_lookup: dict[str, np.ndarray] = {}
    needed_tokens = set(vocab.stoi.keys())
    if glove_path.suffix == ".zip":
        member_name = f"glove.6B.{embedding_dim}d.txt"
        with zipfile.ZipFile(glove_path) as archive:
            archive_names = set(archive.namelist())
            if member_name not in archive_names:
                raise ValueError(
                    f"Could not find {member_name} inside {glove_path}. "
                    f"Available members include: {sorted(list(archive_names))[:10]}"
                )
            with archive.open(member_name, "r") as raw_handle:
                for raw_line in raw_handle:
                    parts = raw_line.decode("utf-8").rstrip().split()
                    if not parts:
                        continue
                    token = parts[0]
                    if token in needed_tokens:
                        vector = np.asarray(parts[1:], dtype=np.float32)
                        if vector.shape[0] == embedding_dim:
                            glove_lookup[token] = vector
    else:
        with glove_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip().split()
                if not parts:
                    continue
                token = parts[0]
                if token in needed_tokens:
                    vector = np.asarray(parts[1:], dtype=np.float32)
                    if vector.shape[0] == embedding_dim:
                        glove_lookup[token] = vector

    matrix = np.random.normal(0.0, 0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab.pad_id] = np.zeros(embedding_dim, dtype=np.float32)
    for token, index in vocab.stoi.items():
        if token in glove_lookup:
            matrix[index] = glove_lookup[token]

    return torch.tensor(matrix, dtype=torch.float32)


def load_or_build_glove_embedding_matrix(
    vocab: Vocabulary,
    embedding_dim: int,
    glove_path: str | Path,
    cache_path: str | Path,
) -> torch.Tensor:
    target = Path(cache_path)
    if target.exists():
        return torch.load(target, map_location="cpu")

    matrix = build_glove_embedding_matrix(vocab, embedding_dim=embedding_dim, glove_path=glove_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(matrix, target)
    return matrix


class ParallelDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: list[TextExample],
        source_type: str,
        source_processor: Vocabulary | PreTrainedTokenizerFast,
        target_tokenizer: PreTrainedTokenizerFast,
        source_max_length: int,
        target_max_length: int,
        source_lowercase: bool,
    ) -> None:
        self.examples = examples
        self.source_type = source_type
        self.source_processor = source_processor
        self.target_tokenizer = target_tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
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

        target_encoded = self.target_tokenizer(
            example.target,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.target_max_length - 2),
        )
        target_ids = [
            self.target_tokenizer.bos_token_id,
            *target_encoded["input_ids"],
            self.target_tokenizer.eos_token_id,
        ]

        return {
            "source_ids": source_ids,
            "source_attention_mask": source_attention_mask,
            "target_ids": target_ids,
            "source_text": example.source,
            "target_text": example.target,
        }


def build_collate_fn(source_pad_id: int, target_pad_id: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_source = max(len(item["source_ids"]) for item in batch)
        max_target = max(len(item["target_ids"]) for item in batch)

        source_ids = torch.full((batch_size, max_source), source_pad_id, dtype=torch.long)
        source_attention_mask = torch.zeros((batch_size, max_source), dtype=torch.long)
        target_ids = torch.full((batch_size, max_target), target_pad_id, dtype=torch.long)

        source_lengths = []
        target_lengths = []

        for row, item in enumerate(batch):
            src_len = len(item["source_ids"])
            tgt_len = len(item["target_ids"])

            source_ids[row, :src_len] = torch.tensor(item["source_ids"], dtype=torch.long)
            source_attention_mask[row, :src_len] = torch.tensor(item["source_attention_mask"], dtype=torch.long)
            target_ids[row, :tgt_len] = torch.tensor(item["target_ids"], dtype=torch.long)

            source_lengths.append(src_len)
            target_lengths.append(tgt_len)

        return {
            "source_ids": source_ids,
            "source_attention_mask": source_attention_mask,
            "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
            "target_ids": target_ids,
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "source_texts": [item["source_text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
        }

    return collate


@dataclass
class DataBundle:
    train_examples: list[TextExample]
    dev_examples: list[TextExample]
    test_examples: list[TextExample]
    train_loader: DataLoader[dict[str, Any]]
    dev_loader: DataLoader[dict[str, Any]]
    test_loader: DataLoader[dict[str, Any]]
    target_tokenizer: PreTrainedTokenizerFast
    source_processor: Vocabulary | PreTrainedTokenizerFast
    source_pad_id: int
    target_pad_id: int
    source_type: str
    source_vocab_path: str | None
    target_tokenizer_path: str


def prepare_data_bundle(config: dict[str, Any], output_dir: str | Path) -> DataBundle:
    data_config = config["data"]
    source_config = config["source"]
    target_config = config["target"]

    train_examples = load_parallel_data(
        data_config["train_path"],
        source_column=data_config["source_column"],
        target_column=data_config["target_column"],
        cleanup_config=data_config["cleanup"],
        csv_sep=data_config["csv_sep"],
        csv_encoding=data_config["csv_encoding"],
        drop_duplicates=data_config["drop_duplicates"],
    )
    dev_examples = load_parallel_data(
        data_config["dev_path"],
        source_column=data_config["source_column"],
        target_column=data_config["target_column"],
        cleanup_config=data_config["cleanup"],
        csv_sep=data_config["csv_sep"],
        csv_encoding=data_config["csv_encoding"],
        drop_duplicates=False,
    )
    test_examples = load_parallel_data(
        data_config["test_path"],
        source_column=data_config["source_column"],
        target_column=data_config["target_column"],
        cleanup_config=data_config["cleanup"],
        csv_sep=data_config["csv_sep"],
        csv_encoding=data_config["csv_encoding"],
        drop_duplicates=False,
    )

    artifacts_dir = ensure_dir(Path(output_dir) / "artifacts")
    tokenizer_dir = ensure_dir(artifacts_dir / "target_tokenizer")
    target_tokenizer = train_or_load_target_tokenizer(
        train_targets=[example.target for example in train_examples],
        output_dir=tokenizer_dir,
        vocab_size=target_config["vocab_size"],
        reuse_tokenizer_path=target_config.get("reuse_tokenizer_path"),
    )
    target_tokenizer_path = target_config.get("reuse_tokenizer_path") or str(tokenizer_dir)
    source_type = source_config["type"]

    if source_type == "word":
        source_vocab_path = artifacts_dir / "source_vocab.json"
        source_processor = build_or_load_source_vocab(
            train_sources=[example.source for example in train_examples],
            vocab_path=source_vocab_path,
            min_freq=source_config["min_freq"],
            max_vocab_size=source_config["max_vocab_size"],
            lowercase=source_config["lowercase"],
        )
        source_pad_id = source_processor.pad_id
    elif source_type == "bert":
        source_processor = AutoTokenizer.from_pretrained(config["model"]["bert_model_name"])
        source_pad_id = source_processor.pad_token_id
        source_vocab_path = None
    else:
        raise ValueError(f"Unsupported source.type: {source_type}")

    train_dataset = ParallelDataset(
        train_examples,
        source_type=source_type,
        source_processor=source_processor,
        target_tokenizer=target_tokenizer,
        source_max_length=source_config["max_length"],
        target_max_length=target_config["max_length"],
        source_lowercase=source_config["lowercase"],
    )
    dev_dataset = ParallelDataset(
        dev_examples,
        source_type=source_type,
        source_processor=source_processor,
        target_tokenizer=target_tokenizer,
        source_max_length=source_config["max_length"],
        target_max_length=target_config["max_length"],
        source_lowercase=source_config["lowercase"],
    )
    test_dataset = ParallelDataset(
        test_examples,
        source_type=source_type,
        source_processor=source_processor,
        target_tokenizer=target_tokenizer,
        source_max_length=source_config["max_length"],
        target_max_length=target_config["max_length"],
        source_lowercase=source_config["lowercase"],
    )

    collate_fn = build_collate_fn(source_pad_id=source_pad_id, target_pad_id=target_tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    return DataBundle(
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        target_tokenizer=target_tokenizer,
        source_processor=source_processor,
        source_pad_id=source_pad_id,
        target_pad_id=target_tokenizer.pad_token_id,
        source_type=source_type,
        source_vocab_path=str(source_vocab_path) if source_vocab_path is not None else None,
        target_tokenizer_path=target_tokenizer_path,
    )
