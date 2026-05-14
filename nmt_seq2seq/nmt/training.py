from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .decoding import generate_text_predictions
from .metrics import compute_mt_metrics
from .model import Seq2SeqModel
from .utils import count_parameters, ensure_dir, move_batch_to_device, save_json


# ------------------------------
# Checkpoint metric helpers
# ------------------------------

def normalize_checkpoint_metric_name(metric_name: str) -> str:
    aliases = {
        "loss": "dev_loss",
        "dev_loss": "dev_loss",
        "bleu": "dev_bleu",
        "dev_bleu": "dev_bleu",
        "chrf": "dev_chrf",
        "dev_chrf": "dev_chrf",
        "ter": "dev_ter",
        "dev_ter": "dev_ter",
    }
    normalized = aliases.get(metric_name.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported checkpoint metric '{metric_name}'. Supported values: {supported}")
    return normalized


def checkpoint_metric_higher_is_better(metric_name: str) -> bool:
    return metric_name in {"dev_bleu", "dev_chrf"}


def get_checkpoint_metric_value(metric_name: str, dev_loss: float, dev_metrics: dict[str, float]) -> float:
    if metric_name == "dev_loss":
        return dev_loss
    if metric_name == "dev_bleu":
        return float(dev_metrics["bleu"])
    if metric_name == "dev_chrf":
        return float(dev_metrics["chrf"])
    if metric_name == "dev_ter":
        return float(dev_metrics["ter"])
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def is_better_checkpoint_score(metric_name: str, candidate_score: float, best_score: float) -> bool:
    if checkpoint_metric_higher_is_better(metric_name):
        return candidate_score > best_score
    return candidate_score < best_score


# ------------------------------
# Teacher forcing helpers
# ------------------------------

def teacher_forcing_ratio_for_epoch(config: dict[str, Any], epoch_index: int) -> float:
    tf_config = config["train"]["teacher_forcing"]
    mode = tf_config["type"]
    if mode == "constant":
        return float(tf_config["value"])
    if mode == "inverse_sigmoid":
        k = float(tf_config["k"])
        return float(k / (k + math.exp((epoch_index + 1) / k)))
    raise ValueError(f"Unsupported teacher forcing schedule: {mode}")


def format_epoch_log_line(
    *,
    epoch_number: int,
    total_epochs: int,
    train_loss: float,
    dev_loss: float,
    dev_metrics: dict[str, float],
    teacher_forcing_ratio: float,
    checkpoint_metric_name: str,
    checkpoint_metric_score: float,
    is_best_checkpoint: bool,
) -> str:
    return (
        f"[epoch {epoch_number}/{total_epochs}] "
        f"train_loss={train_loss:.4f} "
        f"dev_loss={dev_loss:.4f} "
        f"dev_bleu={dev_metrics['bleu']:.4f} "
        f"dev_chrf={dev_metrics['chrf']:.4f} "
        f"dev_ter={dev_metrics['ter']:.4f} "
        f"teacher_forcing={teacher_forcing_ratio:.4f} "
        f"checkpoint_metric={checkpoint_metric_name} "
        f"checkpoint_score={checkpoint_metric_score:.4f} "
        f"best={'yes' if is_best_checkpoint else 'no'}"
    )


def build_epoch_summary(
    *,
    epoch_number: int,
    train_loss: float,
    dev_loss: float,
    dev_metrics: dict[str, float],
    checkpoint_metric_name: str,
    checkpoint_metric_score: float,
    is_best_checkpoint: bool,
) -> dict[str, Any]:
    return {
        "epoch": epoch_number,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_metrics": dev_metrics,
        "checkpoint_metric": checkpoint_metric_name,
        "checkpoint_score": checkpoint_metric_score,
        "is_best_checkpoint": is_best_checkpoint,
    }


# ------------------------------
# Core loss / evaluation helpers
# ------------------------------

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, label_smoothing: float) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def build_optimizer(model: Seq2SeqModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    learning_rate = config["train"]["learning_rate"]
    encoder_learning_rate = config["train"]["encoder_learning_rate"]
    weight_decay = config["train"]["weight_decay"]

    encoder_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.model."):
            encoder_params.append(parameter)
        else:
            other_params.append(parameter)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": learning_rate})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_learning_rate})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def maybe_load_checkpoint(
    model: Seq2SeqModel,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str | None,
    skip_prefixes: list[str],
    load_optimizer_state: bool,
    optimizer_device: torch.device | None,
) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    current_state = model.state_dict()
    filtered_state = {}

    for key, value in checkpoint["model_state"].items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in current_state and current_state[key].shape == value.shape:
            filtered_state[key] = value

    model.load_state_dict(filtered_state, strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None and load_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if optimizer_device is not None:
            move_optimizer_state_to_device(optimizer, optimizer_device)
    return checkpoint


def evaluate_loss(
    model: Seq2SeqModel,
    loader: torch.utils.data.DataLoader[dict[str, Any]],
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    losses = []

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(batch, teacher_forcing_ratio=0.0)
            loss = compute_loss(logits, batch["target_ids"][:, 1:], pad_id, label_smoothing=0.0)
            losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


def evaluate_generation(
    model: Seq2SeqModel,
    loader: torch.utils.data.DataLoader[dict[str, Any]],
    device: torch.device,
    tokenizer,
    beam_size: int,
    max_decode_steps: int,
    length_penalty_alpha: float,
) -> tuple[dict[str, float], list[dict[str, str]], float]:
    predicted_texts: list[str] = []
    reference_texts: list[str] = []
    prediction_rows: list[dict[str, str]] = []
    cumulative_decode_time = 0.0
    sentence_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch_predictions, average_sentence_time = generate_text_predictions(
            model,
            batch,
            tokenizer=tokenizer,
            beam_size=beam_size,
            max_decode_steps=max_decode_steps,
            length_penalty_alpha=length_penalty_alpha,
        )
        batch_references = batch["target_texts"]
        cumulative_decode_time += average_sentence_time * len(batch_predictions)
        sentence_count += len(batch_predictions)
        predicted_texts.extend(batch_predictions)
        reference_texts.extend(batch_references)
        for source_text, predicted_text, reference_text in zip(
            batch["source_texts"], batch_predictions, batch_references
        ):
            prediction_rows.append(
                {
                    "source": source_text,
                    "prediction": predicted_text,
                    "reference": reference_text,
                }
            )

    corpus_metrics = compute_mt_metrics(predicted_texts, reference_texts)
    average_sentence_time = cumulative_decode_time / max(1, sentence_count)
    return corpus_metrics, prediction_rows, average_sentence_time


# ------------------------------
# Plotting / checkpoint IO
# ------------------------------

def save_training_plots(history: dict[str, list[float]], output_dir: str | Path) -> None:
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["train_loss"], label="Train Loss", color="#005f73")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Train Loss")
    axis.grid(True, alpha=0.3)
    tf_axis = axis.twinx()
    tf_axis.plot(epochs, history["teacher_forcing_ratio"], label="TF Ratio", color="#ca6702", linestyle="--")
    tf_axis.set_ylabel("Teacher Forcing Ratio")
    fig.tight_layout()
    fig.savefig(plots_dir / "train_loss_vs_teacher_forcing.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["dev_loss"], label="Dev Loss", color="#0a9396")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Dev Loss")
    axis.grid(True, alpha=0.3)
    tf_axis = axis.twinx()
    tf_axis.plot(epochs, history["teacher_forcing_ratio"], label="TF Ratio", color="#ca6702", linestyle="--")
    tf_axis.set_ylabel("Teacher Forcing Ratio")
    fig.tight_layout()
    fig.savefig(plots_dir / "dev_loss_vs_teacher_forcing.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["dev_bleu"], label="BLEU", color="#005f73")
    axis.plot(epochs, history["dev_chrf"], label="chrF", color="#0a9396")
    axis.plot(epochs, history["dev_ter"], label="TER", color="#bb3e03")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Metric")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "dev_metrics.png", dpi=200)
    plt.close(fig)


def save_checkpoint(
    path: str | Path,
    model: Seq2SeqModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    history: dict[str, list[float]],
    metadata: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "history": history,
            "metadata": metadata,
        },
        target,
    )


# ------------------------------
# Training loop orchestration
# ------------------------------


def initialize_training_history() -> dict[str, list[float]]:
    return {
        "train_loss": [],
        "dev_loss": [],
        "dev_bleu": [],
        "dev_chrf": [],
        "dev_ter": [],
        "checkpoint_score": [],
        "teacher_forcing_ratio": [],
    }


def initial_best_checkpoint_score(checkpoint_metric_name: str) -> float:
    if checkpoint_metric_higher_is_better(checkpoint_metric_name):
        return float("-inf")
    return float("inf")


def run_single_training_epoch(
    *,
    model: Seq2SeqModel,
    train_loader: torch.utils.data.DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_pad_id: int,
    label_smoothing: float,
    grad_clip: float,
    teacher_forcing_ratio: float,
    epoch_number: int,
    total_epochs: int,
) -> float:
    model.train()
    per_batch_losses: list[float] = []
    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch_number}/{total_epochs}",
        leave=False,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = compute_loss(
            logits,
            batch["target_ids"][:, 1:],
            pad_id=target_pad_id,
            label_smoothing=label_smoothing,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        per_batch_losses.append(loss.item())
        progress.set_postfix(loss=f"{loss.item():.4f}", tf=f"{teacher_forcing_ratio:.3f}")

    return float(sum(per_batch_losses) / max(1, len(per_batch_losses)))


def run_single_validation_epoch(
    *,
    model: Seq2SeqModel,
    data_bundle,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, dict[str, float], list[dict[str, str]]]:
    dev_loss = evaluate_loss(model, data_bundle.dev_loader, device, data_bundle.target_pad_id)
    dev_metrics, dev_samples, _ = evaluate_generation(
        model,
        data_bundle.dev_loader,
        device,
        tokenizer=data_bundle.target_tokenizer,
        beam_size=config["train"]["eval_beam_size"],
        max_decode_steps=config["evaluation"]["max_decode_steps"],
        length_penalty_alpha=config["evaluation"]["length_penalty_alpha"],
    )
    return dev_loss, dev_metrics, dev_samples


def append_epoch_history(
    *,
    history: dict[str, list[float]],
    train_loss: float,
    dev_loss: float,
    dev_metrics: dict[str, float],
    checkpoint_metric_score: float,
    teacher_forcing_ratio: float,
) -> None:
    history["train_loss"].append(train_loss)
    history["dev_loss"].append(dev_loss)
    history["dev_bleu"].append(dev_metrics["bleu"])
    history["dev_chrf"].append(dev_metrics["chrf"])
    history["dev_ter"].append(dev_metrics["ter"])
    history["checkpoint_score"].append(checkpoint_metric_score)
    history["teacher_forcing_ratio"].append(teacher_forcing_ratio)


def save_epoch_outputs(
    *,
    output_path: Path,
    epoch_number: int,
    epoch_summary: dict[str, Any],
    dev_samples: list[dict[str, str]],
    num_sample_predictions: int,
) -> None:
    save_json(epoch_summary, output_path / f"epoch_{epoch_number:02d}_summary.json")
    save_json(
        dev_samples[:num_sample_predictions],
        output_path / "dev_samples_latest.json",
    )


def maybe_save_epoch_checkpoint(
    *,
    save_every_epoch: bool,
    checkpoints_dir: Path,
    epoch_number: int,
    model: Seq2SeqModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    history: dict[str, list[float]],
    parameter_summary: dict[str, int],
) -> None:
    if not save_every_epoch:
        return
    save_checkpoint(
        checkpoints_dir / f"epoch_{epoch_number:02d}.pt",
        model,
        optimizer,
        config,
        history,
        metadata={"parameter_summary": parameter_summary},
    )


def update_best_checkpoint_if_needed(
    *,
    is_best_checkpoint: bool,
    epoch_number: int,
    checkpoint_metric_name: str,
    checkpoint_metric_score: float,
    dev_loss: float,
    dev_metrics: dict[str, float],
    best_checkpoint_path: Path,
    model: Seq2SeqModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    history: dict[str, list[float]],
    parameter_summary: dict[str, int],
    output_path: Path,
) -> dict[str, Any]:
    if not is_best_checkpoint:
        return {}

    best_checkpoint_info = {
        "epoch": epoch_number,
        "checkpoint_metric": checkpoint_metric_name,
        "checkpoint_score": checkpoint_metric_score,
        "dev_loss": dev_loss,
        "dev_metrics": dev_metrics,
    }
    save_checkpoint(
        best_checkpoint_path,
        model,
        optimizer,
        config,
        history,
        metadata={
            "parameter_summary": parameter_summary,
            "best_checkpoint_info": best_checkpoint_info,
        },
    )
    save_json(best_checkpoint_info, output_path / "best_checkpoint_summary.json")
    return best_checkpoint_info


def evaluate_test_set_across_beams(
    *,
    model: Seq2SeqModel,
    data_bundle,
    config: dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    test_results_by_beam: dict[str, Any] = {}
    for beam_size in config["evaluation"]["beam_sizes"]:
        metrics, prediction_rows, avg_decode_time = evaluate_generation(
            model,
            data_bundle.test_loader,
            device,
            tokenizer=data_bundle.target_tokenizer,
            beam_size=beam_size,
            max_decode_steps=config["evaluation"]["max_decode_steps"],
            length_penalty_alpha=config["evaluation"]["length_penalty_alpha"],
        )
        beam_label = f"beam_{beam_size}"
        test_results_by_beam[beam_label] = {
            **metrics,
            "avg_decode_time_per_sentence_sec": avg_decode_time,
        }
        print(
            (
                f"[test {beam_label}] "
                f"bleu={metrics['bleu']:.4f} "
                f"chrf={metrics['chrf']:.4f} "
                f"ter={metrics['ter']:.4f} "
                f"avg_decode_time_per_sentence_sec={avg_decode_time:.6f}"
            ),
            flush=True,
        )
        save_json(
            prediction_rows[: config["evaluation"]["num_sample_predictions"]],
            output_path / f"test_samples_{beam_label}.json",
        )
    return test_results_by_beam


def train_model(
    model: Seq2SeqModel,
    data_bundle,
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_path / "checkpoints")
    optimizer = build_optimizer(model, config)
    model.to(device)

    maybe_load_checkpoint(
        model,
        optimizer,
        checkpoint_path=config["initialization"].get("pretrained_checkpoint_path"),
        skip_prefixes=config["initialization"].get("skip_prefixes", []),
        load_optimizer_state=config["initialization"].get("load_optimizer_state", False),
        optimizer_device=device,
    )
    parameter_summary = count_parameters(model)
    history = initialize_training_history()
    checkpoint_metric_name = normalize_checkpoint_metric_name(config["train"].get("checkpoint_metric", "bleu"))
    best_checkpoint_score = initial_best_checkpoint_score(checkpoint_metric_name)
    best_checkpoint_info: dict[str, Any] = {}
    best_checkpoint_path = checkpoints_dir / "best.pt"
    total_epochs = config["train"]["epochs"]

    for epoch_index in range(total_epochs):
        teacher_forcing_ratio = teacher_forcing_ratio_for_epoch(config, epoch_index)
        epoch_number = epoch_index + 1

        train_loss = run_single_training_epoch(
            model=model,
            train_loader=data_bundle.train_loader,
            optimizer=optimizer,
            device=device,
            target_pad_id=data_bundle.target_pad_id,
            label_smoothing=config["train"]["label_smoothing"],
            grad_clip=config["train"]["grad_clip"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            epoch_number=epoch_number,
            total_epochs=total_epochs,
        )
        dev_loss, dev_metrics, dev_samples = run_single_validation_epoch(
            model=model,
            data_bundle=data_bundle,
            config=config,
            device=device,
        )
        checkpoint_metric_score = get_checkpoint_metric_value(checkpoint_metric_name, dev_loss, dev_metrics)
        append_epoch_history(
            history=history,
            train_loss=train_loss,
            dev_loss=dev_loss,
            dev_metrics=dev_metrics,
            checkpoint_metric_score=checkpoint_metric_score,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        is_best_checkpoint = is_better_checkpoint_score(
            checkpoint_metric_name,
            checkpoint_metric_score,
            best_checkpoint_score,
        )

        print(
            format_epoch_log_line(
                epoch_number=epoch_number,
                total_epochs=total_epochs,
                train_loss=train_loss,
                dev_loss=dev_loss,
                dev_metrics=dev_metrics,
                teacher_forcing_ratio=teacher_forcing_ratio,
                checkpoint_metric_name=checkpoint_metric_name,
                checkpoint_metric_score=checkpoint_metric_score,
                is_best_checkpoint=is_best_checkpoint,
            ),
            flush=True,
        )

        epoch_summary = build_epoch_summary(
            epoch_number=epoch_number,
            train_loss=train_loss,
            dev_loss=dev_loss,
            dev_metrics=dev_metrics,
            checkpoint_metric_name=checkpoint_metric_name,
            checkpoint_metric_score=checkpoint_metric_score,
            is_best_checkpoint=is_best_checkpoint,
        )
        save_epoch_outputs(
            output_path=output_path,
            epoch_number=epoch_number,
            epoch_summary=epoch_summary,
            dev_samples=dev_samples,
            num_sample_predictions=config["evaluation"]["num_sample_predictions"],
        )
        maybe_save_epoch_checkpoint(
            save_every_epoch=config["train"].get("save_every_epoch", False),
            checkpoints_dir=checkpoints_dir,
            epoch_number=epoch_number,
            model=model,
            optimizer=optimizer,
            config=config,
            history=history,
            parameter_summary=parameter_summary,
        )

        if is_best_checkpoint:
            best_checkpoint_score = checkpoint_metric_score
            best_checkpoint_info = update_best_checkpoint_if_needed(
                is_best_checkpoint=is_best_checkpoint,
                epoch_number=epoch_number,
                checkpoint_metric_name=checkpoint_metric_name,
                checkpoint_metric_score=checkpoint_metric_score,
                dev_loss=dev_loss,
                dev_metrics=dev_metrics,
                best_checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                history=history,
                parameter_summary=parameter_summary,
                output_path=output_path,
            )

    save_training_plots(history, output_path)
    save_json(history, output_path / "history.json")

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])

    test_results_by_beam = evaluate_test_set_across_beams(
        model=model,
        data_bundle=data_bundle,
        config=config,
        device=device,
        output_path=output_path,
    )

    save_json(test_results_by_beam, output_path / "test_metrics.json")

    return {
        "history": history,
        "test_metrics": test_results_by_beam,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_metric": checkpoint_metric_name,
        "best_checkpoint_score": best_checkpoint_score,
        "best_checkpoint_epoch": best_checkpoint_info.get("epoch"),
        "parameter_summary": parameter_summary,
    }
