from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


NORM_NAME_MAP = {
    "baseline": "baseline",
    "nn": "nn",
    "bn": "bn",
    "in": "in",
    "bin": "bin",
    "ln": "ln",
    "gn": "gn",
}

CANONICAL_NORM_NAMES = {
    "baseline": "Baseline",
    "nn": "NN",
    "bn": "BN",
    "in": "IN",
    "bin": "BIN",
    "ln": "LN",
    "gn": "GN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet image classification model.")
    parser.add_argument("--train_path", type=Path, required=True, help="Path to the training dataset directory.")
    parser.add_argument("--val_path", type=Path, required=True, help="Path to the validation dataset directory.")
    parser.add_argument(
        "--norm_type",
        type=str,
        required=True,
        choices=["Baseline", "NN", "BN", "IN", "BIN", "LN", "GN"],
        help="Normalization variant to train.",
    )
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save the resulting model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_norm = NORM_NAME_MAP[args.norm_type.strip().lower()]
    canonical_norm = CANONICAL_NORM_NAMES[normalized_norm]
    save_dir = args.save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / f"{canonical_norm}_training_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vision_resnet_") as temp_root_str:
        temp_root = Path(temp_root_str)
        (temp_root / "train").symlink_to(args.train_path.resolve(), target_is_directory=True)
        (temp_root / "val").symlink_to(args.val_path.resolve(), target_is_directory=True)

        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "train.py"),
            "--data-root",
            str(temp_root),
            "--output-dir",
            str(run_dir),
            "--norm",
            normalized_norm,
            "--epochs",
            "100",
            "--batch-size",
            "128",
            "--eval-batch-size",
            "128",
            "--lr",
            "0.07",
            "--amp",
            "--train-resize-mode",
            "resnet_paper",
            "--train-resize-max",
            "480",
            "--autoaugment-policy",
            "imagenet",
            "--label-smoothing",
            "0.1",
            "--dropout-prob",
            "0.3",
        ]
        subprocess.run(command, check=True)

    best_checkpoint = run_dir / "checkpoint_best.pt"
    if not best_checkpoint.exists():
        best_checkpoint = run_dir / "checkpoint_last.pt"
    if not best_checkpoint.exists():
        raise FileNotFoundError("Training completed but no checkpoint file was produced.")

    output_checkpoint = save_dir / f"{canonical_norm}.pth"
    shutil.copy2(best_checkpoint, output_checkpoint)
    print(f"Saved trained model to {output_checkpoint}")


if __name__ == "__main__":
    main()
