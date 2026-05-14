from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vision.constants import VALID_IMAGE_EXTENSIONS
from vision.runtime import build_model_from_checkpoint_metadata, build_transform_config_from_checkpoint
from vision.transforms import build_eval_transform
from vision.utils import ensure_dir, get_device, load_checkpoint


class InferenceImageDataset(Dataset):
    def __init__(self, data_root: Path, transform) -> None:
        self.data_root = data_root
        self.transform = transform
        self.image_paths = sorted(
            path
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found under {data_root}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        image_name = image_path.relative_to(self.data_root).as_posix()
        return image_tensor, image_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image classification predictions.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained .pth file.")
    parser.add_argument(
        "--norm_type",
        type=str,
        required=True,
        help="Normalization type: Baseline, NN, BN, IN, BIN, LN, or GN.",
    )
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the test-image directory.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to write predictions.csv into.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.model_path, map_location="cpu")
    checkpoint_metadata = checkpoint["metadata"]
    checkpoint_norm = str(checkpoint_metadata.get("norm", "")).lower()
    requested_norm = args.norm_type.strip().lower()
    if requested_norm == "baseline":
        requested_norm = "baseline"
    if checkpoint_norm and checkpoint_norm != requested_norm:
        raise ValueError(
            f"Checkpoint norm '{checkpoint_norm}' does not match requested norm_type '{args.norm_type}'."
        )

    cached_input_stats = checkpoint.get("input_stats")
    transform_config = build_transform_config_from_checkpoint(
        checkpoint_metadata=checkpoint_metadata,
        cached_input_stats=cached_input_stats,
    )
    eval_transform = build_eval_transform(transform_config)
    dataset = InferenceImageDataset(args.data_path, eval_transform)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(args.num_workers, 0),
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device(args.device)
    model = build_model_from_checkpoint_metadata(checkpoint_metadata)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state_dict is None:
        raise KeyError("Could not find model weights in checkpoint.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    prediction_rows: list[tuple[str, str]] = []
    with torch.inference_mode():
        for image_batch, image_names in data_loader:
            logits = model(image_batch.to(device, non_blocking=True))
            predicted_indices = logits.argmax(dim=1).cpu().tolist()
            for image_name, predicted_index in zip(image_names, predicted_indices):
                # The autograder expects zero-indexed class ids in the same
                # sorted-folder order used during training.
                prediction_rows.append((image_name, str(predicted_index)))

    output_dir = ensure_dir(args.save_dir)
    output_path = output_dir / "predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "predicted_label"])
        writer.writerows(prediction_rows)


if __name__ == "__main__":
    main()
