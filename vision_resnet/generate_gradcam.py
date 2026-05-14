"""Generate Grad-CAM visualizations for a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision.datasets import build_dataloaders
from vision.runtime import (
    build_data_loader_config_from_args,
    build_model_from_checkpoint_metadata,
    build_transform_config_from_checkpoint,
)
from vision.transforms import (
    build_eval_transform,
    build_tensor_transform,
    build_train_transform,
    build_visual_transform,
)
from vision.utils import ensure_dir, get_device, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--target-layer", type=str, default="layer4.1.conv2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names. If omitted, the script auto-selects valid classes.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--disable-pin-memory", action="store_true")
    parser.add_argument("--disable-persistent-workers", action="store_true")
    return parser.parse_args()


def resolve_module_by_name(module: torch.nn.Module, path: str) -> torch.nn.Module:
    current: torch.nn.Module = module
    for token in path.split("."):
        if token.isdigit():
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def collect_predictions(model, loader, device: torch.device) -> list[dict[str, object]]:
    model.eval()
    prediction_records: list[dict[str, object]] = []
    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()
            for path, target, prediction in zip(paths, targets.tolist(), predictions):
                prediction_records.append(
                    {
                        "path": path,
                        "target": int(target),
                        "prediction": int(prediction),
                    }
                )
    return prediction_records


def choose_classes(
    records: list[dict[str, object]],
    class_names: list[str],
    requested_classes: list[str] | None,
    samples_per_class: int,
    allow_partial: bool,
) -> dict[int, dict[str, list[dict[str, object]]]]:
    grouped: dict[int, dict[str, list[dict[str, object]]]] = {
        index: {"correct": [], "incorrect": []} for index in range(len(class_names))
    }
    for record in records:
        label = int(record["target"])
        bucket = "correct" if record["prediction"] == record["target"] else "incorrect"
        grouped[label][bucket].append(record)

    if requested_classes:
        selected_indices = []
        name_to_idx = {name: index for index, name in enumerate(class_names)}
        for class_name in requested_classes:
            if class_name not in name_to_idx:
                raise ValueError(f"Class '{class_name}' not found in dataset.")
            selected_indices.append(name_to_idx[class_name])
    else:
        selected_indices = []
        for index, _ in enumerate(class_names):
            correct_count = len(grouped[index]["correct"])
            incorrect_count = len(grouped[index]["incorrect"])
            if allow_partial:
                if correct_count > 0 and incorrect_count > 0:
                    selected_indices.append(index)
            else:
                if correct_count >= samples_per_class and incorrect_count >= samples_per_class:
                    selected_indices.append(index)
            if len(selected_indices) == 5:
                break

    if len(selected_indices) < 5:
        raise RuntimeError(
            "Unable to select 5 classes with the requested correct/incorrect sample counts. "
            "Try --allow-partial or provide class names manually."
        )

    selected_class_records: dict[int, dict[str, list[dict[str, object]]]] = {}
    for index in selected_indices[:5]:
        correct = grouped[index]["correct"]
        incorrect = grouped[index]["incorrect"]
        if not allow_partial and (
            len(correct) < samples_per_class or len(incorrect) < samples_per_class
        ):
            raise RuntimeError(
                f"Class '{class_names[index]}' does not have enough correct/incorrect examples."
            )
        selected_class_records[index] = {
            "correct": correct[:samples_per_class],
            "incorrect": incorrect[:samples_per_class],
        }
    return selected_class_records


def load_visual_and_tensor(
    image_path: str,
    visual_transform,
    tensor_transform,
) -> tuple[np.ndarray, torch.Tensor]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        visual_image = visual_transform(image)
        model_tensor = tensor_transform(visual_image)
    visual_array = np.asarray(visual_image, dtype=np.float32) / 255.0
    return visual_array, model_tensor


def main() -> None:
    args = parse_args()
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise SystemExit(
            "pytorch-grad-cam is required for this script. Install it with "
            "`pip install grad-cam`."
        ) from exc

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_metadata = checkpoint["metadata"]
    cached_input_stats = checkpoint.get("input_stats")
    device = get_device(args.device)

    transform_config = build_transform_config_from_checkpoint(
        checkpoint_metadata=checkpoint_metadata,
        cached_input_stats=cached_input_stats,
    )
    train_transform = build_train_transform(transform_config)
    eval_transform = build_eval_transform(transform_config)
    visual_transform = build_visual_transform(transform_config)
    tensor_transform = build_tensor_transform(transform_config)

    data_config = build_data_loader_config_from_args(args, device)
    data_bundle = build_dataloaders(
        data_root=args.data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        config=data_config,
    )

    model = build_model_from_checkpoint_metadata(checkpoint_metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    evaluation_loader = data_bundle["train_loader"] if args.split == "train" else data_bundle["val_loader"]
    class_names = list(data_bundle["classes"])
    prediction_records = collect_predictions(model, evaluation_loader, device)
    selected_class_records = choose_classes(
        records=prediction_records,
        class_names=class_names,
        requested_classes=args.classes,
        samples_per_class=args.samples_per_class,
        allow_partial=args.allow_partial,
    )

    target_layer = resolve_module_by_name(model, args.target_layer)
    output_dir = ensure_dir(args.output_dir)
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "target_layer": args.target_layer,
        "classes": [],
    }

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for class_index, buckets in selected_class_records.items():
            class_name = class_names[class_index]
            class_entry = {"class_name": class_name, "class_index": class_index, "samples": []}
            for bucket_name, bucket_records in buckets.items():
                bucket_dir = ensure_dir(output_dir / class_name / bucket_name)
                for sample_number, record in enumerate(bucket_records, start=1):
                    visual_array, input_tensor = load_visual_and_tensor(
                        image_path=str(record["path"]),
                        visual_transform=visual_transform,
                        tensor_transform=tensor_transform,
                    )
                    input_tensor = input_tensor.unsqueeze(0).to(device)
                    targets = [ClassifierOutputTarget(class_index)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
                    overlay = show_cam_on_image(
                        visual_array,
                        grayscale_cam,
                        use_rgb=True,
                    )

                    image_name = Path(str(record["path"])).stem
                    output_path = bucket_dir / f"{sample_number:02d}_{image_name}.png"
                    Image.fromarray(overlay).save(output_path)

                    class_entry["samples"].append(
                        {
                            "bucket": bucket_name,
                            "source_path": str(record["path"]),
                            "prediction": int(record["prediction"]),
                            "target": int(record["target"]),
                            "output_path": str(output_path),
                        }
                    )
            summary["classes"].append(class_entry)

    save_json(output_dir / "gradcam_summary.json", summary)
    print(f"Saved Grad-CAM outputs to {output_dir}.")


if __name__ == "__main__":
    main()
