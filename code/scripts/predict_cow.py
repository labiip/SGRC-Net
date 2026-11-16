"""Predict and combine binary CoW masks on the original CTA grid."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def validate_probabilities(probability: np.ndarray) -> None:
    if probability.ndim != 4 or probability.shape[0] != 2:
        raise ValueError("Expected two probability channels on a three-dimensional grid.")
    if (
        not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
        or not np.allclose(probability.sum(0), 1, atol=1e-4)
    ):
        raise ValueError("Invalid binary probabilities.")


def predict(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = False
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False, device=torch.device(args.device),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model), use_folds=(0,), checkpoint_name=args.checkpoint,
    )
    writer = predictor.plans_manager.image_reader_writer_class()
    if not isinstance(writer, NibabelIOWithReorient):
        raise ValueError("Use the CoW protocol's NibabelIOWithReorient image reader.")
    image, properties = writer.read_images([str(args.image)])
    segmentation, probabilities = predictor.predict_single_npy_array(
        image, copy.deepcopy(properties), save_or_return_probabilities=True,
    )
    validate_probabilities(probabilities)
    if probabilities.shape[1:] != image.shape[1:] or not np.array_equal(
        segmentation, probabilities.argmax(0)
    ):
        raise ValueError("Original-grid prediction or argmax mismatch.")
    writer.write_seg(segmentation.astype(np.uint8), str(args.output), properties)
    if args.probabilities:
        np.savez_compressed(
            args.probabilities,
            probabilities=probabilities.astype(np.float32, copy=False),
            original_affine=properties["nibabel_stuff"]["original_affine"],
            reoriented_affine=properties["nibabel_stuff"]["reoriented_affine"],
            spacing=properties["spacing"],
        )


def ensemble(args: argparse.Namespace) -> None:
    if len({path.resolve() for path in args.folds}) != 5:
        raise ValueError("Supply one distinct original-grid probability file per outer fold.")
    total = None
    geometry = {}
    for path in args.folds:
        with np.load(path, allow_pickle=False) as data:
            probability = data["probabilities"]
            validate_probabilities(probability)
            current = {
                key: data[key] for key in ("original_affine", "reoriented_affine", "spacing")
            }
            if any(not np.isfinite(value).all() for value in current.values()):
                raise ValueError(f"Non-finite ensemble grid metadata in {path}.")
            if total is None:
                total = probability.astype(np.float64)
                geometry = current
            else:
                if probability.shape != total.shape or any(
                    not np.array_equal(geometry[key], current[key]) for key in geometry
                ):
                    raise ValueError(f"Ensemble grid mismatch in {path}.")
                total += probability
    properties = {
        "spacing": geometry["spacing"],
        "nibabel_stuff": {
            "original_affine": geometry["original_affine"],
            "reoriented_affine": geometry["reoriented_affine"],
        },
    }
    segmentation = (total / 5).argmax(0).astype(np.uint8)
    NibabelIOWithReorient().write_seg(segmentation, str(args.output), properties)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("predict", help="Restore one trained fold's prediction.")
    single.add_argument("--image", type=Path, required=True)
    single.add_argument("--model", type=Path, required=True, help="nnU-Net model directory")
    single.add_argument("--checkpoint", default="checkpoint_best.pth")
    single.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    single.add_argument("--probabilities", type=Path, help="Optional original-grid .npz output")
    single.add_argument("--output", type=Path, required=True)

    combine = commands.add_parser("ensemble", help="Average five probability maps, then take argmax.")
    combine.add_argument("--folds", nargs=5, type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.name.endswith(".nii.gz") or args.output.exists():
        parser.error("--output must be a new .nii.gz path.")
    if args.command == "predict":
        if args.probabilities and (args.probabilities.suffix != ".npz" or args.probabilities.exists()):
            parser.error("--probabilities must be a new .npz path.")
        predict(args)
    else:
        ensemble(args)


if __name__ == "__main__":
    main()
