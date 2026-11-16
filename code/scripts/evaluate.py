"""Compute per-volume visible-lumen metrics on the original CTA grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label


def load_mask(path: Path, grid: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != grid.shape or not np.allclose(
        image.affine, grid.affine, rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"Mask shape or affine does not match the prediction grid: {path}")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    return values > 0


def overlap(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    shared = int(np.count_nonzero(prediction & reference))
    predicted = int(np.count_nonzero(prediction))
    annotated = int(np.count_nonzero(reference))
    dice = 2 * shared / max(predicted + annotated, 1)
    iou = shared / max(predicted + annotated - shared, 1)
    return dice, iou


def dominant_component(components: np.ndarray, anchor: np.ndarray) -> int:
    labels = components[anchor]
    labels = labels[labels > 0]
    if labels.size == 0:
        return 0
    unique, counts = np.unique(labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lesion-roi", type=Path)
    parser.add_argument("--lesion-type", choices=("stenosis", "occlusion"))
    parser.add_argument("--proximal-anchor", type=Path)
    parser.add_argument("--distal-anchor", type=Path)
    parser.add_argument("--occlusion-core", type=Path)
    parser.add_argument("--paired-reference-prediction", type=Path)
    args = parser.parse_args()

    if args.lesion_type and not (args.proximal_anchor and args.distal_anchor):
        parser.error("Lesion event evaluation requires both anchor masks.")
    if not args.lesion_type and (args.proximal_anchor or args.distal_anchor):
        parser.error("Select --lesion-type when supplying anchor masks.")
    if bool(args.occlusion_core) != (args.lesion_type == "occlusion"):
        parser.error("Supply --occlusion-core for occlusion evaluation.")
    if args.paired_reference_prediction and not args.lesion_roi:
        parser.error("Paired agreement requires --lesion-roi.")

    grid = nib.load(str(args.prediction))
    if len(grid.shape) != 3:
        raise ValueError("The prediction must be a three-dimensional mask.")
    prediction = load_mask(args.prediction, grid)
    reference = load_mask(args.reference, grid)
    dice, iou = overlap(prediction, reference)
    result = {"whole_tree_dice": dice, "whole_tree_iou": iou}

    if args.lesion_roi:
        roi = load_mask(args.lesion_roi, grid)
        if not roi.any():
            raise ValueError("The lesion ROI must contain foreground.")
        result["lesion_dice"] = overlap(prediction & roi, reference & roi)[0]
        if args.paired_reference_prediction:
            paired = load_mask(args.paired_reference_prediction, grid)
            spacing = tuple(float(value) for value in grid.header.get_zooms()[:3])
            outside = distance_transform_edt(~roi, sampling=spacing) > 5.0
            result["paired_outside_5mm_dice"] = overlap(prediction & outside, paired & outside)[0]

    if args.lesion_type:
        proximal = load_mask(args.proximal_anchor, grid)
        distal = load_mask(args.distal_anchor, grid)
        if not proximal.any() or not distal.any():
            raise ValueError("Both anchor masks must contain foreground.")
        components, _ = label(prediction, structure=np.ones((3, 3, 3), dtype=np.uint8))
        proximal_component = dominant_component(components, proximal)
        distal_component = dominant_component(components, distal)
        connected = proximal_component > 0 and proximal_component == distal_component
        result["proximal_anchor_predicted"] = proximal_component > 0
        result["distal_anchor_predicted"] = distal_component > 0
        result["anchors_connected"] = connected
        if args.lesion_type == "stenosis":
            result["stenosis_break"] = not connected
        else:
            core = load_mask(args.occlusion_core, grid)
            if not core.any():
                raise ValueError("The occlusion core must contain foreground.")
            core_voxels = int(np.count_nonzero(prediction & core))
            result["core_foreground_voxels"] = core_voxels
            result["core_intrusion"] = core_voxels > 0
            result["false_anchor_to_anchor_bridge"] = connected

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
