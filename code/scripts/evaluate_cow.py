"""Evaluate a binary CoW prediction in the official annotated ROI."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import SimpleITK as sitk


METRICS_COMMIT = "25ca8399e056e71e3ff6743ac4a47ce28624cb3d"


def evaluate(prediction: Path, reference: Path, roi: Path, metrics_repo: Path) -> dict:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(metrics_repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_commit != METRICS_COMMIT or not (metrics_repo / "topcow24_eval").is_dir():
        raise ValueError(f"Use the official TopCoW metrics checkout at {METRICS_COMMIT}.")

    sys.path.insert(0, str(metrics_repo.resolve()))
    import skimage.morphology as morphology

    # Keep Lee's three-dimensional skeletonization with scikit-image 0.25.
    if not hasattr(morphology, "skeletonize_3d"):
        morphology.skeletonize_3d = lambda image: morphology.skeletonize(image, method="lee")
    from topcow24_eval.metrics.seg_metrics.clDice import clDice
    from topcow24_eval.metrics.seg_metrics.cls_avg_dice import dice_coefficient_single_label
    from topcow24_eval.metrics.seg_metrics.cls_avg_hd95 import hd95_single_label
    from topcow24_eval.utils.utils_box import get_dcm_slice, parse_roi_txt

    pred = sitk.ReadImage(str(prediction))
    truth = sitk.ReadImage(str(reference))
    for name in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
        if not np.allclose(getattr(pred, name)(), getattr(truth, name)(), atol=1e-4, rtol=1e-5):
            raise ValueError(f"Prediction/reference geometry mismatch: {name}")
    size, location = parse_roi_txt(roi)
    region = get_dcm_slice(truth.GetSize(), size, location)
    pred, truth = pred[region], truth[region]
    p, t = sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(truth)
    if not set(np.unique(p)).issubset({0, 1}) or not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("Use binary prediction and reference masks.")
    if not np.any(t):
        raise ValueError("The reference must contain CoW foreground inside the official ROI.")
    return {
        "Dice": float(dice_coefficient_single_label(gt=truth, pred=pred, label=1)),
        "HD95_mm": float(hd95_single_label(gt=truth, pred=pred, label=1)[0]),
        "clDice": float(clDice(v_p_pred=p, v_l_gt=t)),
        "empty_prediction": not bool(np.any(p)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--roi", type=Path, required=True, help="Official ROI metadata .txt")
    parser.add_argument("--metrics-repo", type=Path, required=True)
    args = parser.parse_args()
    # Upstream ROI utilities print diagnostics. Keep stdout as a single JSON object.
    with redirect_stdout(io.StringIO()):
        result = evaluate(args.prediction, args.reference, args.roi, args.metrics_repo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
