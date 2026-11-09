"""Fuse predicted slice envelopes into volumetric ROI bounds."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import nibabel as nib
import numpy as np
from sgrc_net.roi import BBox, bbox2d, expand_bbox, union_longitudinal_fallback
from sgrc_net.io import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    metadata = json.loads(a.metadata.read_text())
    by_image = defaultdict(list)
    for row in metadata["slices"]: by_image[row["id"]].append(row)
    rois = {}
    for row in metadata["images"]:
        image = nib.load(row["image"])
        candidates = defaultdict(list)
        for slice_row in by_image[row["id"]]:
            bounds = bbox2d(a.predictions / f"{slice_row['slice_id']}.nii.gz")
            if bounds is None: continue
            axes = {"axial": (0, 1), "sagittal": (1, 2), "coronal": (0, 2)}[slice_row["plane"]]
            for axis, lower, upper in zip(axes, bounds[::2], bounds[1::2]):
                candidates[2 * axis].append(lower); candidates[2 * axis + 1].append(upper)
        fallback = metadata["normalized_bounds"]
        fused = [int(min(candidates[i]) if i % 2 == 0 else max(candidates[i]))
                 if candidates[i] else int(round(fallback[i] * image.shape[i // 2])) for i in range(6)]
        fused, _ = union_longitudinal_fallback(fused, fallback, image.shape)
        bbox = expand_bbox(BBox(*fused), image.shape, image.header.get_zooms()[:3], 20.0)
        if any(getattr(bbox, name + "0") >= getattr(bbox, name + "1") for name in "xyz"):
            raise ValueError("Empty predicted ROI")
        rois[row["id"]] = bbox.as_dict()
    write_json(a.output, rois)


if __name__ == "__main__":
    main()
