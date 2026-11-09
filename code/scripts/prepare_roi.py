"""Prepare nine fixed slices per CTA for 2-D ROI learning and inference."""
import argparse
import json
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np
from sgrc_net.io import read_manifest, write_json, load_label, check_roi_axis_order
from sgrc_net.roi import (bbox_from_mask, expand_bbox, plane_slice, plane_size,
    plane_spacing, envelope, save_slice)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="Dataset730_SGRCROI")
    p.add_argument("--prior", type=Path, help="Existing roi_metadata.json for new inference-only CTAs")
    a = p.parse_args()
    rows = read_manifest(a.manifest)
    if a.output.exists():
        raise FileExistsError(a.output)
    raw = a.output / "nnUNet_raw" / a.dataset
    inputs = a.output / "roi_inputs"
    for folder in (raw / "imagesTr", raw / "labelsTr", inputs): folder.mkdir(parents=True)
    slices, bounds, splits = [], [], {"train": [], "val": []}
    for row in rows:
        image = nib.load(row["image"])
        if len(image.shape) != 3: raise ValueError("Expected 3-D CTA")
        check_roi_axis_order(image)
        data = np.asarray(image.dataobj, dtype=np.float32)
        spacing = tuple(image.header.get_zooms()[:3])
        label = None
        if row["role"] != "test":
            label_values = load_label(row["label"], image)
            if not set(np.unique(label_values)).issubset({0, 1}):
                raise ValueError("ROI labels must be binary visible-lumen masks; do not use encoded segmentation targets")
            label = label_values > 0
        if label is not None:
            bbox = expand_bbox(bbox_from_mask(label), image.shape, spacing, 15.0)
            bounds.append([value / image.shape[i // 2] for i, value in enumerate(bbox.as_dict().values())])
        for plane in ("axial", "sagittal", "coronal"):
            for fraction in (0.25, 0.5, 0.75):
                index = int(round((plane_size(image.shape, plane) - 1) * fraction))
                name = f"{row['id']}-{plane}{int(fraction * 100):03d}"
                spacing2d = plane_spacing(spacing, plane)
                target = inputs / f"{name}_0000.nii.gz"
                save_slice(plane_slice(data, plane, index), spacing2d, target, np.float32)
                if label is not None:
                    shutil.copy2(target, raw / "imagesTr" / target.name)
                    mask = envelope(plane_slice(label, plane, index), spacing2d, 15.0)
                    save_slice(mask, spacing2d, raw / "labelsTr" / f"{name}.nii.gz", np.uint8)
                    splits[row["role"]].append(name)
                slices.append({"id": row["id"], "slice_id": name, "plane": plane})
    if a.prior:
        fallback = json.loads(a.prior.read_text())["normalized_bounds"]
    else:
        if not bounds: raise ValueError("Provide labeled training/validation CTAs or --prior for inference-only CTAs")
        normalized = np.asarray(bounds, dtype=np.float64)
        fallback = [float(np.quantile(normalized[:, i], 0.05 if i % 2 == 0 else 0.95)) for i in range(6)]
    write_json(a.output / "control" / "roi_metadata.json", {"images": rows, "slices": slices, "normalized_bounds": fallback})
    write_json(a.output / "control" / "splits_final.json", [splits])
    write_json(raw / "dataset.json", {"channel_names": {"0": "CT"},
        "labels": {"background": 0, "roi_envelope": 1}, "numTraining": len(splits["train"]) + len(splits["val"]),
        "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIOWithReorient", "dataset_name": a.dataset})


if __name__ == "__main__":
    main()
