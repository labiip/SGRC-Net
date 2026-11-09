"""Crop CTA and encoded supervision to the predicted ROIs for nnU-Net."""
import argparse
from pathlib import Path
import json
import nibabel as nib
import numpy as np
from sgrc_net.io import read_manifest, write_json, load_label, save_image
from sgrc_net.roi import BBox, shifted_affine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rois", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="Dataset731_SGRCNet")
    a = p.parse_args()
    rows = read_manifest(a.manifest)
    rois = json.loads(a.rois.read_text())
    if a.output.exists(): raise FileExistsError(a.output)
    raw = a.output / "nnUNet_raw" / a.dataset
    for folder in (raw / "imagesTr", raw / "labelsTr", a.output / "imagesTs"):
        folder.mkdir(parents=True)
    splits, restore = {"train": [], "val": []}, []
    for row in rows:
        image = nib.load(row["image"])
        if len(image.shape) != 3: raise ValueError("Expected 3-D CTA")
        bbox = BBox(**rois[row["id"]])
        for axis, name in enumerate("xyz"):
            if not 0 <= getattr(bbox, name + "0") < getattr(bbox, name + "1") <= image.shape[axis]:
                raise ValueError("ROI bounds outside the source grid")
        cropped_affine = shifted_affine(image.affine, bbox)
        image_dir = a.output / "imagesTs" if row["role"] == "test" else raw / "imagesTr"
        save_image(np.asarray(image.dataobj)[bbox.slices()], cropped_affine, image.header,
            image_dir / f"{row['id']}_0000.nii.gz", np.float32)
        if row["role"] != "test":
            label = load_label(row["encoded_label"], image)
            if not set(np.unique(label)).issubset({0, 1, 2, 3, 4}):
                raise ValueError("Encoded target labels must be 0..4")
            save_image(label[bbox.slices()], cropped_affine, image.header,
                raw / "labelsTr" / f"{row['id']}.nii.gz", np.uint8)
            splits[row["role"]].append(row["id"])
        restore.append({"id": row["id"], "source_image": row["image"], "role": row["role"], **bbox.as_dict()})
    write_json(a.output / "control" / "restore.json", restore)
    write_json(a.output / "control" / "splits_final.json", [splits])
    write_json(raw / "dataset.json", {"channel_names": {"0": "CT"}, "labels": {
        "background": 0, "remaining_visible_lumen": 1, "lesion_associated_visible_lumen": 2,
        "stenosis_visible_path": 3, "occlusion_core": 4},
        "numTraining": len(splits["train"]) + len(splits["val"]), "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient", "dataset_name": a.dataset})


if __name__ == "__main__":
    main()
