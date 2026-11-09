"""User-supplied data manifests and NIfTI geometry checks."""
from pathlib import Path
import csv
import json
import re
import nibabel as nib
import numpy as np


def read_manifest(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    seen, patient_roles = set(), {}
    for row in rows:
        identifier, role = row["id"], row["role"]
        if not re.fullmatch(r"[A-Za-z0-9-]+", identifier) or identifier in seen:
            raise ValueError("IDs must be unique, using letters, digits and hyphens")
        if role not in {"train", "val", "test"}:
            raise ValueError(f"Invalid role {role}")
        seen.add(identifier)
        patient = row["subject_id"]
        if patient in patient_roles and patient_roles[patient] != role:
            raise ValueError("Samples from one subject cannot cross roles")
        patient_roles[patient] = role
        for key in ("image", "label", "encoded_label"):
            if row.get(key):
                value = Path(row[key])
                row[key] = str(value if value.is_absolute() else Path(path).resolve().parent / value)
    return rows


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_label(path, image):
    label = nib.load(path)
    if image.shape != label.shape or not np.allclose(image.affine, label.affine):
        raise ValueError("Image and label grids differ")
    return np.asarray(label.dataobj)


def save_image(data, affine, reference_header, path, dtype):
    header = reference_header.copy(); header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data.astype(dtype), affine, header), str(path))


def check_roi_axis_order(image):
    """The original fixed-plane extraction assumes x=LR, y=AP, z=SI axes."""
    codes = nib.aff2axcodes(image.affine)
    if not (codes[0] in ("L", "R") and codes[1] in ("A", "P") and codes[2] in ("S", "I")):
        raise ValueError("ROI slicing requires voxel axes LR/AP/SI; reorient the CTA and every label/mask together before preparing this view")
