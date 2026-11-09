"""Paste binary predictions back into the original CTA grid without smoothing."""
import argparse
from pathlib import Path
import json
import nibabel as nib
import numpy as np
from sgrc_net.io import save_image
from sgrc_net.roi import BBox, shifted_affine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for row in json.loads(a.metadata.read_text()):
        if row["role"] != "test": continue
        image = nib.load(row["source_image"])
        bbox = BBox(**{k: row[k] for k in ("x0", "x1", "y0", "y1", "z0", "z1")})
        prediction = nib.load(a.predictions / f"{row['id']}.nii.gz")
        values = np.asarray(prediction.dataobj)
        expected = tuple(getattr(bbox, key + "1") - getattr(bbox, key + "0") for key in "xyz")
        if values.shape != expected or not np.allclose(prediction.affine, shifted_affine(image.affine, bbox)):
            raise ValueError("Prediction must first be exported by nnU-Net to the original crop grid")
        full = np.zeros(image.shape, dtype=np.uint8)
        full[bbox.slices()] = (values > 0).astype(np.uint8)
        save_image(full, image.affine, image.header, a.output / f"{row['id']}.nii.gz", np.uint8)


if __name__ == "__main__":
    main()
