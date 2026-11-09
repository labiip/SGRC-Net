"""Predict ROI masks or calibrated visible-lumen masks from a trained model."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="nnU-Net *_0000.nii.gz inputs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="model directory containing fold_*/")
    p.add_argument("--folds", nargs="+", default=["0"])
    p.add_argument("--stage", choices=("roi", "segmentation"), default="segmentation")
    p.add_argument("--checkpoint", default="checkpoint_best.pth")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    folds = [int(x) if x != "all" else x for x in a.folds]
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
        use_mirroring=False, perform_everything_on_device=(a.device == "cuda"),
        device=torch.device(a.device), verbose=False)
    predictor.initialize_from_trained_model_folder(str(a.model), use_folds=folds,
        checkpoint_name=a.checkpoint)
    predictor.predict_from_files(str(a.input), str(a.output), save_probabilities=False,
        overwrite=False, num_processes_preprocessing=2, num_processes_segmentation_export=2)
    if a.stage == "segmentation":
        for path in sorted(a.output.glob("*.nii.gz")):
            image = nib.load(path)
            labels = np.asarray(image.dataobj)
            if not set(np.unique(labels)).issubset({0, 1, 2}):
                raise ValueError(f"Unexpected predicted states in {path.name}")
            binary = np.isin(labels, (1, 2)).astype(np.uint8)
            header = image.header.copy(); header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(binary, image.affine, header), path)


if __name__ == "__main__":
    main()
