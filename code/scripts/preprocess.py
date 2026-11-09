"""Plan and preprocess a prepared ROI or encoded SGRC-Net dataset."""
import argparse
from pathlib import Path
import json
import os
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="output of a preparation script")
    p.add_argument("--dataset", required=True, help="DatasetNNN_Name")
    p.add_argument("--stage", choices=("roi", "segmentation"), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--method-config", type=Path, default=Path(__file__).resolve().parents[2] / "config/method.json")
    a = p.parse_args()
    marker = a.root / "control/preprocessing_complete.json"
    if marker.exists():
        raise FileExistsError("This view has already been preprocessed; use its arrays directly or prepare a fresh output view")
    raw_json = a.root / "nnUNet_raw" / a.dataset / "dataset.json"
    if a.stage == "segmentation":
        labels = json.loads(raw_json.read_text())["labels"]
        if set(labels.values()) != {0, 1, 2, 3, 4}:
            raise ValueError("Preprocessing requires the original 5-value encoded label map; do not rerun on runtime 3-state metadata")
    env = os.environ.copy()
    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        path = (a.root / key).resolve(); path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    identifier = str(int(a.dataset[7:10]))
    subprocess.run(["nnUNetv2_extract_fingerprint", "-d", identifier, "--verify_dataset_integrity", "--clean", "-np", str(a.workers)], env=env, check=True)
    subprocess.run(["nnUNetv2_plan_experiment", "-d", identifier], env=env, check=True)
    folder = Path(env["nnUNet_preprocessed"]) / a.dataset
    configuration = "2d" if a.stage == "roi" else "3d_fullres"
    if a.stage == "segmentation":
        method = json.loads(a.method_config.read_text())["stage2"]
        plans_path = folder / "nnUNetPlans.json"
        plans = json.loads(plans_path.read_text())
        reference_transpose = method["reference_transpose_forward"]
        current_transpose = plans["transpose_forward"]
        physical_spacing, physical_patch = [None] * 3, [None] * 3
        for internal, physical in enumerate(reference_transpose):
            physical_spacing[physical] = method["target_spacing_reference"][internal]
            physical_patch[physical] = method["patch_size_reference"][internal]
        fullres = plans["configurations"]["3d_fullres"]
        fullres["spacing"] = [physical_spacing[i] for i in current_transpose]
        fullres["patch_size"] = [physical_patch[i] for i in current_transpose]
        fullres["batch_size"] = method["batch_size"]
        stride = [1, 1, 1]
        for stage_stride in fullres["architecture"]["arch_kwargs"]["strides"]:
            stride = [stride[i] * int(stage_stride[i]) for i in range(3)]
        if any(fullres["patch_size"][i] % stride[i] for i in range(3)):
            raise ValueError("Fixed patch is incompatible with the planned strides")
        plans_path.write_text(json.dumps(plans, indent=2) + "\n")
    subprocess.run(["nnUNetv2_preprocess", "-d", identifier, "-c", configuration, "-plans_name", "nnUNetPlans", "-np", str(a.workers)], env=env, check=True)
    shutil.copy2(a.root / "control/splits_final.json", folder / "splits_final.json")
    if a.stage == "segmentation":
        # Aux labels remain in the preprocessed arrays; only the main-head label map changes.
        for path in (a.root / "nnUNet_raw" / a.dataset / "dataset.json", folder / "dataset.json"):
            data = json.loads(path.read_text())
            data["labels"] = {"background": 0, "remaining_visible_lumen": 1, "lesion_associated_visible_lumen": 2}
            path.write_text(json.dumps(data, indent=2) + "\n")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"stage": a.stage, "dataset": a.dataset, "status": "complete"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
