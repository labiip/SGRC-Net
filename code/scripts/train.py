"""Run SGRC-Net or its 2-D ROI localizer with nnU-Net."""
import argparse
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="nnU-Net dataset ID or full name")
    p.add_argument("--stage", choices=("roi", "segmentation"), default="segmentation")
    p.add_argument("--fold", default="0", help="index in your splits_final.json")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    env = os.environ.copy()
    env.update(MANUAL104_SEED=str(a.seed), nnUNet_compile="false",
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    trainer = "nnUNetTrainerSGRCROI" if a.stage == "roi" else "nnUNetTrainerSGRCNet"
    configuration = "2d" if a.stage == "roi" else "3d_fullres"
    command = ["nnUNetv2_train", a.dataset, configuration, a.fold, "-tr", trainer,
               "-p", "nnUNetPlans"]
    if a.resume:
        command.append("--c")
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
