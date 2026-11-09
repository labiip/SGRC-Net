from __future__ import annotations

import os
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerManual104ROI100epochs(nnUNetTrainer):
    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda"),
    ):
        seed = int(os.environ.get("MANUAL104_SEED", "20260724"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.num_epochs = 100
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.manual104_seed = seed
        self.print_to_log_file(
            f"Manual104 ROI: random initialization, fixed seed={seed}"
        )


class nnUNetTrainerManual104ROISmoke(nnUNetTrainerManual104ROI100epochs):
    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda"),
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            device,
        )
        self.num_epochs = 1
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1
