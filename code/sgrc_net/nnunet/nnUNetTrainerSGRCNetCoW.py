"""SGRC-Net adapter for full-image binary CoW segmentation with nnU-Net."""

from __future__ import annotations

import importlib
import os
import random

import numpy as np
import torch
from torch import nn

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from sgrc_net.implementation.p3 import P3StateTopologyWrapper


class BinaryCoWNetwork(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.p3 = P3StateTopologyWrapper(backbone, 1, 3, hidden_channels=16)
        self.p3.to(memory_format=torch.channels_last_3d)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.p3(image.contiguous(memory_format=torch.channels_last_3d))
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, torch.Tensor) or logits.shape[1] != 3:
            raise RuntimeError("Expected one full-resolution three-state output.")
        foreground = torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)
        return torch.cat((logits[:, :1], foreground), dim=1)


class ChannelsLastNetwork(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.to(memory_format=torch.channels_last_3d)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.contiguous(memory_format=torch.channels_last_3d))


class ProbabilisticForegroundLoader3D(nnUNetDataLoader3D):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["probabilistic_oversampling"] = True
        super().__init__(*args, **kwargs)


class CompatiblePolyLRScheduler(PolyLRScheduler):
    """Use nnU-Net 2.5.2's polynomial step with the current PyTorch constructor."""

    def __init__(self, optimizer, initial_lr: float, max_steps: int) -> None:
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = 0.9
        self.ctr = 0
        torch.optim.lr_scheduler.LRScheduler.__init__(self, optimizer, last_epoch=-1)


class nnUNetTrainerSGRCNetCoW(nnUNetTrainer):
    """Train the retained SGRC-Net architecture using binary DiceCE."""

    def __init__(
        self, plans, configuration, fold, dataset_json, unpack_dataset=True,
        device=torch.device("cuda"),
    ) -> None:
        seed = int(os.environ.get("SGRC_COW_SEED", "20261012"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if tuple(self.configuration_manager.patch_size) != (96, 96, 96):
            raise ValueError("The binary CoW protocol requires a 96 × 96 × 96 patch.")
        if self.configuration_manager.batch_size != 1:
            raise ValueError("The binary CoW protocol requires batch size 1.")
        if self.label_manager.has_regions or set(self.label_manager.all_labels) != {0, 1}:
            raise ValueError("Expected binary CoW annotation labels 0 and 1.")
        self.enable_deep_supervision = False
        self.num_epochs = 250
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.batch_size = 1
        self.oversample_foreground_percent = 0.5
        self.save_every = 50
        self.print_to_log_file(f"SGRC-Net CoW: random initialization, seed={seed}")

    @staticmethod
    def build_network_architecture(
        architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
        num_input_channels, num_output_channels, enable_deep_supervision=True,
    ) -> nn.Module:
        if (num_input_channels, num_output_channels) != (1, 2):
            raise ValueError("Expected one CTA channel and two binary output channels.")
        backbone = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, 1, 3, False,
        )
        return ChannelsLastNetwork(BinaryCoWNetwork(backbone))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True,
        )
        return optimizer, CompatiblePolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        # The CoW protocol uses the same single full-resolution output throughout.
        return None

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original = trainer_module.nnUNetDataLoader3D
        trainer_module.nnUNetDataLoader3D = ProbabilisticForegroundLoader3D
        try:
            return super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader3D = original

    def on_epoch_end(self) -> None:
        finished = self.current_epoch + 1
        super().on_epoch_end()
        if self.local_rank == 0 and finished % 50 == 0:
            self.save_checkpoint(os.path.join(self.output_folder, f"checkpoint_epoch{finished:03d}.pth"))
