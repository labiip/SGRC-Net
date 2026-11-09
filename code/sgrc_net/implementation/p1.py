from __future__ import annotations

import importlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    autocast,
    dummy_context,
    get_tp_fp_fn_tn,
    nnUNetTrainer,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    axis0 = -functional.max_pool3d(-image, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    axis1 = -functional.max_pool3d(-image, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    axis2 = -functional.max_pool3d(-image, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.minimum(torch.minimum(axis0, axis1), axis2)


def soft_skeleton(image: torch.Tensor, iterations: int = 3) -> torch.Tensor:
    def soft_open(value: torch.Tensor) -> torch.Tensor:
        eroded = soft_erode(value)
        return functional.max_pool3d(eroded, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    opened = soft_open(image)
    skeleton = functional.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        delta = functional.relu(image - soft_open(image))
        skeleton = skeleton + functional.relu(delta - skeleton * delta)
    return skeleton


class VisibleClDiceLoss(nn.Module):
    def forward(self, truth: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        prediction_skeleton = soft_skeleton(probability)
        truth_skeleton = soft_skeleton(truth)
        smooth = 1.0
        precision = (prediction_skeleton * truth).sum() + smooth
        precision = precision / (prediction_skeleton.sum() + smooth)
        recall = (truth_skeleton * probability).sum() + smooth
        recall = recall / (truth_skeleton.sum() + smooth)
        return 1.0 - 2.0 * precision * recall / (precision + recall)


class VisibleContinuityLoss(nn.Module):
    """A1 region loss plus a low-weight continuity signal on visible lumen."""

    continuity_weight = 0.05

    def __init__(self, batch_dice: bool, ddp: bool):
        super().__init__()
        self.region = DC_and_CE_loss(
            {
                "batch_dice": batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": ddp,
            },
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = VisibleClDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits.float(), dim=1)
        visible = probabilities[:, 1:2] + probabilities[:, 2:3]
        truth = ((target == 1) | (target == 2)).float()
        return self.region(logits, target) + self.continuity_weight * self.cldice(truth, visible)


class ProbabilisticForegroundLoader3D(nnUNetDataLoader3D):
    def __init__(self, *args, **kwargs):
        kwargs["probabilistic_oversampling"] = True
        super().__init__(*args, **kwargs)


class A3TrainerBase(nnUNetTrainer):
    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        unpack_dataset=True,
        device=torch.device("cuda"),
    ):
        seed = int(os.environ.get("MANUAL104_SEED", "20260805"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = int(os.environ.get("MANUAL104_EPOCHS", "250"))
        self.num_iterations_per_epoch = int(os.environ.get("MANUAL104_ITERATIONS_PER_EPOCH", "250"))
        self.num_val_iterations_per_epoch = int(os.environ.get("MANUAL104_VAL_ITERATIONS_PER_EPOCH", "50"))
        self.batch_size = 1
        self.oversample_foreground_percent = 0.5
        self.save_every = 50
        self.print_to_log_file(
            f"P1: A3 three-state target plus transition-risk auxiliary weight=0.25 "
            f"and visible soft-clDice weight=0.05, "
            f"random initialization, batch_size=1, seed={seed}, 250-epoch budget"
        )

    def get_dataloaders(self):
        trainer_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        original = trainer_module.nnUNetDataLoader3D
        trainer_module.nnUNetDataLoader3D = ProbabilisticForegroundLoader3D
        try:
            return super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader3D = original

    def on_epoch_end(self):
        finished_epoch = self.current_epoch + 1
        super().on_epoch_end()
        if self.local_rank == 0 and finished_epoch % 50 == 0:
            destination = Path(self.output_folder) / f"checkpoint_epoch{finished_epoch:03d}.pth"
            self.save_checkpoint(str(destination))


class nnUNetTrainerManual104A3VisibleContinuity250epochs(A3TrainerBase):
    """A3: A1 visible transition state with low-weight visible continuity."""

    def _build_loss(self):
        loss = VisibleContinuityLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))], dtype=np.float64)
            weights[-1] = 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class P1TransitionRiskWrapper(nn.Module):
    """A3 main head plus a transition-risk auxiliary head.

    During inference only the A3-compatible three-state main output is returned.
    The auxiliary head exists only to regularize the shared representation during
    training. It is supervised by the existing target==2 transition-risk state;
    inference receives no lesion class, side, ROI, anchor, or test-label input.
    """
    def __init__(self, base_network: nn.Module, main_channels: int):
        super().__init__()
        self.base_network = base_network
        self.transition_risk_head = nn.Conv3d(main_channels, 2, kernel_size=1)

    @property
    def decoder(self):
        return self.base_network.decoder

    def forward(self, x):
        main_output = self.base_network(x)
        if not self.training:
            return main_output[0] if isinstance(main_output, (tuple, list)) else main_output
        if isinstance(main_output, (tuple, list)):
            return [(value, self.transition_risk_head(value)) for value in main_output]
        return main_output, self.transition_risk_head(main_output)


class P1TransitionRiskLoss(nn.Module):
    """A3 loss plus explicit transition-risk DiceCE and agreement.

    The auxiliary target is target==2. Normal reference CTA therefore has an
    all-zero transition-risk target, while synthetic CTA uses the frozen
    transition-risk state already present in the training label.
    """
    auxiliary_weight = 0.25
    agreement_weight = 0.05

    def __init__(self, batch_dice: bool, ddp: bool):
        super().__init__()
        self.main_loss = VisibleContinuityLoss(batch_dice=batch_dice, ddp=ddp)
        self.transition_risk_loss = DC_and_CE_loss(
            {
                "batch_dice": batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": ddp,
            },
            {},
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    def single(self, output, target: torch.Tensor) -> torch.Tensor:
        main_logits, risk_logits = output
        risk_target = (target == 2).long()
        loss = self.main_loss(main_logits, target)
        loss = loss + self.auxiliary_weight * self.transition_risk_loss(risk_logits, risk_target)
        main_visible = torch.softmax(main_logits.float(), dim=1)[:, 1:2] + torch.softmax(main_logits.float(), dim=1)[:, 2:3]
        risk_probability = torch.softmax(risk_logits.float(), dim=1)[:, 1:2]
        risk_probability = risk_probability * main_visible.detach()
        return loss + self.agreement_weight * functional.smooth_l1_loss(risk_probability, (target == 2).float())

    def forward(self, logits, target):
        if isinstance(logits, (tuple, list)) and logits and isinstance(logits[0], (tuple, list)):
            targets = target if isinstance(target, (tuple, list)) else [target] * len(logits)
            weights = np.asarray([1.0 / (2**index) for index in range(len(logits))], dtype=np.float64)
            weights[-1] = 0.0
            weights /= weights.sum()
            return sum(float(weights[index]) * self.single(logits[index], targets[index]) for index in range(len(logits)) if weights[index] != 0.0)
        return self.single(logits, target)


class nnUNetTrainerManual104P1TransitionRisk250epochs(A3TrainerBase):
    """P1: A3 plus a supervised transition-risk localization branch."""

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return P1TransitionRiskWrapper(base_network, num_output_channels)

    def validation_step(self, batch: dict) -> dict:
        """Validate the main head while retaining the auxiliary head for loss."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        was_training = self.network.training
        self.network.train()
        try:
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)
                del data
                loss = self.loss(output, target)
        finally:
            if not was_training:
                self.network.eval()

        if self.enable_deep_supervision:
            main_output = output[0][0]
            target = target[0]
        else:
            main_output = output[0] if isinstance(output, (tuple, list)) else output

        axes = [0] + list(range(2, main_output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(main_output) > 0.5).long()
        else:
            output_seg = main_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                main_output.shape, device=main_output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

    def _build_loss(self):
        return P1TransitionRiskLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )
