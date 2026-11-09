from __future__ import annotations

import importlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    autocast,
    dummy_context,
    get_tp_fp_fn_tn,
    nnUNetTrainer,
)
from sgrc_net.implementation.p1 import (
    P1TransitionRiskLoss,
    nnUNetTrainerManual104P1TransitionRisk250epochs,
)


class P3TopologyLoader3D(nnUNetDataLoader3D):
    """Split encoded topology labels before applying shared spatial transforms.

    Raw/preprocessed class 3 is a stenosis visible path and class 4 is an
    occlusion zero-core. Channel 0 is remapped to the unchanged P2 three-state
    segmentation target; channels 1 and 2 carry binary path/core supervision.
    """

    def __init__(self, *args, **kwargs):
        kwargs["probabilistic_oversampling"] = True
        super().__init__(*args, **kwargs)

    @staticmethod
    def split_encoded_target(seg: np.ndarray) -> np.ndarray:
        if seg.shape[0] != 1:
            raise RuntimeError(f"expected one encoded segmentation channel, got {seg.shape}")
        encoded = seg[0]
        unexpected = set(np.unique(encoded).tolist()) - {-1, 0, 1, 2, 3, 4}
        if unexpected:
            raise RuntimeError(f"unexpected P3 encoded labels: {sorted(unexpected)}")
        main = encoded.copy()
        main[main == 3] = 2
        main[main == 4] = 0
        path = (encoded == 3).astype(seg.dtype)
        core = (encoded == 4).astype(seg.dtype)
        return np.stack((main, path, core), axis=0)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_shape = (self.seg_shape[0], 3, *self.seg_shape[2:])
        seg_all = np.zeros(seg_shape, dtype=np.int16)

        for batch_index, key in enumerate(selected_keys):
            force_fg = self.get_do_oversample(batch_index)
            data, encoded_seg, properties = self._data.load_case(key)
            seg = self.split_encoded_target(encoded_seg)
            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(
                shape, force_fg, properties["class_locations"]
            )
            valid_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_ubs = np.minimum(shape, bbox_ubs)
            data_slice = tuple(
                [slice(0, data.shape[0])]
                + [slice(low, high) for low, high in zip(valid_lbs, valid_ubs)]
            )
            seg_slice = tuple(
                [slice(0, seg.shape[0])]
                + [slice(low, high) for low, high in zip(valid_lbs, valid_ubs)]
            )
            data = data[data_slice]
            seg = seg[seg_slice]
            padding = [
                (-min(0, bbox_lbs[axis]), max(bbox_ubs[axis] - shape[axis], 0))
                for axis in range(len(shape))
            ]
            padding = ((0, 0), *padding)
            data_all[batch_index] = np.pad(
                data, padding, "constant", constant_values=0
            )
            seg_all[batch_index] = np.pad(
                seg, padding, "constant", constant_values=-1
            )

        if self.transforms is None:
            return {"data": data_all, "target": seg_all, "keys": selected_keys}

        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            data_tensor = torch.from_numpy(data_all).float()
            seg_tensor = torch.from_numpy(seg_all).to(torch.int16)
            images, segmentations = [], []
            for batch_index in range(self.batch_size):
                transformed = self.transforms(
                    **{
                        "image": data_tensor[batch_index],
                        "segmentation": seg_tensor[batch_index],
                    }
                )
                images.append(transformed["image"])
                segmentations.append(transformed["segmentation"])
            data_tensor = torch.stack(images)
            if isinstance(segmentations[0], list):
                targets = [
                    torch.stack([item[level] for item in segmentations])
                    for level in range(len(segmentations[0]))
                ]
            else:
                targets = torch.stack(segmentations)
        return {"data": data_tensor, "target": targets, "keys": selected_keys}


class P3StateTopologyWrapper(nn.Module):
    """P2 local refinement plus a learned zero-core risk state."""

    def __init__(
        self,
        base_network: nn.Module,
        num_input_channels: int,
        main_channels: int,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.base_network = base_network
        self.transition_risk_head = nn.Conv3d(main_channels, 2, kernel_size=1)
        self.core_risk_head = nn.Conv3d(main_channels, 2, kernel_size=1)
        local_channels = num_input_channels + main_channels + 4
        self.local_block = nn.Sequential(
            nn.Conv3d(local_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.delta_head = nn.Conv3d(hidden_channels, main_channels, kernel_size=1)
        self.gate_head = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.0)

    @property
    def decoder(self):
        return self.base_network.decoder

    @staticmethod
    def resize(value: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
        if tuple(value.shape[2:]) == tuple(spatial):
            return value
        return F.interpolate(value, size=spatial, mode="trilinear", align_corners=False)

    def fuse(self, image, global_logits, risk_logits, core_logits):
        spatial = tuple(global_logits.shape[2:])
        image = self.resize(image, spatial)
        risk_logits = self.resize(risk_logits, spatial)
        core_logits = self.resize(core_logits, spatial)
        features = self.local_block(
            torch.cat((image, global_logits, risk_logits, core_logits), dim=1)
        )
        gate = torch.sigmoid(self.gate_head(features))
        return global_logits + gate * self.delta_head(features)

    def forward(self, image: torch.Tensor):
        base_output = self.base_network(image)
        if isinstance(base_output, (tuple, list)):
            outputs = []
            for index, global_logits in enumerate(base_output):
                risk_logits = self.transition_risk_head(global_logits)
                core_logits = self.core_risk_head(global_logits)
                fused = (
                    self.fuse(image, global_logits, risk_logits, core_logits)
                    if index == 0
                    else global_logits
                )
                outputs.append((fused, risk_logits, core_logits))
            return outputs if self.training else outputs[0][0]
        risk_logits = self.transition_risk_head(base_output)
        core_logits = self.core_risk_head(base_output)
        fused = self.fuse(image, base_output, risk_logits, core_logits)
        return (fused, risk_logits, core_logits) if self.training else fused


class P3StateTopologyLoss(nn.Module):
    """P2 objective plus bounded path-bottleneck and zero-core penalties."""

    local_focus_weight = 0.15
    path_bottleneck_weight = 0.05
    core_state_weight = 0.05
    core_suppression_weight = 0.10

    def __init__(self, batch_dice: bool, ddp: bool) -> None:
        super().__init__()
        self.p1 = P1TransitionRiskLoss(batch_dice=batch_dice, ddp=ddp)

    @staticmethod
    def visible_logit(main_logits: torch.Tensor) -> torch.Tensor:
        foreground = torch.logsumexp(main_logits.float()[:, 1:3], dim=1, keepdim=True)
        return foreground - main_logits.float()[:, 0:1]

    @staticmethod
    def balanced_binary_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive = target > 0.5
        negative = ~positive
        parts = []
        if bool(positive.any()):
            parts.append(F.softplus(-logit[positive]).mean())
        if bool(negative.any()):
            parts.append(0.1 * F.softplus(logit[negative]).mean())
        return sum(parts) if parts else logit.sum() * 0.0

    @staticmethod
    def bottleneck_loss(visible_probability: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        losses = []
        for batch_index in range(visible_probability.shape[0]):
            values = visible_probability[batch_index][path[batch_index] > 0.5]
            if values.numel() == 0:
                continue
            count = max(1, int(np.ceil(values.numel() * 0.25)))
            worst = torch.topk(-torch.log(values.clamp_min(1e-5)), count).values
            losses.append(worst.mean())
        return torch.stack(losses).mean() if losses else visible_probability.sum() * 0.0

    def single(self, output, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] != 3:
            raise RuntimeError(f"P3 requires main/path/core target channels, got {target.shape}")
        main_logits, risk_logits, core_logits = output
        main_target = target[:, 0:1]
        path_target = target[:, 1:2].float()
        core_target = target[:, 2:3].float()

        loss = self.p1.single((main_logits, risk_logits), main_target)
        visible_truth = ((main_target == 1) | (main_target == 2)).float()
        lesion_focus = (main_target == 2).float()
        visible_logit = self.visible_logit(main_logits)
        focused_bce = F.binary_cross_entropy_with_logits(
            visible_logit,
            visible_truth,
            weight=1.0 + 0.75 * lesion_focus,
        )
        visible_probability = torch.sigmoid(visible_logit)
        path_loss = self.bottleneck_loss(visible_probability, path_target)

        core_binary_logit = core_logits.float()[:, 1:2] - core_logits.float()[:, 0:1]
        core_state = self.balanced_binary_loss(core_binary_logit, core_target)
        core_voxels = core_target.sum()
        core_suppression = (
            (visible_probability * core_target).sum() / core_voxels.clamp_min(1.0)
            if bool(core_voxels > 0)
            else visible_probability.sum() * 0.0
        )
        return (
            loss
            + self.local_focus_weight * focused_bce
            + self.path_bottleneck_weight * path_loss
            + self.core_state_weight * core_state
            + self.core_suppression_weight * core_suppression
        )

    def forward(self, logits, target):
        if isinstance(logits, (tuple, list)) and logits and isinstance(logits[0], (tuple, list)):
            targets = target if isinstance(target, (tuple, list)) else [target] * len(logits)
            weights = np.asarray(
                [1.0 / (2**index) for index in range(len(logits))], dtype=np.float64
            )
            weights[-1] = 0.0
            weights /= weights.sum()
            return sum(
                float(weights[index]) * self.single(logits[index], targets[index])
                for index in range(len(logits))
                if weights[index] != 0.0
            )
        return self.single(logits, target)


class nnUNetTrainerManual104P3StateTopology250epochs(
    nnUNetTrainerManual104P1TransitionRisk250epochs
):
    """P3: inference-blind state-conditioned topology calibration."""

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
        return P3StateTopologyWrapper(
            base_network,
            num_input_channels=num_input_channels,
            main_channels=num_output_channels,
            hidden_channels=16,
        )

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original = trainer_module.nnUNetDataLoader3D
        trainer_module.nnUNetDataLoader3D = P3TopologyLoader3D
        try:
            return nnUNetTrainer.get_dataloaders(self)
        finally:
            trainer_module.nnUNetDataLoader3D = original

    def _build_loss(self):
        return P3StateTopologyLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )

    def validation_step(self, batch: dict) -> dict:
        data, target = batch["data"], batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        was_training = self.network.training
        self.network.train()
        try:
            context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
            with context:
                output = self.network(data)
                loss = self.loss(output, target)
        finally:
            if not was_training:
                self.network.eval()

        if self.enable_deep_supervision:
            main_output = output[0][0]
            metric_target = target[0][:, 0:1]
        else:
            main_output = output[0]
            metric_target = target[:, 0:1]
        axes = [0] + list(range(2, main_output.ndim))
        output_seg = main_output.argmax(1)[:, None]
        onehot = torch.zeros(
            main_output.shape, device=main_output.device, dtype=torch.float32
        )
        onehot.scatter_(1, output_seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(onehot, metric_target, axes=axes, mask=None)
        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy()[1:],
            "fp_hard": fp.detach().cpu().numpy()[1:],
            "fn_hard": fn.detach().cpu().numpy()[1:],
        }
