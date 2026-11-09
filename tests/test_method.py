"""CPU checks of the released method operators; no clinical data are used."""
import unittest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sgrc_net.implementation.p3 import P3StateTopologyWrapper, P3StateTopologyLoss, P3TopologyLoader3D
from sgrc_net.roi import BBox, expand_bbox, shifted_affine, union_longitudinal_fallback
from sgrc_net.targets import stenosis_path


class DummyGlobalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv3d(1, 3, 1)
        self.decoder = nn.Identity()

    def forward(self, x):
        logits = self.head(x)
        return [logits, F.avg_pool3d(logits, 2)]


class MethodTests(unittest.TestCase):
    def test_initial_residual_identity_and_inference(self):
        torch.manual_seed(4)
        global_branch = DummyGlobalBranch()
        model = P3StateTopologyWrapper(global_branch, 1, 3)
        x = torch.randn(1, 1, 8, 8, 8)
        output = model(x)
        self.assertEqual(len(output), 2)
        self.assertEqual([v.shape[1] for v in output[0]], [3, 2, 2])
        torch.testing.assert_close(output[0][0], global_branch(x)[0], rtol=0, atol=0)
        model.eval()
        with torch.no_grad():
            inference = model(x)
        self.assertEqual(tuple(inference.shape), (1, 3, 8, 8, 8))
        torch.testing.assert_close(inference, global_branch(x)[0], rtol=0, atol=0)

    def test_public_trainer_real_nnunet_backbone(self):
        from sgrc_net.nnunet.nnUNetTrainerSGRCNet import nnUNetTrainerSGRCNet
        kwargs = dict(n_stages=3, features_per_stage=[8, 16, 32],
            conv_op="torch.nn.modules.conv.Conv3d", kernel_sizes=[[3] * 3] * 3,
            strides=[[1] * 3, [2] * 3, [2] * 3], n_conv_per_stage=[2, 2, 2],
            n_conv_per_stage_decoder=[2, 2], conv_bias=True,
            norm_op="torch.nn.modules.instancenorm.InstanceNorm3d",
            norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
            dropout_op_kwargs=None, nonlin="torch.nn.LeakyReLU",
            nonlin_kwargs={"inplace": True})
        model = nnUNetTrainerSGRCNet.build_network_architecture(
            "dynamic_network_architectures.architectures.unet.PlainConvUNet", kwargs,
            ["conv_op", "norm_op", "nonlin"], 1, 3, True)
        outputs = model(torch.randn(1, 1, 16, 16, 16))
        self.assertEqual([tuple(v.shape) for v in outputs[0]],
            [(1, 3, 16, 16, 16), (1, 2, 16, 16, 16), (1, 2, 16, 16, 16)])

    def test_path_lowest_quarter(self):
        values = torch.tensor([0.9, 0.7, 0.2, 0.8]).reshape(1, 1, 1, 1, 4)
        loss = P3StateTopologyLoss.bottleneck_loss(values, torch.ones_like(values))
        torch.testing.assert_close(loss, -torch.log(torch.tensor(0.2)))
        empty = P3StateTopologyLoss.bottleneck_loss(values, torch.zeros_like(values))
        self.assertEqual(empty.item(), 0)

    def test_encoded_target_split(self):
        encoded = np.asarray([0, 1, 2, 3, 4], dtype=np.int16).reshape(1, 1, 1, 5)
        split = P3TopologyLoader3D.split_encoded_target(encoded)
        np.testing.assert_array_equal(split[0].ravel(), [0, 1, 2, 2, 0])
        np.testing.assert_array_equal(split[1].ravel(), [0, 0, 0, 1, 0])
        np.testing.assert_array_equal(split[2].ravel(), [0, 0, 0, 0, 1])

    def test_objective_backward(self):
        torch.manual_seed(2)
        model = P3StateTopologyWrapper(DummyGlobalBranch(), 1, 3)
        output = model(torch.randn(1, 1, 8, 8, 8))
        target = torch.zeros(1, 3, 8, 8, 8, dtype=torch.long)
        target[:, 0, 1:7, 2:6, 2:6] = 1
        target[:, 0, 3:5, 2:6, 2:6] = 2
        target[:, 1, 3:5, 3:5, 3:5] = 1
        target[:, 2, 0, 0, 0] = 1
        target_low = F.interpolate(target.float(), size=(4, 4, 4), mode="nearest").long()
        loss = P3StateTopologyLoss(batch_dice=False, ddp=False)(output, [target, target_low])
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.delta_head.weight.grad)
        self.assertTrue(torch.isfinite(model.delta_head.weight.grad).all().item())

    def test_geometry_and_extent_union(self):
        bbox = BBox(2, 7, 3, 9, 4, 11)
        expanded = expand_bbox(bbox, (20, 20, 20), (2, 2, 2), 4)
        self.assertEqual(expanded, BBox(0, 9, 1, 11, 2, 13))
        affine = np.diag([2., 3., 4., 1.])
        np.testing.assert_array_equal(shifted_affine(affine, bbox)[:3, 3], [4., 9., 16.])
        fused, used = union_longitudinal_fallback([2, 7, 3, 9, 4, 11], [0, 1, 0, 1, 0.1, 0.9], (20, 20, 20))
        self.assertTrue(used); self.assertEqual(fused, [2, 7, 3, 9, 2, 18])

    def test_annotation_path_construction(self):
        visible = np.zeros((12, 12, 12), dtype=bool)
        visible[2:10, 5:8, 5:8] = True
        proximal = np.zeros_like(visible); proximal[2:4, 5:8, 5:8] = True
        distal = np.zeros_like(visible); distal[8:10, 5:8, 5:8] = True
        roi = visible.copy()
        coords = stenosis_path(visible, roi, proximal, distal)
        self.assertGreater(len(coords), 0)
        self.assertTrue(visible[tuple(coords.T)].all())


if __name__ == "__main__":
    unittest.main()
