"""Public nnU-Net trainer names for SGRC-Net and its ROI localizer."""
from sgrc_net.implementation.p3 import nnUNetTrainerManual104P3StateTopology250epochs
from sgrc_net.implementation.roi import nnUNetTrainerManual104ROI100epochs


class nnUNetTrainerSGRCNet(nnUNetTrainerManual104P3StateTopology250epochs):
    """SGRC-Net: three-state segmentation with state-guided residual calibration."""


class nnUNetTrainerSGRCROI(nnUNetTrainerManual104ROI100epochs):
    """Fixed nine-slice 2-D ROI localization used before SGRC-Net segmentation."""
