# Third-party code and dependencies

The model and training implementation extends nnU-Net v2 by the German Cancer Research Center (DKFZ). The custom trainer and loader follow its interfaces and extend its data-loading, training, loss, and validation operations.

- nnU-Net v2: https://github.com/MIC-DKFZ/nnUNet, Apache License 2.0. A copy of the upstream license is in `licenses/nnUNet-Apache-2.0.txt`.
- Dynamic Network Architectures: https://github.com/MIC-DKFZ/dynamic-network-architectures, Apache License 2.0, installed as a dependency.
- PyTorch, NumPy, SciPy, NiBabel, scikit-image and threadpoolctl are installed as dependencies under their respective upstream licenses.

The nnU-Net-derived loader is adapted to split the encoded path/core targets into three jointly transformed supervision channels. The custom network adds two state-response heads and gated residual calibration; the custom loss adds the SGRC-Net supervision terms.

