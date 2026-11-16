# SGRC-Net

SGRC-Net segments the visible lumen of the internal carotid artery on CTA using a slice-based localizer and state-guided residual calibration.

### 1. Install and verify

Use Python 3.10–3.12 and install [PyTorch](https://pytorch.org/get-started/locally/) for your platform. Use Linux with CUDA for training. From the repository root, run:

```bash
python -m pip install -r requirements.txt
python code/scripts/install_trainers.py
python -m unittest discover -s tests -v
```

### 2. Prepare the input

Create one manifest and one working directory per outer fold. Assign paired scans from the same patient to the same fold. See the [manifest example and target encoding](docs/workflow.md#prepare-data-and-folds).

### 3. Train and predict

Follow the [ICA commands](docs/workflow.md#train-and-predict) to train the localizer and segmenter, run held-out predictions, and restore the masks to the input CTA grid. The [binary CoW workflow](docs/workflow.md#binary-cow-evaluation) specifies the CoW trainer and five-fold ensemble commands.

### 4. Evaluate

Use `code/scripts/evaluate.py` to evaluate ICA masks against corresponding annotations on the original grid. Use `code/scripts/evaluate_cow.py` with an official CoW ROI for binary CoW evaluation. The [workflow](docs/workflow.md#evaluate) lists the inputs and command examples.
