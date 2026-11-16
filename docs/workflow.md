# SGRC-Net workflow

## Prepare data and folds

Use NIfTI CTA volumes with voxel axes ordered left–right, anterior–posterior, and inferior–superior. Either axis sign is accepted. Each annotation must share its image's shape and affine. Prepare a common anatomical grid before processing oblique scans.

Prepare one UTF-8 CSV manifest and a separate working directory per outer fold. Assign all scans from the same patient, including each reference/variant pair, to the same role in that fold. Each prepared view contains one nnU-Net split and uses `--fold 0`.

```csv
id,subject_id,role,image,label,encoded_label
scanA-reference,subjectA,train,input/a-reference.nii.gz,input/a-reference-lumen.nii.gz,input/a-reference-encoded.nii.gz
scanA-variant,subjectA,train,input/a-variant.nii.gz,input/a-variant-lumen.nii.gz,input/a-variant-encoded.nii.gz
scanB,subjectB,val,input/b.nii.gz,input/b-lumen.nii.gz,input/b-encoded.nii.gz
scanC,subjectC,test,input/c.nii.gz,,
```

`label` is the binary visible-lumen mask for ROI learning. `encoded_label` stores background (0), remaining visible lumen (1), lesion-associated visible lumen (2), visible stenosis path (3), and non-enhancing occlusion core (4). The loader maps 3 to main state 2 and 4 to main state 0 while retaining their separate supervision. For each training or validation scan, encode an existing main-state annotation with its available path/core masks. Paths in the CSV are relative to the manifest or absolute. Identifiers use letters, digits, and hyphens.

```bash
python code/scripts/encode_targets.py --states states.nii.gz \
  --path path.nii.gz --core core.nii.gz --output encoded.nii.gz
```

Omit a path or core mask when it is absent. To derive a path from a reference lumen annotation, replace `--path` with `--lesion-roi`, `--proximal-anchor`, and `--distal-anchor`. Reference scans use main states 0 and 1. For inference-only data, use test rows and pass the trained view's `work/roi/control/roi_metadata.json` to `prepare_roi.py --prior`.

## Train and predict

Run these commands for each prepared outer-fold manifest, using distinct `work` paths. The example uses seed 0. [Method settings](config/method.json) record the reference spacing, patch, and model settings.

```bash
python code/scripts/prepare_roi.py --manifest manifest.csv --output work/roi
python code/scripts/preprocess.py --root work/roi --dataset Dataset730_SGRCROI --stage roi
export nnUNet_raw="$PWD/work/roi/nnUNet_raw"
export nnUNet_preprocessed="$PWD/work/roi/nnUNet_preprocessed"
export nnUNet_results="$PWD/work/roi/nnUNet_results"
python code/scripts/train.py --dataset 730 --stage roi --fold 0 --seed 0
python code/scripts/predict.py --input work/roi/roi_inputs --output work/roi_predictions \
  --stage roi --checkpoint checkpoint_final.pth \
  --model "$nnUNet_results/Dataset730_SGRCROI/nnUNetTrainerSGRCROI__nnUNetPlans__2d"
python code/scripts/fuse_roi.py --metadata work/roi/control/roi_metadata.json \
  --predictions work/roi_predictions --output work/rois.json

python code/scripts/prepare_segmentation.py --manifest manifest.csv \
  --rois work/rois.json --output work/segmentation
python code/scripts/preprocess.py --root work/segmentation --dataset Dataset731_SGRCNet --stage segmentation
export nnUNet_raw="$PWD/work/segmentation/nnUNet_raw"
export nnUNet_preprocessed="$PWD/work/segmentation/nnUNet_preprocessed"
export nnUNet_results="$PWD/work/segmentation/nnUNet_results"
python code/scripts/train.py --dataset 731 --stage segmentation --fold 0 --seed 0
python code/scripts/predict.py --input work/segmentation/imagesTs \
  --output work/crop_predictions --checkpoint checkpoint_best.pth \
  --model "$nnUNet_results/Dataset731_SGRCNet/nnUNetTrainerSGRCNet__nnUNetPlans__3d_fullres"
python code/scripts/restore.py --metadata work/segmentation/control/restore.json \
  --predictions work/crop_predictions --output work/original_grid_predictions
```

ROI training uses 100 epochs and segmentation uses 250. The segmentation script merges the two predicted visible states into foreground before restoring the mask to the original CTA grid.

## Evaluate

Evaluate each held-out prediction against its corresponding reference mask on the original grid. Whole-tree Dice and IoU use the entire mask. Lesion Dice uses the fixed lesion ROI. The two anchor masks define connectivity through the 26-connected components of the complete predicted binary mask. For occlusion, core intrusion is any predicted foreground inside the core mask.

```bash
python code/scripts/evaluate.py --prediction prediction.nii.gz --reference reference-lumen.nii.gz \
  --lesion-roi lesion-roi.nii.gz --lesion-type stenosis \
  --proximal-anchor proximal.nii.gz --distal-anchor distal.nii.gz
```

For an occlusion, set `--lesion-type occlusion` and supply `--occlusion-core core.nii.gz`. Add `--paired-reference-prediction` to measure agreement outside the 5-mm expansion of the lesion ROI. Evaluate all held-out cases, then summarize per-volume overlap and event counts for the respective lesion groups.

## Binary CoW evaluation

The [TopCoW challenge data release](https://zenodo.org/records/15692630) provides the TopCoW CTA data and the ISLES 2024 (TUM) external test set. Merge annotated CoW vessels into binary foreground. The CoW trainer uses the same SGRC-Net calibration architecture, merges the two foreground logits with log-sum-exp, and optimizes one full-resolution binary DiceCE loss.

**Prepare the folds.** Sort the TopCoW case stems and shuffle them with Python `random.Random(20261012)`. Divide the shuffled list into five equal consecutive outer test folds numbered 0 to 4. For each fold, reshuffle the non-test cases in their current order using seed `20261013 + outer_fold`. Use the first 13 cases for validation and the remainder for training. Prepare a separate nnU-Net view per outer fold, with one entry in `splits_final.json`. Derive the CT normalization statistics and median target spacing from the training cases in that view. Use `NibabelIOWithReorient`, a 96 × 96 × 96 patch, batch size 1, and six-stage `PlainConvUNet` with isotropic stride 2 after the first stage. The trainer expects these preprocessed views and binary labels.

**Train and predict.** With the selected outer fold's nnU-Net directories set, run the following example for outer fold 0. Increase `SGRC_COW_SEED` by the outer fold index. The trainer uses SGD with polynomial learning-rate decay, 250 epochs, 250 training and 50 validation iterations per epoch, and probabilistic foreground sampling with probability 0.5.

```bash
SGRC_COW_SEED=20261012 nnUNet_compile=false nnUNetv2_train "$DATASET_ID" 3d_fullres 0 \
  -tr nnUNetTrainerSGRCNetCoW -p nnUNetPlans
python code/scripts/predict_cow.py predict --image input.nii.gz --model cow-model-fold0 \
  --checkpoint checkpoint_best.pth --output fold0.nii.gz --probabilities fold0.npz
```

For TopCoW, use the model from the fold that held out that scan. For each ISLES scan, predict with all five models trained on TopCoW. Each `.npz` stores the restored background and foreground probabilities plus the grid metadata. Combine the five predictions with equal weights before taking binary argmax.

```bash
python code/scripts/predict_cow.py ensemble --folds fold0.npz fold1.npz fold2.npz fold3.npz fold4.npz \
  --output isles-ensemble.nii.gz
```

**Score the official ROI.** Apply the CoW ROI after full-volume prediction. The evaluator reads the official ROI text file, checks the original prediction/reference geometry, and calls the pinned TopCoW implementations for Dice, HD95 in mm, and clDice.

```bash
git clone https://github.com/CoWBenchmark/TopCoW_Eval_Metrics.git
git -C TopCoW_Eval_Metrics checkout 25ca8399e056e71e3ff6743ac4a47ce28624cb3d
python code/scripts/evaluate_cow.py --prediction isles-ensemble.nii.gz \
  --reference binary-reference.nii.gz --roi cow-roi.txt --metrics-repo TopCoW_Eval_Metrics
```


The implementation uses [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for its global segmenter, preprocessing, and prediction. See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream attribution.
