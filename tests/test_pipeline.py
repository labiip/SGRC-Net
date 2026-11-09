"""NIfTI/manifest adapter roundtrip on generated arrays."""
from pathlib import Path
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "code/scripts" / name), *map(str, args)], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")


def save(path, data, affine):
    nib.save(nib.Nifti1Image(data, affine), path)


class AdapterTests(unittest.TestCase):
    def test_entrypoints_import_without_running(self):
        for path in sorted((ROOT / "code/scripts").glob("*.py")):
            spec = importlib.util.spec_from_file_location("check_" + path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))

    def test_roi_rejects_encoded_core_as_binary_foreground(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp); shape = (8, 8, 8); affine = np.eye(4)
            image = np.zeros(shape, dtype=np.float32)
            label = np.zeros(shape, dtype=np.uint8); label[2:4, 2:4, 2:4] = 4
            save(t / "image.nii.gz", image, affine)
            save(t / "encoded.nii.gz", label, affine)
            manifest = t / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "subject_id", "role", "image", "label"])
                writer.writeheader()
                writer.writerow({"id": "sample-a", "subject_id": "sample-a", "role": "train",
                    "image": "image.nii.gz", "label": "encoded.nii.gz"})
            command = [sys.executable, "-X", "utf8", str(ROOT / "code/scripts/prepare_roi.py"),
                "--manifest", str(manifest), "--output", str(t / "bad-roi")]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("binary", result.stderr.lower())

    def test_subject_roles_cannot_leak(self):
        from sgrc_net.io import read_manifest
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "subject_id", "role", "image"])
                writer.writeheader()
                writer.writerow({"id": "variant-a", "subject_id": "subject-a", "role": "train", "image": "a.nii.gz"})
                writer.writerow({"id": "variant-b", "subject_id": "subject-a", "role": "test", "image": "b.nii.gz"})
            with self.assertRaisesRegex(ValueError, "cannot cross roles"):
                read_manifest(manifest)

    def test_encoding_crop_restore_and_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp); affine = np.diag([2., 2., 2., 1.])
            shape = (32, 36, 40)
            cta = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            states = np.zeros(shape, dtype=np.uint8)
            states[8:24, 10:26, 10:30] = 1
            path = np.zeros(shape, dtype=np.uint8); path[13:16, 16, 16] = 1
            core = np.zeros(shape, dtype=np.uint8); core[4:7, 4:7, 4:7] = 1
            for name, data in (("image", cta), ("states", states), ("path", path), ("core", core)):
                save(t / f"{name}.nii.gz", data, affine)
            encoded_path = t / "encoded.nii.gz"
            run_script("encode_targets.py", "--states", t / "states.nii.gz", "--path", t / "path.nii.gz",
                "--core", t / "core.nii.gz", "--output", encoded_path)
            encoded = np.asarray(nib.load(encoded_path).dataobj)
            self.assertTrue((encoded[path > 0] == 3).all())
            self.assertTrue((encoded[core > 0] == 4).all())
            manifest = t / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "subject_id", "role", "image", "label", "encoded_label"])
                writer.writeheader()
                for name, role in (("sample-a", "train"), ("sample-b", "val"), ("sample-c", "test")):
                    writer.writerow({"id": name, "subject_id": name, "role": role,
                        "image": "image.nii.gz", "label": "states.nii.gz" if role != "test" else "not-read.nii.gz",
                        "encoded_label": "encoded.nii.gz" if role != "test" else "not-read.nii.gz"})
            roi_root = t / "roi"
            run_script("prepare_roi.py", "--manifest", manifest, "--output", roi_root)
            metadata = json.loads((roi_root / "control/roi_metadata.json").read_text())
            self.assertEqual(len(metadata["slices"]), 27)
            # Identical generated test masks let us verify fusion independent of a learned model.
            pred = t / "roi_pred"; pred.mkdir()
            for row in metadata["slices"]:
                image = nib.load(roi_root / "roi_inputs" / f"{row['slice_id']}_0000.nii.gz")
                labels = np.zeros(image.shape, dtype=np.uint8); labels[3:-3, 3:-3] = 1
                save(pred / f"{row['slice_id']}.nii.gz", labels, image.affine)
            run_script("fuse_roi.py", "--metadata", roi_root / "control/roi_metadata.json", "--predictions", pred, "--output", t / "fused.json")
            self.assertEqual(set(json.loads((t / "fused.json").read_text())), {"sample-a", "sample-b", "sample-c"})
            # Use a deliberately partial valid crop to test original-space placement.
            bbox = {"x0": 2, "x1": 28, "y0": 3, "y1": 31, "z0": 4, "z1": 35}
            (t / "rois.json").write_text(json.dumps({name: bbox for name in ("sample-a", "sample-b", "sample-c")}))
            seg_root = t / "seg"
            run_script("prepare_segmentation.py", "--manifest", manifest, "--rois", t / "rois.json", "--output", seg_root)
            self.assertEqual(len(list((seg_root / "nnUNet_raw/Dataset731_SGRCNet/labelsTr").glob("*.nii.gz"))), 2)
            crop_input = nib.load(seg_root / "imagesTs/sample-c_0000.nii.gz")
            self.assertEqual(crop_input.shape, (26, 28, 31))
            np.testing.assert_array_equal(np.asarray(crop_input.dataobj), cta[2:28, 3:31, 4:35])
            predictions = t / "predictions"; predictions.mkdir()
            binary = np.ones(crop_input.shape, dtype=np.uint8)
            save(predictions / "sample-c.nii.gz", binary, crop_input.affine)
            run_script("restore.py", "--metadata", seg_root / "control/restore.json", "--predictions", predictions, "--output", t / "restored")
            result = nib.load(t / "restored/sample-c.nii.gz")
            expected = np.zeros(shape, dtype=np.uint8); expected[2:28, 3:31, 4:35] = 1
            np.testing.assert_array_equal(np.asarray(result.dataobj), expected)
            np.testing.assert_array_equal(result.affine, affine)


if __name__ == "__main__":
    unittest.main()
