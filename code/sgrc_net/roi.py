"""ROI geometry operations used by the fixed nine-slice localization stage."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

@dataclass(frozen=True)
class BBox:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.x0, self.x1),
            slice(self.y0, self.y1),
            slice(self.z0, self.z1),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            key: int(getattr(self, key))
            for key in ("x0", "x1", "y0", "y1", "z0", "z1")
        }

def bbox_from_mask(mask: np.ndarray) -> BBox:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("empty mask")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    return BBox(
        int(lower[0]),
        int(upper[0]),
        int(lower[1]),
        int(upper[1]),
        int(lower[2]),
        int(upper[2]),
    )

def expand_bbox(
    bbox: BBox,
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    margin_mm: float,
) -> BBox:
    margins = [
        int(math.ceil(float(margin_mm) / max(float(value), 1e-6)))
        for value in spacing
    ]
    return BBox(
        max(0, bbox.x0 - margins[0]),
        min(shape[0], bbox.x1 + margins[0]),
        max(0, bbox.y0 - margins[1]),
        min(shape[1], bbox.y1 + margins[1]),
        max(0, bbox.z0 - margins[2]),
        min(shape[2], bbox.z1 + margins[2]),
    )

def shifted_affine(affine: np.ndarray, bbox: BBox) -> np.ndarray:
    offset = np.eye(4, dtype=np.float64)
    offset[:3, 3] = [bbox.x0, bbox.y0, bbox.z0]
    return np.asarray(affine) @ offset

def save_nifti_like(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    destination: str | Path,
    dtype,
) -> None:
    output_header = header.copy()
    output_header.set_data_dtype(dtype)
    nib.save(
        nib.Nifti1Image(data.astype(dtype, copy=False), affine, output_header),
        str(destination),
    )

def plane_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[:, :, index]
    if plane == "sagittal":
        return volume[index, :, :]
    if plane == "coronal":
        return volume[:, index, :]
    raise ValueError(plane)

def plane_size(shape: tuple[int, int, int], plane: str) -> int:
    return {"sagittal": shape[0], "coronal": shape[1], "axial": shape[2]}[plane]

def plane_spacing(
    spacing: tuple[float, float, float], plane: str
) -> tuple[float, float]:
    if plane == "axial":
        return spacing[0], spacing[1]
    if plane == "sagittal":
        return spacing[1], spacing[2]
    if plane == "coronal":
        return spacing[0], spacing[2]
    raise ValueError(plane)

def envelope(
    mask: np.ndarray,
    spacing: tuple[float, float],
    margin_mm: float,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    distance = distance_transform_edt(~mask.astype(bool), sampling=spacing)
    return (distance <= margin_mm).astype(np.uint8)

def save_slice(
    data: np.ndarray,
    spacing: tuple[float, float],
    destination: Path,
    dtype,
) -> None:
    affine = np.diag([spacing[0], spacing[1], 1.0, 1.0])
    header = nib.Nifti1Header()
    save_nifti_like(data[:, :, None], affine, header, destination, dtype)

def bbox2d(path: Path) -> tuple[int, int, int, int] | None:
    data = np.asarray(nib.load(str(path)).dataobj)
    if data.ndim == 3:
        data = data[:, :, 0]
    coordinates = np.argwhere(data > 0)
    if coordinates.size == 0:
        return None
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    return int(lower[0]), int(upper[0]), int(lower[1]), int(upper[1])

def union_longitudinal_fallback(
    fused: list[int],
    normalized_fallback: list[float],
    shape: tuple[int, int, int],
) -> tuple[list[int], bool]:
    """Preserve predicted x/y while protecting the long cranio-caudal extent."""
    result = list(fused)
    fallback_z0 = int(round(float(normalized_fallback[4]) * shape[2]))
    fallback_z1 = int(round(float(normalized_fallback[5]) * shape[2]))
    original = result[4], result[5]
    result[4] = min(result[4], fallback_z0)
    result[5] = max(result[5], fallback_z1)
    return result, (result[4], result[5]) != original
