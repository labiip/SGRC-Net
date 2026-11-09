"""Topology target construction: use only training/validation annotation masks."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, label as connected_components
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

def nearest_visible(anchor: np.ndarray, visible_points: np.ndarray) -> tuple[int, int, int]:
    anchor_points = np.argwhere(anchor)
    if not anchor_points.size:
        raise RuntimeError("empty topology anchor")
    center = anchor_points.mean(axis=0)
    _, index = cKDTree(visible_points).query(center, k=1)
    return tuple(int(value) for value in visible_points[int(index)])

def stenosis_path(
    visible: np.ndarray,
    lesion_roi: np.ndarray,
    proximal: np.ndarray,
    distal: np.ndarray,
) -> np.ndarray:
    focus = lesion_roi | proximal | distal
    focus_points = np.argwhere(focus)
    if not focus_points.size:
        raise RuntimeError("empty lesion/anchor focus")
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for margin in (24, 48, 96, 160):
        lower = np.maximum(focus_points.min(axis=0) - margin, 0)
        upper = np.minimum(
            focus_points.max(axis=0) + margin + 1, np.asarray(visible.shape)
        )
        slices = tuple(slice(int(low), int(high)) for low, high in zip(lower, upper))
        local_visible = visible[slices]
        visible_points = np.argwhere(local_visible)
        if not visible_points.size:
            continue
        local_start = nearest_visible(proximal[slices], visible_points)
        local_end = nearest_visible(distal[slices], visible_points)
        components, _ = connected_components(local_visible, structure=structure)
        component = int(components[local_start])
        if component == 0 or int(components[local_end]) != component:
            continue
        corridor = components == component
        cost = np.where(corridor, 1.0, 1_000_000.0).astype(np.float32)
        route, _ = route_through_array(
            cost, local_start, local_end, fully_connected=True, geometric=True
        )
        route_array = np.asarray(route, dtype=np.int64)
        if not bool(corridor[tuple(route_array.T)].all()):
            continue
        local_path = np.zeros(corridor.shape, dtype=bool)
        local_path[tuple(route_array.T)] = True
        local_path = binary_dilation(
            local_path, structure=np.ones((3, 3, 3), dtype=bool), iterations=1
        ) & corridor
        return np.argwhere(local_path).astype(np.int64) + lower
    raise RuntimeError(
        "audited stenosis anchors are not connected in the visible target "
        "within the maximum adaptive topology corridor"
    )
