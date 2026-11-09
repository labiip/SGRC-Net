"""Encode main states plus path/core supervision into a single label volume."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from sgrc_net.io import load_label, save_image
from sgrc_net.targets import stenosis_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", type=Path, required=True, help="0 background, 1 remaining lumen, 2 lesion-associated lumen")
    p.add_argument("--path", type=Path, help="Binary visible-stenosis path mask")
    p.add_argument("--core", type=Path, help="Binary non-enhancing occlusion-core mask")
    p.add_argument("--lesion-roi", type=Path)
    p.add_argument("--proximal-anchor", type=Path)
    p.add_argument("--distal-anchor", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    image = nib.load(a.states)
    states = np.asarray(image.dataobj)
    if not set(np.unique(states)).issubset({0, 1, 2}): raise ValueError("Main states must be 0, 1, 2")
    path = np.zeros(states.shape, dtype=bool)
    if a.path:
        path = load_label(a.path, image) > 0
    elif any((a.lesion_roi, a.proximal_anchor, a.distal_anchor)):
        if not all((a.lesion_roi, a.proximal_anchor, a.distal_anchor)):
            raise ValueError("All three lesion/anchor masks are required to construct a path")
        coordinates = stenosis_path(states > 0, load_label(a.lesion_roi, image) > 0,
            load_label(a.proximal_anchor, image) > 0, load_label(a.distal_anchor, image) > 0)
        path[tuple(coordinates.T)] = True
    core = load_label(a.core, image) > 0 if a.core else np.zeros(states.shape, dtype=bool)
    if np.any(path & ~np.isin(states, (1, 2))):
        raise ValueError("Path voxels must lie within visible lumen (states 1 or 2)")
    if np.any(core & (states != 0)) or np.any(path & core):
        raise ValueError("Core must be background and disjoint from visible path")
    encoded = states.astype(np.uint8)
    encoded[path] = 3; encoded[core] = 4
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(encoded, image.affine, image.header, a.output, np.uint8)


if __name__ == "__main__":
    main()
