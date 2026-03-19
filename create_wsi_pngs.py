import os, glob
import h5py
import numpy as np
from PIL import Image
import openslide
import argparse

def collect_args():
    parser = argparse.ArgumentParser(description="Create PNG patches from WSIs based on H5 coordinates.")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory containing .h5 files with 'coords'.")
    parser.add_argument("--wsi_dir", type=str, required=True, help="Directory containing WSI files.")
    parser.add_argument("--out_root", type=str, required=True, help="Root directory to save output PNGs.")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the square patch to extract.")
    parser.add_argument("--patch_level", type=int, default=0, help="Pyramid level to extract patches from (0 is highest resolution).")
    return parser.parse_args()

def make_pngs_for_slide(h5_path, wsi_path, out_dir, patch_size=256, patch_level=0):

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]  # level-0 coords

    slide = openslide.OpenSlide(wsi_path)
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]

    for i, (x, y) in enumerate(coords):
        patch = slide.read_region(
            (int(x), int(y)),      # level-0 coords
            patch_level,           # <-- MUST be the same as the one we used in CLAM patch extraction
            (patch_size, patch_size)
        ).convert("RGB")

        patch.save(os.path.join(out_dir, f"{slide_name}_{i:06d}_x{int(x)}_y{int(y)}.png"))

    slide.close()

def batch_make_pngs(h5_dir, wsi_dir, out_root, wsi_exts=(".tif",".tiff",".svs",".ndpi",".mrxs"), patch_size=256, patch_level=0):
    os.makedirs(out_root, exist_ok=True)

    h5_paths = sorted(glob.glob(os.path.join(h5_dir, "*.h5")))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    # build a map from basename -> full wsi path
    wsi_map = {}
    for ext in wsi_exts:
        for p in glob.glob(os.path.join(wsi_dir, f"*{ext}")):
            wsi_map[os.path.splitext(os.path.basename(p))[0]] = p

    missing = []
    for h5_path in h5_paths:
        stem = os.path.splitext(os.path.basename(h5_path))[0]  # e.g. normal_001
        wsi_path = wsi_map.get(stem, None)
        if wsi_path is None:
            missing.append(stem)
            continue

        out_dir = os.path.join(out_root, stem)
        make_pngs_for_slide(h5_path, wsi_path, out_dir, patch_size=patch_size, patch_level=patch_level)

    if missing:
        print("No matching WSI found for:", missing)

if __name__ == "__main__":
    args= collect_args()

    batch_make_pngs(
        h5_dir=args.h5_dir,
        wsi_dir=args.wsi_dir,
        out_root=args.out_root,
        patch_size=args.patch_size,
        patch_level=args.patch_level
    )