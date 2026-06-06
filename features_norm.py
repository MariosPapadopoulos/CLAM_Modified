import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse



def collect_args():
    parser = argparse.ArgumentParser(description='Normalize features')
    parser.add_argument('--features_folder', type=str, default='./LUAD_LUSC/pt_files/resnet50', help='Path to the folder containing feature files')
    parser.add_argument('--output_folder', type=str, default='./LUAD_LUSC/pt_files/resnet50_normalized', help='Path to the folder to save normalized features')
    return parser.parse_args()


def needs_normalization(features: torch.Tensor, tol: float = 1e-3) -> bool:
    """Return True if any row norm deviates from 1.0 by more than tol."""
    norms = features.norm(p=2, dim=1)
    max_dev = (norms - 1.0).abs().max().item()
    print(f"  L2 norm — min: {norms.min():.6f}, max: {norms.max():.6f}, "
          f"mean: {norms.mean():.6f}, max_deviation_from_1: {max_dev:.6f}")
    return max_dev > tol


def verify_normalized(path: str, tol: float = 1e-3) -> bool:
    """Load a saved .pt file and confirm every row is unit-normalized."""
    features = torch.load(path)
    norms = features.norm(p=2, dim=1)
    max_dev = (norms - 1.0).abs().max().item()
    ok = max_dev <= tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {os.path.basename(path)}: max norm deviation = {max_dev:.2e}")
    return ok


def main():
    args = collect_args()

    features_folder = args.features_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    feature_files = [f for f in os.listdir(features_folder) if f.endswith('.pt')]

    d = None
    verification_failures = []
    for feature_file in feature_files:
        feature_path = os.path.join(features_folder, feature_file)
        features = torch.load(feature_path)

        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Expected tensor in {feature_path}, got {type(features)}")

        if features.ndim == 1:
            features = features.unsqueeze(0)
        elif features.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix in {feature_path}, got shape {features.shape}")

        # Ensure same feature dimension across files
        if d is None:
            d = features.size(1)
        elif features.size(1) != d:
            raise ValueError(
                f"Dimension mismatch: file {feature_file} has d={features.size(1)} but previous files have d={d}"
            )

        print(f"[{feature_file}] Pre-normalization norms:")
        already_normalized = not needs_normalization(features)
        if already_normalized:
            print(f"  -> already unit-normalized, skipping")

        # Unit-normalize each patch feature vector
        normalized_features = F.normalize(features, p=2, dim=1)

        output_path = os.path.join(output_folder, feature_file)
        torch.save(normalized_features, output_path)
        print(f"  Saved: {output_path} | shape: {normalized_features.shape}")

        print(f"[{feature_file}] Post-save verification:")
        if not verify_normalized(output_path):
            verification_failures.append(feature_file)

    if verification_failures:
        raise RuntimeError(f"Normalization verification failed for: {verification_failures}")
    print(f"\nAll {len(feature_files)} file(s) normalized and verified.")

if __name__ == "__main__":
    main()