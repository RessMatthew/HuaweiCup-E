import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
import argparse

def mmd_distance(X, Y, gamma=1.0):
    """Calculate Maximum Mean Discrepancy (MMD) between two datasets"""
    n_X, n_Y = len(X), len(Y)

    # Calculate kernel matrices
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    # Calculate MMD^2
    mmd_squared = (1/(n_X*(n_X-1))) * np.sum(K_XX - np.diag(K_XX)) + \
                  (1/(n_Y*(n_Y-1))) * np.sum(K_YY - np.diag(K_YY)) - \
                  (2/(n_X*n_Y)) * np.sum(K_XY)

    return np.sqrt(max(0, mmd_squared))

def mmd_alignment_fast(X_source, X_target, gamma=1.0, n_components=10):
    """
    Fast MMD alignment using kernel PCA approach
    """
    n_source, n_target = X_source.shape[0], X_target.shape[0]

    # Center the data
    source_mean = np.mean(X_source, axis=0)
    target_mean = np.mean(X_target, axis=0)

    X_source_centered = X_source - source_mean
    X_target_centered = X_target - target_mean

    # Compute covariance matrices
    source_cov = np.cov(X_source_centered.T)
    target_cov = np.cov(X_target_centered.T)

    # Joint covariance
    joint_cov = (source_cov + target_cov) / 2

    # Regularize to avoid singularity
    joint_cov += 1e-6 * np.eye(joint_cov.shape[0])

    # Compute transformation matrix using joint covariance
    try:
        # Whitening transformation
        joint_sqrt_inv = np.linalg.inv(np.linalg.cholesky(joint_cov))

        # Apply transformation to target domain
        X_target_aligned = (X_target_centered @ joint_sqrt_inv) + source_mean

    except np.linalg.LinAlgError:
        # Fallback to simple mean and variance matching
        X_target_aligned = X_target.copy()
        for i in range(X_target.shape[1]):
            # Match mean and variance to source domain
            target_std = np.std(X_target[:, i])
            source_std = np.std(X_source[:, i])
            if target_std > 1e-6:
                X_target_aligned[:, i] = (X_target[:, i] - target_mean[i]) * (source_std / target_std) + source_mean[i]
            else:
                X_target_aligned[:, i] = X_target[:, i] - target_mean[i] + source_mean[i]

    return X_target_aligned

def main():
    parser = argparse.ArgumentParser(description='MMD-based domain alignment')
    parser.add_argument('--source', type=str, default='data/data_out.csv',
                       help='Source domain CSV file path')
    parser.add_argument('--target', type=str, default='data/t_data_out.csv',
                       help='Target domain CSV file path')
    parser.add_argument('--output', type=str, default='data/t_data_aligned.csv',
                       help='Output aligned target domain CSV file path')
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='RBF kernel bandwidth parameter')

    args = parser.parse_args()

    # Load source data (skip label column)
    print("Loading source domain data...")
    source_data = pd.read_csv(args.source)
    X_source = source_data.iloc[:, 1:].values  # Skip first column (label)

    # Load target data
    print("Loading target domain data...")
    target_data = pd.read_csv(args.target)
    X_target = target_data.values

    print(f"Source domain shape: {X_source.shape}")
    print(f"Target domain shape: {X_target.shape}")

    # Calculate initial MMD
    initial_mmd = mmd_distance(X_source, X_target, gamma=args.gamma)
    print(f"Initial MMD distance: {initial_mmd:.6f}")

    # Perform MMD alignment
    print("Performing MMD alignment...")
    X_target_aligned = mmd_alignment_fast(X_source, X_target, gamma=args.gamma)

    # Calculate final MMD
    final_mmd = mmd_distance(X_source, X_target_aligned, gamma=args.gamma)
    print(f"Final MMD distance: {final_mmd:.6f}")
    print(f"MMD reduction: {((initial_mmd - final_mmd) / initial_mmd * 100):.2f}%")

    # Save aligned target data
    aligned_df = pd.DataFrame(X_target_aligned, columns=target_data.columns)
    aligned_df.to_csv(args.output, index=False)
    print(f"Aligned target data saved to: {args.output}")

if __name__ == "__main__":
    main()