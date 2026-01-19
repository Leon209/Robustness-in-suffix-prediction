"""
Visualization backend for comparing clean and perturbed encoded datasets.

This module provides functions to load encoded prefix data, reduce dimensionality,
and visualize clean vs perturbed datasets in 2D space.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_and_extract_prefixes(file_path: str) -> np.ndarray:
    """
    Load pickle file and extract flattened prefix features.
    
    Args:
        file_path: Path to pickle file containing encoded data
        
    Returns:
        numpy array of shape (n_samples, n_features) with flattened prefix features
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data structure is unexpected
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the pickle file
    data = torch.load(file_path, weights_only=False)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary, got {type(data)}")
    
    # Extract and flatten prefixes
    flattened_prefixes = []
    
    for (case_id, prefix_len), (prefix, suffix) in data.items():
        # Prefix is a tuple: (cat_tensors, num_tensors)
        cat_tensors, num_tensors = prefix
        
        # Flatten categorical tensors
        cat_features = []
        for tensor in cat_tensors:
            # Remove batch dimension and flatten: (1, window_size) -> (window_size,)
            flattened = tensor.squeeze(0).flatten().numpy()
            cat_features.append(flattened)
        
        # Flatten numerical tensors
        num_features = []
        for tensor in num_tensors:
            # Remove batch dimension and flatten: (1, window_size) -> (window_size,)
            flattened = tensor.squeeze(0).flatten().numpy()
            num_features.append(flattened)
        
        # Concatenate all features
        if cat_features:
            cat_combined = np.concatenate(cat_features)
        else:
            cat_combined = np.array([])
        
        if num_features:
            num_combined = np.concatenate(num_features)
        else:
            num_combined = np.array([])
        
        # Final feature vector: categorical + numerical
        if len(cat_combined) > 0 and len(num_combined) > 0:
            feature_vector = np.concatenate([cat_combined, num_combined])
        elif len(cat_combined) > 0:
            feature_vector = cat_combined
        elif len(num_combined) > 0:
            feature_vector = num_combined
        else:
            raise ValueError(f"Empty prefix for case {case_id}, prefix_len {prefix_len}")
        
        flattened_prefixes.append(feature_vector)
    
    # Convert to numpy array
    if len(flattened_prefixes) == 0:
        raise ValueError("No prefixes found in data")
    
    # Check that all prefixes have the same dimensionality
    feature_dim = len(flattened_prefixes[0])
    for i, prefix in enumerate(flattened_prefixes):
        if len(prefix) != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimensions: prefix {i} has {len(prefix)} features, "
                f"expected {feature_dim}"
            )
    
    return np.array(flattened_prefixes)


def reduce_to_2d(data: np.ndarray, pca_model: Optional[PCA] = None) -> Tuple[np.ndarray, PCA]:
    """
    Reduce data to 2D using PCA.
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        pca_model: Optional pre-fitted PCA model. If None, fits a new one.
        
    Returns:
        Tuple of (2d_coordinates, pca_model) where:
        - 2d_coordinates: numpy array of shape (n_samples, 2)
        - pca_model: The fitted PCA model
    """
    if pca_model is None:
        pca_model = PCA(n_components=2)
        data_2d = pca_model.fit_transform(data)
    else:
        data_2d = pca_model.transform(data)
    
    return data_2d, pca_model


def visualize_datasets(
    clean_data_path: str,
    perturbed_data_path: str,
    output_path: Optional[str] = None,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize clean and perturbed datasets in 2D space.
    
    Args:
        clean_data_path: Path to clean dataset pickle file
        perturbed_data_path: Path to perturbed dataset pickle file
        output_path: Optional path to save the plot. If None, displays the plot.
        alpha: Transparency level for scatter points (default: 0.6)
        figsize: Figure size tuple (default: (12, 8))
    """
    print(f"Loading clean dataset from: {clean_data_path}")
    clean_data = load_and_extract_prefixes(clean_data_path)
    print(f"Loaded {len(clean_data)} clean samples with {clean_data.shape[1]} features")
    
    print(f"Loading perturbed dataset from: {perturbed_data_path}")
    perturbed_data = load_and_extract_prefixes(perturbed_data_path)
    print(f"Loaded {len(perturbed_data)} perturbed samples with {perturbed_data.shape[1]} features")
    
    # Check feature dimensions match
    if clean_data.shape[1] != perturbed_data.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: clean has {clean_data.shape[1]} features, "
            f"perturbed has {perturbed_data.shape[1]} features"
        )
    
    # Combine data for PCA fitting
    print("Combining datasets for PCA fitting...")
    combined_data = np.vstack([clean_data, perturbed_data])
    
    # Fit PCA on combined data
    print("Fitting PCA model...")
    pca_model = PCA(n_components=2)
    combined_2d = pca_model.fit_transform(combined_data)
    
    # Split back into clean and perturbed
    n_clean = len(clean_data)
    clean_2d = combined_2d[:n_clean]
    perturbed_2d = combined_2d[n_clean:]
    
    # Calculate explained variance
    explained_variance = pca_model.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    print(f"PCA explained variance: PC1={explained_variance[0]:.2%}, "
          f"PC2={explained_variance[1]:.2%}, Total={total_variance:.2%}")
    
    # Create visualization
    plt.figure(figsize=figsize)
    
    # Plot clean data (blue)
    plt.scatter(
        clean_2d[:, 0],
        clean_2d[:, 1],
        c='blue',
        alpha=alpha,
        label='Clean Data',
        s=20
    )
    
    # Plot perturbed data (orange)
    plt.scatter(
        perturbed_2d[:, 0],
        perturbed_2d[:, 1],
        c='orange',
        alpha=alpha,
        label='Perturbed Data',
        s=20
    )
    
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)', fontsize=12)
    plt.title('Clean vs Perturbed Dataset Visualization (2D PCA)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
