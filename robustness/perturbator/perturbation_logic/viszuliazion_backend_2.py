"""
Interactive visualization backend for comparing clean and perturbed encoded datasets.

This module provides functions to load encoded prefix data with metadata, match pairs,
and create interactive visualizations with hover functionality using Plotly.
"""

import os
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
from sklearn.decomposition import PCA

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Install with: pip install plotly")


def load_data_with_prefixes(file_path: str) -> Tuple[np.ndarray, List[Dict], List]:
    """
    Load pickle file and extract flattened prefix features along with metadata.
    
    Args:
        file_path: Path to pickle file containing encoded data
        
    Returns:
        Tuple of:
        - numpy array of shape (n_samples, n_features) with flattened prefix features
        - List of metadata dicts: [{'case_id': ..., 'prefix_len': ..., 'prefix': ...}, ...]
        - List of keys: [(case_id, prefix_len), ...]
        
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
    metadata_list = []
    keys_list = []
    
    for (case_id, prefix_len), (prefix, suffix) in data.items():
        keys_list.append((case_id, prefix_len))
        
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
        
        # Store metadata
        metadata_list.append({
            'case_id': case_id,
            'prefix_len': prefix_len,
            'prefix': prefix,  # Store original prefix tuple
            'cat_tensors': cat_tensors,
            'num_tensors': num_tensors
        })
    
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
    
    return np.array(flattened_prefixes), metadata_list, keys_list


def match_clean_perturbed_pairs(clean_keys: List, perturbed_keys: List) -> Dict[int, int]:
    """
    Match clean and perturbed entries by (case_id, prefix_length).
    
    Args:
        clean_keys: List of (case_id, prefix_length) tuples from clean dataset
        perturbed_keys: List of (case_id, prefix_length) tuples from perturbed dataset
        
    Returns:
        Dictionary mapping clean index to perturbed index: {clean_idx: perturbed_idx}
    """
    # Create mapping from key to index for perturbed dataset
    perturbed_key_to_idx = {key: idx for idx, key in enumerate(perturbed_keys)}
    
    # Match clean indices to perturbed indices
    matching = {}
    for clean_idx, clean_key in enumerate(clean_keys):
        if clean_key in perturbed_key_to_idx:
            matching[clean_idx] = perturbed_key_to_idx[clean_key]
    
    return matching


def format_prefix_info_for_plotly(metadata: Dict, dataset_type: str) -> str:
    """
    Format prefix information for display in Plotly hover tooltip.
    
    Args:
        metadata: Metadata dictionary with 'case_id', 'prefix_len', 'cat_tensors', 'num_tensors'
        dataset_type: 'Clean' or 'Perturbed'
        
    Returns:
        Formatted HTML string with prefix information
    """
    case_id = metadata['case_id']
    prefix_len = metadata['prefix_len']
    cat_tensors = metadata['cat_tensors']
    num_tensors = metadata['num_tensors']
    
    # Get tensor shapes
    cat_shapes = [list(tensor.shape) for tensor in cat_tensors]
    num_shapes = [list(tensor.shape) for tensor in num_tensors]
    
    # Count features
    n_cat_features = sum(tensor.numel() for tensor in cat_tensors)
    n_num_features = sum(tensor.numel() for tensor in num_tensors)
    
    # Format as HTML for Plotly
    lines = [
        f"<b>{dataset_type} Dataset</b>",
        f"Case ID: {case_id}",
        f"Prefix Length: {prefix_len}",
        "",
        f"Categorical tensors: {len(cat_tensors)}",
        f"  Shapes: {cat_shapes}",
        f"  Total features: {n_cat_features}",
        "",
        f"Numerical tensors: {len(num_tensors)}",
        f"  Shapes: {num_shapes}",
        f"  Total features: {n_num_features}",
    ]
    
    return "<br>".join(lines)


def create_interactive_visualization(
    clean_data_path: str,
    perturbed_data_path: str,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create interactive visualization with hover functionality using Plotly.
    
    Args:
        clean_data_path: Path to clean dataset pickle file
        perturbed_data_path: Path to perturbed dataset pickle file
        alpha: Transparency level for scatter points (default: 0.6)
        figsize: Figure size tuple (default: (12, 8)) - width, height in pixels
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        )
    
    print(f"Loading clean dataset from: {clean_data_path}")
    clean_data, clean_metadata, clean_keys = load_data_with_prefixes(clean_data_path)
    print(f"Loaded {len(clean_data)} clean samples with {clean_data.shape[1]} features")
    
    print(f"Loading perturbed dataset from: {perturbed_data_path}")
    perturbed_data, perturbed_metadata, perturbed_keys = load_data_with_prefixes(perturbed_data_path)
    print(f"Loaded {len(perturbed_data)} perturbed samples with {perturbed_data.shape[1]} features")
    
    # Check feature dimensions match
    if clean_data.shape[1] != perturbed_data.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: clean has {clean_data.shape[1]} features, "
            f"perturbed has {perturbed_data.shape[1]} features"
        )
    
    # Match pairs
    print("Matching clean and perturbed pairs...")
    clean_to_perturbed = match_clean_perturbed_pairs(clean_keys, perturbed_keys)
    perturbed_to_clean = {v: k for k, v in clean_to_perturbed.items()}
    print(f"Matched {len(clean_to_perturbed)} pairs")
    
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
    
    # Prepare hover text for clean data
    clean_hover_texts = []
    for idx, metadata in enumerate(clean_metadata):
        hover_text = format_prefix_info_for_plotly(metadata, "Clean")
        # Add corresponding perturbed info if available
        if idx in clean_to_perturbed:
            pert_idx = clean_to_perturbed[idx]
            pert_metadata = perturbed_metadata[pert_idx]
            hover_text += "<br><br>" + "="*40 + "<br><br>"
            hover_text += format_prefix_info_for_plotly(pert_metadata, "Perturbed")
        clean_hover_texts.append(hover_text)
    
    # Prepare hover text for perturbed data
    perturbed_hover_texts = []
    for idx, metadata in enumerate(perturbed_metadata):
        hover_text = format_prefix_info_for_plotly(metadata, "Perturbed")
        # Add corresponding clean info if available
        if idx in perturbed_to_clean:
            clean_idx = perturbed_to_clean[idx]
            clean_meta = clean_metadata[clean_idx]
            hover_text += "<br><br>" + "="*40 + "<br><br>"
            hover_text += format_prefix_info_for_plotly(clean_meta, "Clean")
        perturbed_hover_texts.append(hover_text)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add clean data trace
    fig.add_trace(go.Scatter(
        x=clean_2d[:, 0],
        y=clean_2d[:, 1],
        mode='markers',
        name='Clean Data',
        marker=dict(
            color='blue',
            size=8,
            opacity=alpha,
            line=dict(width=0.5, color='darkblue')
        ),
        text=clean_hover_texts,
        hovertemplate='%{text}<extra></extra>',
        customdata=[(clean_keys[i], clean_to_perturbed.get(i, None)) for i in range(len(clean_2d))],
        showlegend=True
    ))
    
    # Add perturbed data trace
    fig.add_trace(go.Scatter(
        x=perturbed_2d[:, 0],
        y=perturbed_2d[:, 1],
        mode='markers',
        name='Perturbed Data',
        marker=dict(
            color='orange',
            size=8,
            opacity=alpha,
            line=dict(width=0.5, color='darkorange')
        ),
        text=perturbed_hover_texts,
        hovertemplate='%{text}<extra></extra>',
        customdata=[(perturbed_keys[i], perturbed_to_clean.get(i, None)) for i in range(len(perturbed_2d))],
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Clean vs Perturbed Dataset Visualization (2D PCA)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'black'}
        },
        xaxis_title=f'PC1 ({explained_variance[0]:.2%} variance)',
        yaxis_title=f'PC2 ({explained_variance[1]:.2%} variance)',
        width=figsize[0] * 100,  # Convert to pixels (assuming 100 dpi)
        height=figsize[1] * 100,
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Show the figure
    fig.show()
