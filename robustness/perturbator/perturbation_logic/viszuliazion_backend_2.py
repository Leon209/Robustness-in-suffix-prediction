"""
Interactive visualization backend for comparing clean and perturbed encoded datasets.

This module provides functions to load encoded prefix data with metadata, match pairs,
and create interactive visualizations with hover functionality using Plotly.
"""

import os
import json
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
    figsize: Tuple[int, int] = (12, 8),
    save_html: bool = True,
    html_filename: Optional[str] = None
) -> None:
    """
    Create interactive visualization with hover functionality using Plotly.
    
    Args:
        clean_data_path: Path to clean dataset pickle file
        perturbed_data_path: Path to perturbed dataset pickle file
        alpha: Transparency level for scatter points (default: 0.6)
        figsize: Figure size tuple (default: (12, 8)) - width, height in pixels
        save_html: Whether to save an HTML file (default: True)
        html_filename: Optional custom filename for HTML export (default: "interactive_visualization.html")
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        )
    
    # Configure renderer for remote/SSH access
    import plotly.io as pio
    renderers_to_try = [
        "plotly_mimetype+notebook_connected",  # Best for JupyterLab
        "notebook",  # Classic Jupyter
        "iframe",  # Fallback
        "browser"  # Opens in browser (requires X11 forwarding)
    ]
    
    renderer_set = False
    for renderer in renderers_to_try:
        try:
            pio.renderers.default = renderer
            print(f"✓ Set Plotly renderer to: {renderer}")
            renderer_set = True
            break
        except Exception as e:
            continue
    
    if not renderer_set:
        print("⚠ Could not set custom renderer, using default")
    
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
    
    # Prepare hover text and store corresponding indices for highlighting
    clean_hover_texts = []
    clean_corresponding_indices = []  # Store corresponding perturbed index for each clean point
    
    for idx, metadata in enumerate(clean_metadata):
        hover_text = format_prefix_info_for_plotly(metadata, "Clean")
        corr_idx = clean_to_perturbed.get(idx, None)
        clean_corresponding_indices.append(corr_idx)
        if corr_idx is not None:
            pert_metadata = perturbed_metadata[corr_idx]
            hover_text += "<br><br>" + "="*40 + "<br><br>"
            hover_text += format_prefix_info_for_plotly(pert_metadata, "Perturbed")
        clean_hover_texts.append(hover_text)
    
    # Prepare hover text for perturbed data
    perturbed_hover_texts = []
    perturbed_corresponding_indices = []
    
    for idx, metadata in enumerate(perturbed_metadata):
        hover_text = format_prefix_info_for_plotly(metadata, "Perturbed")
        corr_idx = perturbed_to_clean.get(idx, None)
        perturbed_corresponding_indices.append(corr_idx)
        if corr_idx is not None:
            clean_meta = clean_metadata[corr_idx]
            hover_text += "<br><br>" + "="*40 + "<br><br>"
            hover_text += format_prefix_info_for_plotly(clean_meta, "Clean")
        perturbed_hover_texts.append(hover_text)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Base and highlight sizes for visual feedback
    base_size = 8
    highlight_size = 20
    
    # Add clean data trace
    # Store corresponding indices in customdata for potential JavaScript highlighting
    fig.add_trace(go.Scatter(
        x=clean_2d[:, 0],
        y=clean_2d[:, 1],
        mode='markers',
        name='Clean Data',
        marker=dict(
            color='blue',
            size=base_size,
            opacity=alpha,
            line=dict(width=0.5, color='darkblue')
        ),
        text=clean_hover_texts,
        hovertemplate='%{text}<extra></extra>',
        customdata=list(zip(range(len(clean_2d)), clean_corresponding_indices, ['clean'] * len(clean_2d))),
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
            size=base_size,
            opacity=alpha,
            line=dict(width=0.5, color='darkorange')
        ),
        text=perturbed_hover_texts,
        hovertemplate='%{text}<extra></extra>',
        customdata=list(zip(range(len(perturbed_2d)), perturbed_corresponding_indices, ['perturbed'] * len(perturbed_2d))),
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
    
    # Store original colors for reset functionality
    original_colors_clean = ['blue'] * len(clean_2d)
    original_colors_perturbed = ['orange'] * len(perturbed_2d)
    
    # Convert matching dictionaries to JSON strings for JavaScript
    clean_to_perturbed_json = json.dumps(clean_to_perturbed)
    perturbed_to_clean_json = json.dumps(perturbed_to_clean)
    
    # Add JavaScript for hover highlighting
    # This will grey out all points except the hovered one and its matching pair
    hover_script = f"""
    <script>
    (function() {{
        function setupHoverHighlighting() {{
            // Find the Plotly graph div - try multiple selectors
            var gd = document.querySelector('.plotly-graph-div') || 
                     document.querySelector('[id*="plotly"]') ||
                     document.querySelector('[class*="js-plotly"]') ||
                     document.querySelector('div[data-id]');
            
            // If still not found, try to find any div containing Plotly
            if (!gd) {{
                var allDivs = document.querySelectorAll('div');
                for (var i = 0; i < allDivs.length; i++) {{
                    if (allDivs[i]._fullLayout || allDivs[i]._fullData) {{
                        gd = allDivs[i];
                        break;
                    }}
                }}
            }}
            
            if (!gd) {{
                // Retry after a short delay
                setTimeout(setupHoverHighlighting, 100);
                return;
            }}
            
            // Store original colors
            var originalColorsClean = {json.dumps(original_colors_clean)};
            var originalColorsPerturbed = {json.dumps(original_colors_perturbed)};
            var cleanToPerturbed = {clean_to_perturbed_json};
            var perturbedToClean = {perturbed_to_clean_json};
            
            // Function to reset all colors
            function resetColors() {{
                var update = {{
                    'marker.color': [originalColorsClean, originalColorsPerturbed]
                }};
                Plotly.restyle(gd, update, [0, 1]);
            }}
            
            // Function to highlight matching pair
            function highlightPair(traceIndex, pointIndex) {{
                var colors = [];
                var matchingIndex = null;
                
                if (traceIndex === 0) {{
                    // Hovering on clean data
                    colors[0] = originalColorsClean.map(function(c, i) {{
                        return (i === pointIndex) ? c : 'lightgray';
                    }});
                    
                    matchingIndex = cleanToPerturbed[pointIndex];
                    if (matchingIndex !== null && matchingIndex !== undefined) {{
                        colors[1] = originalColorsPerturbed.map(function(c, i) {{
                            return (i === matchingIndex) ? c : 'lightgray';
                        }});
                    }} else {{
                        colors[1] = originalColorsPerturbed.map(function() {{ return 'lightgray'; }});
                    }}
                }} else {{
                    // Hovering on perturbed data
                    colors[1] = originalColorsPerturbed.map(function(c, i) {{
                        return (i === pointIndex) ? c : 'lightgray';
                    }});
                    
                    matchingIndex = perturbedToClean[pointIndex];
                    if (matchingIndex !== null && matchingIndex !== undefined) {{
                        colors[0] = originalColorsClean.map(function(c, i) {{
                            return (i === matchingIndex) ? c : 'lightgray';
                        }});
                    }} else {{
                        colors[0] = originalColorsClean.map(function() {{ return 'lightgray'; }});
                    }}
                }}
                
                var update = {{
                    'marker.color': colors
                }};
                Plotly.restyle(gd, update, [0, 1]);
            }}
            
            // Attach hover event listener
            gd.on('plotly_hover', function(data) {{
                if (data.points.length > 0) {{
                    var point = data.points[0];
                    highlightPair(point.curveNumber, point.pointNumber);
                }}
            }});
            
            // Reset colors when mouse leaves
            gd.on('plotly_unhover', function(data) {{
                resetColors();
            }});
        }}
        
        // Wait for Plotly to be ready
        if (document.readyState === 'loading') {{
            window.addEventListener('load', setupHoverHighlighting);
        }} else {{
            // DOM is already loaded, but Plotly might not be ready
            setTimeout(setupHoverHighlighting, 100);
        }}
    }})();
    </script>
    """
    
    # Show the figure
    fig.show()
    
    # Also save as HTML file for guaranteed interactivity (works even over SSH)
    if save_html:
        if html_filename is None:
            html_filename = "interactive_visualization.html"
        try:
            # Read the HTML content and inject our JavaScript
            html_string = fig.to_html(include_plotlyjs='cdn')
            
            # Inject JavaScript before closing body tag
            if '</body>' in html_string:
                html_string = html_string.replace('</body>', hover_script + '</body>')
            else:
                html_string = html_string + hover_script
            
            # Write the modified HTML
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_string)
            
            print(f"\n✓ Saved interactive plot to: {html_filename}")
            print("  You can download and open this file in any browser for full hover functionality!")
            print("  Hover over a point to see it and its matching pair highlighted (others turn grey)!")
        except Exception as e:
            print(f"\n⚠ Could not save HTML file: {e}")
            print("  The plot should still be visible in the notebook.")
    
    # Print instructions for troubleshooting
    print("\n" + "="*60)
    print("Hover Instructions:")
    print("="*60)
    print("Hover over any point to see prefix information for both datasets.")
    print("\nIf hover doesn't work in the notebook:")
    print("1. Download the HTML file above and open it in your browser")
    print("2. Ensure you're accessing Jupyter via a browser (not terminal)")
    print("3. Install Plotly extensions: pip install jupyterlab-plotly")
    print("4. Trust the notebook: File -> Trust Notebook")
    print("5. Restart JupyterLab after installing extensions")
    print("6. For SSH access, use port forwarding:")
    print("   ssh -L 8888:localhost:8888 user@server")
    print("   Then access via http://localhost:8888")
    print("="*60)