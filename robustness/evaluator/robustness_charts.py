"""
Robustness Charts Generation Module

This module provides functions to generate comparison charts for robustness evaluation
results. Supports 1-3 models dynamically.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def setup_plot_style():
    """Setup consistent plot styling."""
    mpl.rcdefaults()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.fontsize': 8,
        'lines.linewidth': 1.2,
        'lines.markersize': 5
    })


def _get_common_prefix_lengths(models: List[Dict[str, Any]]) -> List[int]:
    """Get common prefix lengths across all models."""
    if not models:
        return []
    
    common_lengths = set(models[0]['data']['prefix_lengths'])
    for model in models[1:]:
        common_lengths &= set(model['data']['prefix_lengths'])
    
    return sorted(common_lengths)


def plot_comparison_chart(
    models: List[Dict[str, Any]],
    key: str,
    ylabel: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = (0, 1.05),
    include_iqr: bool = False,
    q25_key: Optional[str] = None,
    q75_key: Optional[str] = None
):
    """
    Generic function to plot comparison charts for 1-3 models.
    
    Args:
        models: List of model dictionaries, each containing:
            - 'data': aggregate metrics dict
            - 'name': model display name
            - 'color': plot color
            - 'marker': plot marker
        key: Key to extract from data (e.g., 'activity_match_rates')
        ylabel: Y-axis label
        title: Chart title
        ylim: Y-axis limits tuple, or None for auto-scale
        include_iqr: Whether to include IQR shading
        q25_key: Key for Q25 values (if include_iqr=True)
        q75_key: Key for Q75 values (if include_iqr=True)
    
    Returns:
        matplotlib figure or None if no data
    """
    setup_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    
    # Get common prefix lengths
    common_prefix_lengths = _get_common_prefix_lengths(models)
    
    if not common_prefix_lengths:
        plt.close(fig)
        return None
    
    # Extract values for each model
    model_values = []
    model_counts = []
    
    for model in models:
        data = model['data']
        values_dict = dict(zip(data['prefix_lengths'], data[key]))
        counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))
        
        values = [values_dict.get(p, 0) for p in common_prefix_lengths]
        counts = [counts_dict.get(p, 0) for p in common_prefix_lengths]
        
        model_values.append(values)
        model_counts.append(counts)
    
    # Plot IQR if requested
    if include_iqr and q25_key and q75_key:
        for model in models:
            data = model['data']
            q25_dict = dict(zip(data['prefix_lengths'], data[q25_key]))
            q75_dict = dict(zip(data['prefix_lengths'], data[q75_key]))
            
            q25_values = [q25_dict.get(p, 0) for p in common_prefix_lengths]
            q75_values = [q75_dict.get(p, 0) for p in common_prefix_lengths]
            
            ax1.fill_between(common_prefix_lengths, q25_values, q75_values,
                            color=model['color'], alpha=0.15, 
                            label=f"{model['name']} IQR")
    
    # Plot lines for each model
    for model, values in zip(models, model_values):
        ax1.plot(common_prefix_lengths, values, marker=model['marker'],
                 linewidth=1.2, markersize=5, label=model['name'],
                 color=model['color'], alpha=0.8)
    
    # Secondary y-axis for instance counts
    ax2 = ax1.twinx()
    total_counts = [sum(counts[i] for counts in model_counts) 
                   for i in range(len(common_prefix_lengths))]
    ax2.plot(common_prefix_lengths, total_counts,
             linestyle='--', color='gray', label='# instances')
    ax2.fill_between(common_prefix_lengths, total_counts, color='gray', alpha=0.3)
    
    # Style axes
    ax1.set_xlabel('prefix len', labelpad=0.5)
    ax1.set_ylabel(ylabel, labelpad=0.5)
    ax2.set_ylabel('instances', labelpad=0.5)
    
    if ylim:
        ax1.set_ylim(ylim)
    ax1.set_xlim(left=min(common_prefix_lengths) - 0.5, 
                 right=max(common_prefix_lengths) + 0.5)
    ax2.set_ylim(bottom=0)
    
    # Remove spines
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Add grid
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    if ylim and ylim[1] >= 1.0:
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               frameon=True, fontsize=8)
    
    ax2.set_yticks([])
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    
    plt.title(title, fontsize=10)
    plt.tight_layout()
    
    return fig


def plot_clean_pert_comparison(
    models: List[Dict[str, Any]],
    clean_key: str,
    pert_key: str,
    ylabel: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = (0, 1.05)
):
    """
    Plot comparison with clean and perturbed lines for 1-3 models.
    
    Args:
        models: List of model dictionaries
        clean_key: Key for clean data (e.g., 'support_clean')
        pert_key: Key for perturbed data (e.g., 'support_perturbed')
        ylabel: Y-axis label
        title: Chart title
        ylim: Y-axis limits tuple, or None for auto-scale
    
    Returns:
        matplotlib figure or None if no data
    """
    setup_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    
    common_prefix_lengths = _get_common_prefix_lengths(models)
    
    if not common_prefix_lengths:
        plt.close(fig)
        return None
    
    # Extract values for each model
    model_clean_values = []
    model_pert_values = []
    model_counts = []
    
    for model in models:
        data = model['data']
        clean_dict = dict(zip(data['prefix_lengths'], data[clean_key]))
        pert_dict = dict(zip(data['prefix_lengths'], data[pert_key]))
        counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))
        
        clean_values = [clean_dict.get(p, 0) for p in common_prefix_lengths]
        pert_values = [pert_dict.get(p, 0) for p in common_prefix_lengths]
        counts = [counts_dict.get(p, 0) for p in common_prefix_lengths]
        
        model_clean_values.append(clean_values)
        model_pert_values.append(pert_values)
        model_counts.append(counts)
    
    # Plot clean and perturbed lines for each model
    for model, clean_values, pert_values in zip(models, model_clean_values, model_pert_values):
        ax1.plot(common_prefix_lengths, clean_values, marker=model['marker'],
                 linewidth=1.2, markersize=5, label=f"{model['name']} (clean)",
                 color=model['color'], alpha=0.9)
        ax1.plot(common_prefix_lengths, pert_values, marker=model['marker'], 
                 linestyle='--',
                 linewidth=1.2, markersize=5, label=f"{model['name']} (perturbed)",
                 color=model['color'], alpha=0.6)
    
    # Secondary y-axis
    ax2 = ax1.twinx()
    total_counts = [sum(counts[i] for counts in model_counts) 
                   for i in range(len(common_prefix_lengths))]
    ax2.plot(common_prefix_lengths, total_counts,
             linestyle='--', color='gray', label='# instances')
    ax2.fill_between(common_prefix_lengths, total_counts, color='gray', alpha=0.3)
    
    # Style
    ax1.set_xlabel('prefix len', labelpad=0.5)
    ax1.set_ylabel(ylabel, labelpad=0.5)
    ax2.set_ylabel('instances', labelpad=0.5)
    if ylim:
        ax1.set_ylim(ylim)
    ax1.set_xlim(left=min(common_prefix_lengths) - 0.5, 
                 right=max(common_prefix_lengths) + 0.5)
    ax2.set_ylim(bottom=0)
    
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    if ylim and ylim[1] >= 1.0:
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               frameon=True, fontsize=8)
    
    ax2.set_yticks([])
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    
    plt.title(title, fontsize=10)
    plt.tight_layout()
    
    return fig


def plot_single_model_chart(
    model: Dict[str, Any],
    key: str,
    ylabel: str,
    title: str
):
    """
    Plot chart for a single model.
    
    Args:
        model: Model dictionary with 'data', 'name', 'color', 'marker'
        key: Key to extract from data
        ylabel: Y-axis label
        title: Chart title
    
    Returns:
        matplotlib figure or None if no data
    """
    setup_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
    
    data = model['data']
    prefix_lengths = sorted(data['prefix_lengths'])
    
    if not prefix_lengths:
        plt.close(fig)
        return None
    
    values_dict = dict(zip(data['prefix_lengths'], data[key]))
    counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))
    
    values = [values_dict.get(p, 0) for p in prefix_lengths]
    counts = [counts_dict.get(p, 0) for p in prefix_lengths]
    
    ax1.plot(prefix_lengths, values, marker=model['marker'],
             linewidth=1.2, markersize=5, label=model['name'],
             color=model['color'], alpha=0.9)
    
    ax2 = ax1.twinx()
    ax2.plot(prefix_lengths, counts,
             linestyle='--', color='gray', label='# instances')
    ax2.fill_between(prefix_lengths, counts, color='gray', alpha=0.3)
    
    ax1.set_xlabel('prefix len', labelpad=0.5)
    ax1.set_ylabel(ylabel, labelpad=0.5)
    ax2.set_ylabel('instances', labelpad=0.5)
    ax1.set_xlim(left=min(prefix_lengths) - 0.5, right=max(prefix_lengths) + 0.5)
    ax2.set_ylim(bottom=0)
    
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               frameon=True, fontsize=8)
    
    ax2.set_yticks([])
    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)
    
    plt.title(title, fontsize=10)
    plt.tight_layout()
    
    return fig


def plot_subplot_most_likely(
    models: List[Dict[str, Any]],
    title: str
):
    """
    Plot a 3-panel subplot with attack success rate, DLS (clean vs perturbed),
    and remaining time MAE in days (clean vs perturbed) side by side.

    Args:
        models: List of model dictionaries with 'data', 'name', 'color', 'marker'
        title: Overall figure title (suptitle)

    Returns:
        matplotlib figure or None if no data
    """
    setup_plot_style()

    common_prefix_lengths = _get_common_prefix_lengths(models)
    if not common_prefix_lengths:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

    panel_specs = [
        {
            'key': 'attack_success_rates',
            'ylabel': 'Attack Success Rate',
            'subtitle': 'Attack Success Rate',
            'ylim': (0, 1.05),
            'mode': 'single',
        },
        {
            'clean_key': 'clean_dls',
            'pert_key': 'perturbed_dls',
            'ylabel': 'DLS',
            'subtitle': 'DLS',
            'ylim': (0, 1.05),
            'mode': 'clean_pert',
        },
        {
            'clean_key': 'remaining_time_mae_clean',
            'pert_key': 'remaining_time_mae_perturbed',
            'ylabel': 'Remaining Time MAE (days)',
            'subtitle': 'Remaining Time MAE (days)',
            'ylim': None,
            'mode': 'clean_pert',
        },
    ]

    for ax1, spec in zip(axes, panel_specs):
        panel_models = spec.get('models_override', models)
        panel_prefix_lengths = _get_common_prefix_lengths(panel_models) or common_prefix_lengths
        model_counts = []

        if spec['mode'] == 'single':
            model_values = []
            for model in panel_models:
                data = model['data']
                values_dict = dict(zip(data['prefix_lengths'], data[spec['key']]))
                counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))
                model_values.append([values_dict.get(p, 0) for p in panel_prefix_lengths])
                model_counts.append([counts_dict.get(p, 0) for p in panel_prefix_lengths])

            for model, values in zip(panel_models, model_values):
                ax1.plot(panel_prefix_lengths, values, marker=model['marker'],
                         linewidth=1.2, markersize=5, label=model['name'],
                         color=model['color'], alpha=0.8)

        else:  # clean_pert
            for model in panel_models:
                data = model['data']
                clean_dict = dict(zip(data['prefix_lengths'], data[spec['clean_key']]))
                pert_dict = dict(zip(data['prefix_lengths'], data[spec['pert_key']]))
                counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))

                clean_values = [clean_dict.get(p, 0) for p in panel_prefix_lengths]
                pert_values = [pert_dict.get(p, 0) for p in panel_prefix_lengths]
                model_counts.append([counts_dict.get(p, 0) for p in panel_prefix_lengths])

                ax1.plot(panel_prefix_lengths, clean_values, marker=model['marker'],
                         linewidth=1.2, markersize=5, label=f"{model['name']} (clean)",
                         color=model['color'], alpha=0.9)
                ax1.plot(panel_prefix_lengths, pert_values, marker=model['marker'],
                         linestyle='--', linewidth=1.2, markersize=5,
                         label=f"{model['name']} (perturbed)",
                         color=model['color'], alpha=0.6)

        # Secondary axis: instance count as dashed line + fill
        ax2 = ax1.twinx()
        total_counts = [sum(counts[i] for counts in model_counts)
                        for i in range(len(panel_prefix_lengths))]
        ax2.plot(panel_prefix_lengths, total_counts,
                 linestyle='--', color='gray', label='# instances')
        ax2.fill_between(panel_prefix_lengths, total_counts, color='gray', alpha=0.3)

        ax1.set_xlabel('prefix len', labelpad=0.5)
        ax1.set_ylabel(spec['ylabel'], labelpad=0.5)
        ax2.set_ylabel('instances', labelpad=0.5)

        if spec['ylim']:
            ax1.set_ylim(spec['ylim'])
        ax1.set_xlim(left=min(panel_prefix_lengths) - 0.5,
                     right=max(panel_prefix_lengths) + 0.5)
        ax2.set_ylim(bottom=0)

        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        if spec['ylim'] and spec['ylim'][1] >= 1.0:
            ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   frameon=True, fontsize=7)

        ax2.set_yticks([])
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_visible(False)

        ax1.set_title(spec['subtitle'], fontsize=10)

    plt.tight_layout()
    fig.text(0.01, 1.01, title, fontsize=15, fontweight='bold',
             ha='left', va='bottom', transform=fig.transFigure)

    return fig


def plot_subplot_probabilistic_prediction(
    models: List[Dict[str, Any]],
    title: str
):
    """
    Plot a 2-panel subplot with Support of Correct Prediction (clean vs perturbed)
    and Wasserstein Distance side by side.

    Args:
        models: List of model dictionaries with 'data', 'name', 'color', 'marker'
        title: Overall figure title shown bold and left-aligned above the panels

    Returns:
        matplotlib figure or None if no data
    """
    setup_plot_style()

    common_prefix_lengths = _get_common_prefix_lengths(models)
    if not common_prefix_lengths:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    panel_specs = [
        {
            'clean_key': 'support_clean',
            'pert_key': 'support_perturbed',
            'ylabel': 'Support of Correct Prediction',
            'subtitle': 'Support of Correct Prediction',
            'ylim': (0, 1.05),
            'mode': 'clean_pert',
        },
        {
            'key': 'wasserstein_distance',
            'ylabel': 'Wasserstein Distance',
            'subtitle': 'Wasserstein Distance',
            'ylim': None,
            'mode': 'single',
        },
    ]

    for ax1, spec in zip(axes, panel_specs):
        model_counts = []

        if spec['mode'] == 'single':
            model_values = []
            for model in models:
                data = model['data']
                values_dict = dict(zip(data['prefix_lengths'], data[spec['key']]))
                counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))
                model_values.append([values_dict.get(p, 0) for p in common_prefix_lengths])
                model_counts.append([counts_dict.get(p, 0) for p in common_prefix_lengths])

            for model, values in zip(models, model_values):
                ax1.plot(common_prefix_lengths, values, marker=model['marker'],
                         linewidth=1.2, markersize=5, label=model['name'],
                         color=model['color'], alpha=0.8)

        else:  # clean_pert
            for model in models:
                data = model['data']
                clean_dict = dict(zip(data['prefix_lengths'], data[spec['clean_key']]))
                pert_dict = dict(zip(data['prefix_lengths'], data[spec['pert_key']]))
                counts_dict = dict(zip(data['prefix_lengths'], data['sample_counts']))

                clean_values = [clean_dict.get(p, 0) for p in common_prefix_lengths]
                pert_values = [pert_dict.get(p, 0) for p in common_prefix_lengths]
                model_counts.append([counts_dict.get(p, 0) for p in common_prefix_lengths])

                ax1.plot(common_prefix_lengths, clean_values, marker=model['marker'],
                         linewidth=1.2, markersize=5, label=f"{model['name']} (clean)",
                         color=model['color'], alpha=0.9)
                ax1.plot(common_prefix_lengths, pert_values, marker=model['marker'],
                         linestyle='--', linewidth=1.2, markersize=5,
                         label=f"{model['name']} (perturbed)",
                         color=model['color'], alpha=0.6)

        # Secondary axis: instance count as dashed line + fill
        ax2 = ax1.twinx()
        total_counts = [sum(counts[i] for counts in model_counts)
                        for i in range(len(common_prefix_lengths))]
        ax2.plot(common_prefix_lengths, total_counts,
                 linestyle='--', color='gray', label='# instances')
        ax2.fill_between(common_prefix_lengths, total_counts, color='gray', alpha=0.3)

        ax1.set_xlabel('prefix len', labelpad=0.5)
        ax1.set_ylabel(spec['ylabel'], labelpad=0.5)
        ax2.set_ylabel('instances', labelpad=0.5)

        if spec['ylim']:
            ax1.set_ylim(spec['ylim'])
        ax1.set_xlim(left=min(common_prefix_lengths) - 0.5,
                     right=max(common_prefix_lengths) + 0.5)
        ax2.set_ylim(bottom=0)

        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        if spec['ylim'] and spec['ylim'][1] >= 1.0:
            ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   frameon=True, fontsize=7)

        ax2.set_yticks([])
        ax1.set_zorder(2)
        ax2.set_zorder(1)
        ax1.patch.set_visible(False)

        ax1.set_title(spec['subtitle'], fontsize=10)

    plt.tight_layout()
    fig.text(0.01, 1.01, title, fontsize=15, fontweight='bold',
             ha='left', va='bottom', transform=fig.transFigure)

    return fig


def generate_all_charts_for_comparison(
    dataset: str,
    attack: str,
    models: List[Dict[str, Any]],
    output_base_dir: str,
    display_dataset: Optional[str] = None
) -> List[str]:
    """
    Generate all charts for a dataset-attack combination.
    
    Args:
        dataset: Dataset folder name (used for output path)
        attack: Attack name
        models: List of model dictionaries with 'data', 'name', 'color', 'marker', 'results'
        output_base_dir: Base output directory
        display_dataset: Optional display name for the dataset shown in chart titles.
                         If None, falls back to dataset.
    
    Returns:
        List of generated chart file paths
    """
    # Create output directory for this combination
    output_subdir = f"{output_base_dir}/{dataset}/{attack}"
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    
    charts_generated = []
    
    # Determine number of models
    num_models = len(models)
    
    # 1. Attack Success Rate
    try:
        fig = plot_comparison_chart(
            models, 'attack_success_rates',
            'Attack Success Rate',
            f'{dataset} - {attack}: Attack Success Rate'
        )
        if fig:
            save_path = f"{output_subdir}/attack_success_rate.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating attack_success_rate: {e}")
    
    # 2. Length Match Rate
    try:
        fig = plot_comparison_chart(
            models, 'length_match_rates',
            'Length Match Rate',
            f'{dataset} - {attack}: Length Match Rate'
        )
        if fig:
            save_path = f"{output_subdir}/length_match_rate.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating length_match_rate: {e}")
    
    # 3. Remaining Time MAE (days) - Clean vs Perturbed
    try:
        fig = plot_clean_pert_comparison(
            models, 'remaining_time_mae_clean', 'remaining_time_mae_perturbed',
            'Remaining Time MAE (days)',
            f'{dataset} - {attack}: Remaining Time MAE',
            ylim=None
        )
        if fig:
            save_path = f"{output_subdir}/remaining_time_mae.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating remaining_time_mae: {e}")
    
    # 4. Clean DLS
    try:
        fig = plot_comparison_chart(
            models, 'clean_dls',
            'Clean DLS',
            f'{dataset} - {attack}: Clean DLS'
        )
        if fig:
            save_path = f"{output_subdir}/clean_dls.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating clean_dls: {e}")
    
    # 5. DLS (Clean vs Perturbed)
    try:
        fig = plot_clean_pert_comparison(
            models, 'clean_dls', 'perturbed_dls',
            'DLS',
            f'{dataset} - {attack}: DLS'
        )
        if fig:
            save_path = f"{output_subdir}/dls_drop.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating dls_drop: {e}")
    
    # 6. Subplot Most Likely (Attack Success Rate | DLS | Remaining Time MAE)
    try:
        fig = plot_subplot_most_likely(
            models,
            display_dataset if display_dataset is not None else dataset
        )
        if fig:
            save_path = f"{output_subdir}/subplot_most_likely.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating subplot_most_likely: {e}")

    # 7. Modal Clean DLS with IQR
    try:
        fig = plot_comparison_chart(
            models, 'modal_clean_dls',
            'Modal DLS on Clean Data',
            f'{dataset} - {attack}: Modal Clean DLS',
            include_iqr=True,
            q25_key='clean_dls_q25',
            q75_key='clean_dls_q75'
        )
        if fig:
            save_path = f"{output_subdir}/modal_clean_dls.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating modal_clean_dls: {e}")
    
    # 8. Modal Perturbed DLS with IQR
    try:
        fig = plot_comparison_chart(
            models, 'modal_perturbed_dls',
            'Modal DLS on Perturbed Data',
            f'{dataset} - {attack}: Modal Perturbed DLS',
            include_iqr=True,
            q25_key='perturbed_dls_q25',
            q75_key='perturbed_dls_q75'
        )
        if fig:
            save_path = f"{output_subdir}/modal_perturbed_dls.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating modal_perturbed_dls: {e}")
    
    # 8. Support of Correct Prediction
    try:
        fig = plot_clean_pert_comparison(
            models, 'support_clean', 'support_perturbed',
            'Support of Correct Prediction',
            f'{dataset} - {attack}: Support of Correct Prediction'
        )
        if fig:
            save_path = f"{output_subdir}/support_clean_pert.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating support_clean_pert: {e}")
    
    # 9. ROUGE-L Score
    try:
        fig = plot_clean_pert_comparison(
            models, 'rouge_l_clean', 'rouge_l_perturbed',
            'ROUGE-L Score',
            f'{dataset} - {attack}: ROUGE-L Score'
        )
        if fig:
            save_path = f"{output_subdir}/rouge_l.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating rouge_l: {e}")
    
    # 10. chrF Score
    try:
        fig = plot_clean_pert_comparison(
            models, 'chrf_clean', 'chrf_perturbed',
            'chrF Score',
            f'{dataset} - {attack}: chrF Score'
        )
        if fig:
            save_path = f"{output_subdir}/chrf.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating chrf: {e}")
    
    # 11. Negative Log Likelihood
    try:
        fig = plot_clean_pert_comparison(
            models, 'nll_clean', 'nll_perturbed',
            'Negative Log Likelihood',
            f'{dataset} - {attack}: Negative Log Likelihood',
        )
        if fig:
            save_path = f"{output_subdir}/nll.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating nll: {e}")
    
    # 12. Wasserstein Distance (all models in one chart)
    try:
        fig = plot_comparison_chart(
            models, 'wasserstein_distance',
            'Wasserstein Distance',
            f'{dataset} - {attack}: Wasserstein Distance',
            ylim=None
        )
        if fig:
            save_path = f"{output_subdir}/wasserstein_distance.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating wasserstein_distance: {e}")

    # 13. Subplot Probabilistic Prediction (Support of Correct Prediction | Wasserstein Distance)
    try:
        fig = plot_subplot_probabilistic_prediction(
            models,
            display_dataset if display_dataset is not None else dataset
        )
        if fig:
            save_path = f"{output_subdir}/subplot_probabilistic_prediction.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating subplot_probabilistic_prediction: {e}")

    return charts_generated


def generate_summary_table(
    dataset: str,
    attack: str,
    models: List[Dict[str, Any]],
    output_base_dir: str,
    display_dataset: Optional[str] = None
) -> str:
    """
    Generate and save summary statistics text file with average values per metric per model.

    Args:
        dataset: Dataset folder name (used for output path)
        attack: Attack name
        models: List of model dictionaries with 'data', 'results', and 'name'
        output_base_dir: Base output directory
        display_dataset: Optional display name for the dataset shown in the header

    Returns:
        Summary text string
    """
    output_subdir = f"{output_base_dir}/{dataset}/{attack}"
    Path(output_subdir).mkdir(parents=True, exist_ok=True)

    title = display_dataset if display_dataset is not None else dataset
    lines = []
    lines.append("=" * 60)
    lines.append(f"Summary Statistics: {title} — {attack}")
    lines.append("=" * 60)

    # Metrics derived from the per-prefix aggregate data (mean over all prefix lengths)
    aggregate_metrics = [
        ("Attack Success Rate",          "attack_success_rates"),
        ("Length Match Rate",            "length_match_rates"),
        ("Clean DLS",                    "clean_dls"),
        ("Perturbed DLS",                "perturbed_dls"),
        ("chrF Score (clean)",           "chrf_clean"),
        ("chrF Score (perturbed)",       "chrf_perturbed"),
        ("ROUGE-L (clean)",              "rouge_l_clean"),
        ("ROUGE-L (perturbed)",          "rouge_l_perturbed"),
        ("NLL (clean)",                  "nll_clean"),
        ("NLL (perturbed)",              "nll_perturbed"),
        ("Wasserstein Distance",         "wasserstein_distance"),
        ("Remaining Time MAE clean (d)", "remaining_time_mae_clean"),
        ("Remaining Time MAE pert (d)",  "remaining_time_mae_perturbed"),
        ("Support Correct (clean)",      "support_clean"),
        ("Support Correct (perturbed)",  "support_perturbed"),
    ]

    for label, key in aggregate_metrics:
        lines.append(f"\n{label}")
        for model in models:
            data = model['data']
            values = data.get(key, [])
            if values:
                avg = float(np.mean([v for v in values if v is not None]))
                lines.append(f"  {model['name']}: {avg:.4f}")
            else:
                lines.append(f"  {model['name']}: n/a")

    lines.append("\n" + "=" * 60)
    lines.append("Total Evaluations")
    lines.append("=" * 60)
    for model in models:
        lines.append(f"  {model['name']}: {len(model['results'])}")

    summary_text = "\n".join(lines)

    with open(f"{output_subdir}/summary_statistics.txt", 'w') as f:
        f.write(summary_text)

    return summary_text
