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
    ax2.bar(common_prefix_lengths, total_counts, alpha=0.15, color='gray',
            width=0.6, label='Total instances')
    
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
    title: str
):
    """
    Plot comparison with clean and perturbed lines for 1-3 models.
    
    Args:
        models: List of model dictionaries
        clean_key: Key for clean data (e.g., 'support_clean')
        pert_key: Key for perturbed data (e.g., 'support_perturbed')
        ylabel: Y-axis label
        title: Chart title
    
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
    ax2.bar(common_prefix_lengths, total_counts, alpha=0.15, color='gray',
            width=0.6, label='Total instances')
    
    # Style
    ax1.set_xlabel('prefix len', labelpad=0.5)
    ax1.set_ylabel(ylabel, labelpad=0.5)
    ax2.set_ylabel('instances', labelpad=0.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(left=min(common_prefix_lengths) - 0.5, 
                 right=max(common_prefix_lengths) + 0.5)
    ax2.set_ylim(bottom=0)
    
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
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
    ax2.bar(prefix_lengths, counts, alpha=0.15, color='gray',
            width=0.6, label='Instances')
    
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


def generate_all_charts_for_comparison(
    dataset: str,
    attack: str,
    models: List[Dict[str, Any]],
    output_base_dir: str
) -> List[str]:
    """
    Generate all charts for a dataset-attack combination.
    
    Args:
        dataset: Dataset name
        attack: Attack name
        models: List of model dictionaries with 'data', 'name', 'color', 'marker', 'results'
        output_base_dir: Base output directory
    
    Returns:
        List of generated chart file paths
    """
    # Create output directory for this combination
    output_subdir = f"{output_base_dir}/{dataset}/{attack}"
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    
    charts_generated = []
    
    # Determine number of models
    num_models = len(models)
    
    # 1. Activity Match Rate
    try:
        fig = plot_comparison_chart(
            models, 'activity_match_rates',
            'Activity Sequence Match Rate', 
            f'{dataset} - {attack}: Activity Match Rate'
        )
        if fig:
            save_path = f"{output_subdir}/activity_match_rate.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating activity_match_rate: {e}")
    
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
    
    # 3. Remaining Time Prediction Shift
    try:
        fig = plot_comparison_chart(
            models, 'remaining_time_prediction_shift',
            'Remaining Time Prediction Shift',
            f'{dataset} - {attack}: Remaining Time Shift',
            ylim=None
        )
        if fig:
            save_path = f"{output_subdir}/remaining_time_shift.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating remaining_time_shift: {e}")
    
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
    
    # 5. DLS Drop
    try:
        fig = plot_comparison_chart(
            models, 'relative_dls_drop',
            'DLS drop under Perturbation',
            f'{dataset} - {attack}: DLS Drop'
        )
        if fig:
            save_path = f"{output_subdir}/dls_drop.png"
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append(save_path)
    except Exception as e:
        print(f"    Error generating dls_drop: {e}")
    
    # 6. Modal Clean DLS with IQR
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
    
    # 7. Modal Perturbed DLS with IQR
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
    
    # 12-13. Wasserstein Distance (one chart per model)
    for model in models:
        try:
            fig = plot_single_model_chart(
                model, 'wasserstein_distance',
                'Wasserstein Distance',
                f'{dataset} - {attack}: Wasserstein Distance ({model["name"]})'
            )
            if fig:
                save_path = f"{output_subdir}/wasserstein_{model['model_id']}.png"
                fig.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                charts_generated.append(save_path)
        except Exception as e:
            print(f"    Error generating wasserstein_{model['model_id']}: {e}")
    
    return charts_generated


def generate_summary_table(
    dataset: str,
    attack: str,
    models: List[Dict[str, Any]],
    output_base_dir: str
) -> str:
    """
    Generate and save summary comparison table.
    
    Args:
        dataset: Dataset name
        attack: Attack name
        models: List of model dictionaries with 'results' and 'name'
        output_base_dir: Base output directory
    
    Returns:
        Summary text string
    """
    output_subdir = f"{output_base_dir}/{dataset}/{attack}"
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append(f"SUMMARY COMPARISON: {dataset} - {attack}")
    summary_lines.append("="*80)
    
    # Create header with model names
    header = f"{'Metric':<30}"
    for model in models:
        header += f" {model['name']:<20}"
    if len(models) > 1:
        header += f" {'Difference':<15}"
    summary_lines.append(f"\n{header}")
    summary_lines.append("-"*80)
    
    # Calculate metrics for each model
    all_mean_metrics = []
    all_prob_metrics = []
    
    for model in models:
        results = model['results']
        mean_metrics = [entry['robustness_metrics']['mean_prediction'] 
                       for entry in results.values() 
                       if 'robustness_metrics' in entry]
        prob_metrics = [entry['robustness_metrics'].get('probabilistic_prediction', {}) 
                       for entry in results.values() 
                       if 'robustness_metrics' in entry]
        prob_metrics = [m for m in prob_metrics if m]
        
        all_mean_metrics.append(mean_metrics)
        all_prob_metrics.append(prob_metrics)
    
    # Calculate aggregate metrics
    metrics_data = []
    for i, mean_metrics in enumerate(all_mean_metrics):
        activity_match = np.mean([m['activity_sequence_match'] for m in mean_metrics])
        length_match = np.mean([m['length_match'] for m in mean_metrics])
        metrics_data.append({
            'activity_match': activity_match,
            'length_match': length_match
        })
    
    # Add top-k if available
    topk_available = all([len(pm) > 0 for pm in all_prob_metrics])
    if topk_available:
        for i, prob_metrics in enumerate(all_prob_metrics):
            topk = np.mean([m.get('top_k_activity_match_rate', 0.0) for m in prob_metrics])
            metrics_data[i]['topk'] = topk
    
    # Activity Match Rate
    metric_row = f"{'Activity Match Rate':<30}"
    for i, model_data in enumerate(metrics_data):
        metric_row += f" {model_data['activity_match']:<20.4f}"
    if len(models) > 1:
        diff = metrics_data[0]['activity_match'] - metrics_data[-1]['activity_match']
        diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "≈ 0.0000"
        metric_row += f" {diff_str:<15}"
    summary_lines.append(metric_row)
    
    # Length Match Rate
    metric_row = f"{'Length Match Rate':<30}"
    for i, model_data in enumerate(metrics_data):
        metric_row += f" {model_data['length_match']:<20.4f}"
    if len(models) > 1:
        diff = metrics_data[0]['length_match'] - metrics_data[-1]['length_match']
        diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "≈ 0.0000"
        metric_row += f" {diff_str:<15}"
    summary_lines.append(metric_row)
    
    # Top k Activity Match Rate
    if topk_available:
        metric_row = f"{'Top k Activity Match Rate':<30}"
        for i, model_data in enumerate(metrics_data):
            metric_row += f" {model_data['topk']:<20.4f}"
        if len(models) > 1:
            diff = metrics_data[0]['topk'] - metrics_data[-1]['topk']
            diff_str = f"{diff:+.4f}" if abs(diff) > 0.0001 else "≈ 0.0000"
            metric_row += f" {diff_str:<15}"
        summary_lines.append(metric_row)
    
    summary_lines.append(f"\nTotal Evaluations:")
    for model in models:
        summary_lines.append(f"  {model['name']}: {len(model['results'])}")
    
    summary_text = "\n".join(summary_lines)
    
    # Save to file
    with open(f"{output_subdir}/summary.txt", 'w') as f:
        f.write(summary_text)
    
    return summary_text
