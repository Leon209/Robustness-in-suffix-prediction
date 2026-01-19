"""
Robustness Metrics Calculation Module

This module provides functions to calculate robustness metrics by comparing
predictions on original vs perturbed test data.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
from pyxdameraulevenshtein import damerau_levenshtein_distance

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Callable Functions:


def load_results(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Load robustness results from a pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Robustness results file not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected data format in {path}; expected a dict.")
    return data


def prepare_robustness_results(
    results: Dict[Tuple[str, int], Dict[str, Any]],
    concept_name: str = 'Activity',
    top_k: int = 3,
    save_path: Optional[str] = None
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Calculate robustness metrics for all entries in results.
    
    Args:
        results: Dictionary mapping (case_name, prefix_len) to result entries
        concept_name: Name of the concept to compare (default: 'Activity')
        top_k: Number of top sequences to consider (default: 3)
        save_path: Optional path to save updated results
    
    Returns:
        Updated results dictionary with computed metrics
    """
    for (case_name, prefix_len), entry in results.items():
        if 'original' not in entry or 'perturbed' not in entry:
            raise KeyError(
                "Each result entry must contain 'original' and 'perturbed' data to compute metrics."
            )
        
        # Handle both old (4-tuple) and new (6-tuple) formats for backward compatibility
        try:
            prefix_orig, suffix_orig, mean_orig, sampled_orig, mean_remaining_time_orig, sampled_remaining_time_orig = entry['original']
            prefix_pert, suffix_pert, mean_pert, sampled_pert, mean_remaining_time_pert, sampled_remaining_time_pert = entry['perturbed']
        except ValueError:
            # Old format - fallback to 4-tuple and calculate remaining time
            prefix_orig, suffix_orig, mean_orig, sampled_orig = entry['original']
            prefix_pert, suffix_pert, mean_pert, sampled_pert = entry['perturbed']
            mean_remaining_time_orig = None
            sampled_remaining_time_orig = None
            mean_remaining_time_pert = None
            sampled_remaining_time_pert = None
        entry['robustness_metrics'] = calculate_observation(
            mean_orig, mean_pert,
            sampled_orig, sampled_pert,
            case_name, prefix_len,
            concept_name, top_k
        )

        clean_support_ratio = 0.0
        clean_support_matches = 0
        clean_support_total = 0
        perturbed_support_ratio = 0.0
        perturbed_support_matches = 0
        perturbed_support_total = 0

        if sampled_orig is not None and suffix_orig is not None:
            (
                clean_support_ratio,
                clean_support_matches,
                clean_support_total,
            ) = _calculate_support_of_correct_prediction(
                sampled_orig, suffix_orig, concept_name
            )

        if sampled_pert is not None and suffix_orig is not None:
            (
                perturbed_support_ratio,
                perturbed_support_matches,
                perturbed_support_total,
            ) = _calculate_support_of_correct_prediction(
                sampled_pert, suffix_orig, concept_name
            )

        entry['support_of_correct_prediction'] = {
            'clean_ratio': clean_support_ratio,
            'clean_matches': clean_support_matches,
            'clean_total': clean_support_total,
            'perturbed_ratio': perturbed_support_ratio,
            'perturbed_matches': perturbed_support_matches,
            'perturbed_total': perturbed_support_total,
        }

        # Calculate Negative Log Likelihood (NLL)
        nll_clean = 0.0
        nll_perturbed = 0.0

        if sampled_orig is not None and suffix_orig is not None:
            nll_clean = _calculate_negative_log_likelihood(
                sampled_orig, suffix_orig, concept_name=concept_name
            )

        if sampled_pert is not None and suffix_orig is not None:
            nll_perturbed = _calculate_negative_log_likelihood(
                sampled_pert, suffix_orig, concept_name=concept_name
            )

        entry['nll_scores'] = {
            'clean': nll_clean,
            'perturbed': nll_perturbed,
        }

        # Calculate Wasserstein distance between clean and perturbed distributions
        wasserstein_distance = 0.0
        if sampled_orig is not None and sampled_pert is not None and len(sampled_orig) > 0 and len(sampled_pert) > 0:
            wasserstein_distance = _calculate_wasserstein_distance(
                sampled_orig, sampled_pert, concept_name=concept_name
            )

        entry['wasserstein_distance'] = wasserstein_distance

        # Calculate ROUGE-L and chrF scores for modal predictions
        rouge_l_clean = 0.0
        rouge_l_perturbed = 0.0
        chrf_clean = 0.0
        chrf_perturbed = 0.0

        if sampled_orig is not None and suffix_orig is not None and len(sampled_orig) > 0:
            modal_clean = _get_modal_prediction(sampled_orig, concept_name=concept_name)
            rouge_l_clean = _calculate_rouge_l(modal_clean, suffix_orig, concept_name=concept_name)
            chrf_clean = _calculate_chrf(modal_clean, suffix_orig, concept_name=concept_name)

        if sampled_pert is not None and suffix_orig is not None and len(sampled_pert) > 0:
            modal_perturbed = _get_modal_prediction(sampled_pert, concept_name=concept_name)
            rouge_l_perturbed = _calculate_rouge_l(modal_perturbed, suffix_orig, concept_name=concept_name)
            chrf_perturbed = _calculate_chrf(modal_perturbed, suffix_orig, concept_name=concept_name)

        entry['rouge_l_scores'] = {
            'clean': rouge_l_clean,
            'perturbed': rouge_l_perturbed,
        }

        entry['chrf_scores'] = {
            'clean': chrf_clean,
            'perturbed': chrf_perturbed,
        }

        # Calculate Remaining Time Prediction Shift
        # Use pre-calculated remaining times if available (new format), otherwise calculate (old format)
        remaining_time_shift = None
        
        if mean_remaining_time_orig is not None and mean_remaining_time_pert is not None:
            # New format: use pre-calculated remaining times
            remaining_time_shift = abs(mean_remaining_time_orig - mean_remaining_time_pert)
        else:
            # Old format: calculate remaining time from predictions
            # Check if we have the necessary data for remaining time calculation
            has_elapsed_time = False
            if prefix_orig and len(prefix_orig) > 0:
                # Check if prefix has case_elapsed_time field
                last_prefix_event = prefix_orig[-1]
                if isinstance(last_prefix_event, dict) and 'case_elapsed_time' in last_prefix_event:
                    has_elapsed_time = True
            
            if has_elapsed_time:
                # Get current case elapsed time at prefix (should be same for clean and perturbed)
                current_case_elapsed_time_at_prefix = prefix_orig[-1].get('case_elapsed_time', 0.0)
                
                # Calculate for most-likely (mean) predictions
                remaining_time_clean = None
                remaining_time_pert = None
                
                # Clean prediction - use mean_orig
                if mean_orig and len(mean_orig) > 0:
                    last_event_clean = mean_orig[-1]
                    if isinstance(last_event_clean, dict) and 'case_elapsed_time' in last_event_clean:
                        final_predicted_case_elapsed_time_clean = last_event_clean.get('case_elapsed_time', 0.0)
                        remaining_time_clean = final_predicted_case_elapsed_time_clean - current_case_elapsed_time_at_prefix
                
                # Perturbed prediction - use mean_pert
                if mean_pert and len(mean_pert) > 0:
                    last_event_pert = mean_pert[-1]
                    if isinstance(last_event_pert, dict) and 'case_elapsed_time' in last_event_pert:
                        final_predicted_case_elapsed_time_pert = last_event_pert.get('case_elapsed_time', 0.0)
                        remaining_time_pert = final_predicted_case_elapsed_time_pert - current_case_elapsed_time_at_prefix
                
                # Calculate shift for most-likely prediction
                if remaining_time_clean is not None and remaining_time_pert is not None:
                    remaining_time_shift = abs(remaining_time_clean - remaining_time_pert)
        
        entry['remaining_time_prediction_shift'] = remaining_time_shift

        # Calculate DLS between clean and perturbed predictions (most-likely)
        # This measures how much the prediction changed due to the perturbation
        # Lower DLS values indicate bigger prediction shifts (more adversarial effect)
        prediction_shift_dls = 0.0
        if mean_orig is not None and mean_pert is not None:
            prediction_shift_dls = _calculate_dls(
                mean_orig, mean_pert, concept_name=concept_name
            )
        
        entry['prediction_shift_dls'] = prediction_shift_dls
    
    if save_path:
        save_results(save_path, results)
    
    return results


def calculate_aggregate_metrics(results: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, List]:
    """
    Calculate aggregate metrics by prefix length from results.

    Returns:
        Dictionary containing:
            - prefix_lengths: List of prefix lengths
            - activity_match_rates: List of mean activity match rates per prefix length
            - length_match_rates: List of mean length match rates per prefix length
            - top_k_activity_match_rates: List of mean top-k activity match rates per prefix length
            - sample_counts: List of sample counts per prefix length
            - clean_dls: List of mean DLS scores per prefix length (mean_orig vs suffix_orig)
            - perturbed_dls: List of mean DLS scores per prefix length (mean_pert vs suffix_orig)
            - relative_dls_drop: List of mean relative DLS drop per prefix length (perturbed_dls / clean_dls)
            - modal_clean_dls: List of mean modal prediction DLS per prefix length (clean)
            - modal_perturbed_dls: List of mean modal prediction DLS per prefix length (perturbed)
            - clean_dls_q25: List of mean 25th percentile DLS per prefix length (clean)
            - clean_dls_q75: List of mean 75th percentile DLS per prefix length (clean)
            - perturbed_dls_q25: List of mean 25th percentile DLS per prefix length (perturbed)
            - perturbed_dls_q75: List of mean 75th percentile DLS per prefix length (perturbed)
            - support_clean: List of mean support of correct prediction per prefix length (clean)
            - support_perturbed: List of mean support of correct prediction per prefix length (perturbed)
            - rouge_l_clean: List of mean ROUGE-L score per prefix length (clean, modal prediction)
            - rouge_l_perturbed: List of mean ROUGE-L score per prefix length (perturbed, modal prediction)
            - chrf_clean: List of mean chrF score per prefix length (clean, modal prediction)
            - chrf_perturbed: List of mean chrF score per prefix length (perturbed, modal prediction)
            - nll_clean: List of mean negative log-likelihood per prefix length (clean)
            - nll_perturbed: List of mean negative log-likelihood per prefix length (perturbed)
            - wasserstein_distance: List of mean Wasserstein distance per prefix length (between clean and perturbed distributions)
            - remaining_time_prediction_shift: List of mean remaining time prediction shift per prefix length (most-likely predictions only)
    """
    # Group metrics by prefix length
    by_prefix = defaultdict(list)
    clean_dls_by_prefix = defaultdict(list)
    perturbed_dls_by_prefix = defaultdict(list)
    relative_dls_drop_by_prefix = defaultdict(list)
    modal_clean_dls_by_prefix = defaultdict(list)
    modal_perturbed_dls_by_prefix = defaultdict(list)
    clean_dls_q25_by_prefix = defaultdict(list)
    clean_dls_q75_by_prefix = defaultdict(list)
    perturbed_dls_q25_by_prefix = defaultdict(list)
    perturbed_dls_q75_by_prefix = defaultdict(list)
    support_clean_by_prefix = defaultdict(list)
    support_perturbed_by_prefix = defaultdict(list)
    rouge_l_clean_by_prefix = defaultdict(list)
    rouge_l_perturbed_by_prefix = defaultdict(list)
    chrf_clean_by_prefix = defaultdict(list)
    chrf_perturbed_by_prefix = defaultdict(list)
    nll_clean_by_prefix = defaultdict(list)
    nll_perturbed_by_prefix = defaultdict(list)
    wasserstein_distance_by_prefix = defaultdict(list)
    remaining_time_shift_by_prefix = defaultdict(list)
    concept_name = 'Activity'
    
    for (case_name, prefix_len), entry in results.items():
        if 'robustness_metrics' in entry:
            metrics = entry['robustness_metrics']
            by_prefix[prefix_len].append(metrics)

            support_metrics = entry.get('support_of_correct_prediction', {})
            if support_metrics:
                support_clean_by_prefix[prefix_len].append(
                    support_metrics.get('clean_ratio', 0.0)
                )
                support_perturbed_by_prefix[prefix_len].append(
                    support_metrics.get('perturbed_ratio', 0.0)
                )

            rouge_l_metrics = entry.get('rouge_l_scores', {})
            if rouge_l_metrics:
                rouge_l_clean_by_prefix[prefix_len].append(
                    rouge_l_metrics.get('clean', 0.0)
                )
                rouge_l_perturbed_by_prefix[prefix_len].append(
                    rouge_l_metrics.get('perturbed', 0.0)
                )

            chrf_metrics = entry.get('chrf_scores', {})
            if chrf_metrics:
                chrf_clean_by_prefix[prefix_len].append(
                    chrf_metrics.get('clean', 0.0)
                )
                chrf_perturbed_by_prefix[prefix_len].append(
                    chrf_metrics.get('perturbed', 0.0)
                )

            nll_metrics = entry.get('nll_scores', {})
            if nll_metrics:
                nll_clean_by_prefix[prefix_len].append(
                    nll_metrics.get('clean', 0.0)
                )
                nll_perturbed_by_prefix[prefix_len].append(
                    nll_metrics.get('perturbed', 0.0)
                )

            wasserstein_distance = entry.get('wasserstein_distance')
            if wasserstein_distance is not None:
                wasserstein_distance_by_prefix[prefix_len].append(wasserstein_distance)
            
            # Extract remaining time prediction shift
            remaining_time_shift = entry.get('remaining_time_prediction_shift')
            if remaining_time_shift is not None:
                remaining_time_shift_by_prefix[prefix_len].append(remaining_time_shift)
            
            # Calculate clean DLS (mean_orig vs suffix_orig)
            if 'original' in entry:
                try:
                    # Handle both old (4-tuple) and new (6-tuple) formats
                    orig_tuple = entry['original']
                    if len(orig_tuple) == 6:
                        _, suffix_orig, mean_orig, sampled_orig, _, _ = orig_tuple
                    else:
                        _, suffix_orig, mean_orig, sampled_orig = orig_tuple
                except (ValueError, TypeError):
                    suffix_orig, mean_orig, sampled_orig = None, None, None
                if mean_orig is not None and suffix_orig is not None:
                    clean_dls_value = _calculate_dls(mean_orig, suffix_orig, concept_name=concept_name)
                    clean_dls_by_prefix[prefix_len].append(clean_dls_value)
                    
                    # Calculate modal prediction DLS for clean
                    if sampled_orig is not None and len(sampled_orig) > 0:
                        modal_clean = _get_modal_prediction(sampled_orig, concept_name=concept_name)
                        modal_clean_dls_value = _calculate_dls(modal_clean, suffix_orig, concept_name=concept_name)
                        modal_clean_dls_by_prefix[prefix_len].append(modal_clean_dls_value)
                        
                        # Calculate DLS for all probabilistic samples (for percentile calculation)
                        clean_dls_samples = _calculate_dls_from_samples(sampled_orig, suffix_orig, concept_name=concept_name)
                        if clean_dls_samples:
                            clean_dls_q25 = np.percentile(clean_dls_samples, 25)
                            clean_dls_q75 = np.percentile(clean_dls_samples, 75)
                            clean_dls_q25_by_prefix[prefix_len].append(clean_dls_q25)
                            clean_dls_q75_by_prefix[prefix_len].append(clean_dls_q75)
                    
                    # Calculate perturbed DLS (mean_pert vs suffix_orig)
                    if 'perturbed' in entry:
                        try:
                            # Handle both old (4-tuple) and new (6-tuple) formats
                            pert_tuple = entry['perturbed']
                            if len(pert_tuple) == 6:
                                _, _, mean_pert, sampled_pert, _, _ = pert_tuple
                            else:
                                _, _, mean_pert, sampled_pert = pert_tuple
                        except (ValueError, TypeError):
                            mean_pert, sampled_pert = None, None
                        if mean_pert is not None:
                            perturbed_dls_value = _calculate_dls(mean_pert, suffix_orig, concept_name=concept_name)
                            perturbed_dls_by_prefix[prefix_len].append(perturbed_dls_value)
                            
                            # Calculate modal prediction DLS for perturbed
                            if sampled_pert is not None and len(sampled_pert) > 0:
                                modal_perturbed = _get_modal_prediction(sampled_pert, concept_name=concept_name)
                                modal_perturbed_dls_value = _calculate_dls(modal_perturbed, suffix_orig, concept_name=concept_name)
                                modal_perturbed_dls_by_prefix[prefix_len].append(modal_perturbed_dls_value)
                                
                                # Calculate DLS for all probabilistic samples (for percentile calculation)
                                perturbed_dls_samples = _calculate_dls_from_samples(sampled_pert, suffix_orig, concept_name=concept_name)
                                if perturbed_dls_samples:
                                    perturbed_dls_q25 = np.percentile(perturbed_dls_samples, 25)
                                    perturbed_dls_q75 = np.percentile(perturbed_dls_samples, 75)
                                    perturbed_dls_q25_by_prefix[prefix_len].append(perturbed_dls_q25)
                                    perturbed_dls_q75_by_prefix[prefix_len].append(perturbed_dls_q75)
                            
                            # Calculate relative DLS drop (perturbed_dls / clean_dls)
                            # This represents how well the model performs under attack relative to clean performance
                            # 1.0 = no drop, < 1.0 = performance drop, > 1.0 = performance improvement
                            if clean_dls_value > 0:
                                relative_drop = perturbed_dls_value / clean_dls_value
                            else:
                                # Edge case: if clean_dls is 0 (perfect prediction)
                                # If perturbed is also 0, no change (1.0)
                                # If perturbed > 0, complete drop (0.0)
                                relative_drop = 1.0 if perturbed_dls_value == 0 else 0.0
                            relative_dls_drop_by_prefix[prefix_len].append(relative_drop)
    
    prefix_lengths = []
    activity_match_rates = []
    length_match_rates = []
    top_k_activity_match_rates = []
    clean_dls = []
    perturbed_dls = []
    relative_dls_drop = []
    modal_clean_dls = []
    modal_perturbed_dls = []
    clean_dls_q25 = []
    clean_dls_q75 = []
    perturbed_dls_q25 = []
    perturbed_dls_q75 = []
    sample_counts = []
    support_clean = []
    support_perturbed = []
    chrf_clean = []
    chrf_perturbed = []
    rouge_l_clean = []
    rouge_l_perturbed = []
    nll_clean = []
    nll_perturbed = []
    wasserstein_distance = []
    remaining_time_prediction_shift = []
    
    for prefix_len in sorted(by_prefix.keys()):
        metrics_group = by_prefix[prefix_len]
        mean_metrics = [m['mean_prediction'] for m in metrics_group]
        prob_metrics = [m.get('probabilistic_prediction', {}) for m in metrics_group]
        prob_metrics = [m for m in prob_metrics if m]
        
        prefix_lengths.append(prefix_len)
        activity_match_rates.append(np.mean([m['activity_sequence_match'] for m in mean_metrics]))
        length_match_rates.append(np.mean([m['length_match'] for m in mean_metrics]))
        
        # Aggregate DLS metrics
        clean_dls_values = clean_dls_by_prefix.get(prefix_len, [])
        clean_dls.append(float(np.mean(clean_dls_values)) if clean_dls_values else 0.0)
        
        perturbed_dls_values = perturbed_dls_by_prefix.get(prefix_len, [])
        perturbed_dls.append(float(np.mean(perturbed_dls_values)) if perturbed_dls_values else 0.0)
        
        relative_drop_values = relative_dls_drop_by_prefix.get(prefix_len, [])
        relative_dls_drop.append(float(np.mean(relative_drop_values)) if relative_drop_values else 0.0)
        
        # Aggregate modal DLS metrics
        modal_clean_dls_values = modal_clean_dls_by_prefix.get(prefix_len, [])
        modal_clean_dls.append(float(np.mean(modal_clean_dls_values)) if modal_clean_dls_values else 0.0)
        
        modal_perturbed_dls_values = modal_perturbed_dls_by_prefix.get(prefix_len, [])
        modal_perturbed_dls.append(float(np.mean(modal_perturbed_dls_values)) if modal_perturbed_dls_values else 0.0)
        
        # Aggregate percentile DLS metrics
        clean_dls_q25_values = clean_dls_q25_by_prefix.get(prefix_len, [])
        clean_dls_q25.append(float(np.mean(clean_dls_q25_values)) if clean_dls_q25_values else 0.0)
        
        clean_dls_q75_values = clean_dls_q75_by_prefix.get(prefix_len, [])
        clean_dls_q75.append(float(np.mean(clean_dls_q75_values)) if clean_dls_q75_values else 0.0)
        
        perturbed_dls_q25_values = perturbed_dls_q25_by_prefix.get(prefix_len, [])
        perturbed_dls_q25.append(float(np.mean(perturbed_dls_q25_values)) if perturbed_dls_q25_values else 0.0)
        
        perturbed_dls_q75_values = perturbed_dls_q75_by_prefix.get(prefix_len, [])
        perturbed_dls_q75.append(float(np.mean(perturbed_dls_q75_values)) if perturbed_dls_q75_values else 0.0)
        
        sample_counts.append(len(metrics_group))

        support_clean_values = support_clean_by_prefix.get(prefix_len, [])
        support_clean.append(
            float(np.mean(support_clean_values)) if support_clean_values else 0.0
        )

        support_perturbed_values = support_perturbed_by_prefix.get(prefix_len, [])
        support_perturbed.append(
            float(np.mean(support_perturbed_values)) if support_perturbed_values else 0.0
        )

        chrf_clean_values = chrf_clean_by_prefix.get(prefix_len, [])
        chrf_clean.append(
            float(np.mean(chrf_clean_values)) if chrf_clean_values else 0.0
        )

        chrf_perturbed_values = chrf_perturbed_by_prefix.get(prefix_len, [])
        chrf_perturbed.append(
            float(np.mean(chrf_perturbed_values)) if chrf_perturbed_values else 0.0
        )

        rouge_l_clean_values = rouge_l_clean_by_prefix.get(prefix_len, [])
        rouge_l_clean.append(
            float(np.mean(rouge_l_clean_values)) if rouge_l_clean_values else 0.0
        )

        rouge_l_perturbed_values = rouge_l_perturbed_by_prefix.get(prefix_len, [])
        rouge_l_perturbed.append(
            float(np.mean(rouge_l_perturbed_values)) if rouge_l_perturbed_values else 0.0
        )

        nll_clean_values = nll_clean_by_prefix.get(prefix_len, [])
        nll_clean.append(
            float(np.mean(nll_clean_values)) if nll_clean_values else 0.0
        )

        nll_perturbed_values = nll_perturbed_by_prefix.get(prefix_len, [])
        nll_perturbed.append(
            float(np.mean(nll_perturbed_values)) if nll_perturbed_values else 0.0
        )

        wasserstein_distance_values = wasserstein_distance_by_prefix.get(prefix_len, [])
        wasserstein_distance.append(
            float(np.mean(wasserstein_distance_values)) if wasserstein_distance_values else 0.0
        )
        
        # Aggregate remaining time prediction shift metrics
        remaining_time_shift_values = remaining_time_shift_by_prefix.get(prefix_len, [])
        remaining_time_prediction_shift.append(
            float(np.mean(remaining_time_shift_values)) if remaining_time_shift_values else 0.0
        )
        
        if prob_metrics:
            top_k_activity_match_rates.append(np.mean([m.get('top_k_activity_match_rate', 0.0) for m in prob_metrics]))
        else:
            top_k_activity_match_rates.append(0.0)
    
    return {
        'prefix_lengths': prefix_lengths,
        'activity_match_rates': activity_match_rates,
        'length_match_rates': length_match_rates,
        'top_k_activity_match_rates': top_k_activity_match_rates,
        'clean_dls': clean_dls,
        'perturbed_dls': perturbed_dls,
        'relative_dls_drop': relative_dls_drop,
        'modal_clean_dls': modal_clean_dls,
        'modal_perturbed_dls': modal_perturbed_dls,
        'clean_dls_q25': clean_dls_q25,
        'clean_dls_q75': clean_dls_q75,
        'perturbed_dls_q25': perturbed_dls_q25,
        'perturbed_dls_q75': perturbed_dls_q75,
        'sample_counts': sample_counts,
        'support_clean': support_clean,
        'support_perturbed': support_perturbed,
        'rouge_l_clean': rouge_l_clean,
        'rouge_l_perturbed': rouge_l_perturbed,
        'chrf_clean': chrf_clean,
        'chrf_perturbed': chrf_perturbed,
        'nll_clean': nll_clean,
        'nll_perturbed': nll_perturbed,
        'wasserstein_distance': wasserstein_distance,
        'remaining_time_prediction_shift': remaining_time_prediction_shift,
    }


# ============================================================================
# Helper Functions

def save_results(path: str, results: Dict[Tuple[str, int], Dict[str, Any]]) -> None:
    """Save robustness results to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def calculate_observation(
    mean_pred_orig: List[Dict],
    mean_pred_pert: List[Dict],
    prob_pred_orig: List[List[Dict]],
    prob_pred_pert: List[List[Dict]],
    case_name: str,
    prefix_len: int,
    concept_name: str = 'Activity',
    top_k: int = 3
) -> Dict[str, Any]:
    """Calculate robustness metrics for a single observation."""
    metrics = {
        'case_name': case_name,
        'prefix_len': prefix_len,
        'mean_prediction': _compare_mean_predictions(mean_pred_orig, mean_pred_pert, concept_name),
        'probabilistic_prediction': _compare_probabilistic_predictions(
            prob_pred_orig, prob_pred_pert, concept_name, top_k
        )
    }
    return metrics


def _compare_mean_predictions(
    pred_orig: List[Dict],
    pred_pert: List[Dict],
    concept_name: str = 'Activity'
) -> Dict:
    """Compare mean predictions (single deterministic prediction)."""
    return {
        'length_match': len(pred_orig) == len(pred_pert),
        'activity_sequence_match': _compare_activity_sequences(pred_orig, pred_pert, concept_name),
        'length_original': len(pred_orig),
        'length_perturbed': len(pred_pert)
    }


def _compare_probabilistic_predictions(
    pred_orig: List[List[Dict]],
    pred_pert: List[List[Dict]],
    concept_name: str = 'Activity',
    top_k: int = 3
) -> Dict:
    """Compare probabilistic predictions across multiple samples."""
    num_samples = len(pred_orig)
    assert len(pred_pert) == num_samples, "Sample count mismatch"

    lengths_orig = [len(sample) for sample in pred_orig]
    lengths_pert = [len(sample) for sample in pred_pert]

    avg_length_original = float(np.mean(lengths_orig)) if lengths_orig else 0.0
    avg_length_perturbed = float(np.mean(lengths_pert)) if lengths_pert else 0.0
    var_length_original = float(np.var(lengths_orig)) if lengths_orig else 0.0
    var_length_perturbed = float(np.var(lengths_pert)) if lengths_pert else 0.0

    top_k_activity_match_rate = _top_k_activity_match_rate(pred_orig, pred_pert, concept_name, top_k)

    return {
        'top_k': top_k,
        'top_k_activity_match_rate': top_k_activity_match_rate,
        'avg_length_original': avg_length_original,
        'avg_length_perturbed': avg_length_perturbed,
        'var_length_original': var_length_original,
        'var_length_perturbed': var_length_perturbed
    }


def _compare_activity_sequences(
    pred_orig: List[Dict],
    pred_pert: List[Dict],
    concept_name: str = 'Activity'
) -> bool:
    """Compare activity sequences between predictions."""
    seq_orig = [event.get(concept_name) for event in pred_orig]
    seq_pert = [event.get(concept_name) for event in pred_pert]
    return seq_orig == seq_pert


def _top_k_activity_match_rate(
    pred_orig: List[List[Dict]],
    pred_pert: List[List[Dict]],
    concept_name: str = 'Activity',
    top_k: int = 3
) -> float:
    """Calculate top-k activity sequence match rate."""
    top_k_orig = _get_top_k_sequences(pred_orig, concept_name, top_k)
    top_k_pert = _get_top_k_sequences(pred_pert, concept_name, top_k)
    matching_sequences = sum(1 for seq in top_k_orig if seq in top_k_pert)
    return matching_sequences / top_k if top_k > 0 else 0.0


def _get_top_k_sequences(
    prob_predictions: List[List[Dict]],
    concept_name: str = 'Activity',
    top_k: int = 3
) -> List[tuple]:
    """Extract activity sequences from probabilistic predictions and return top k by frequency."""
    sequence_counts = defaultdict(int)
    for sample in prob_predictions:
        seq = tuple([event.get(concept_name) for event in sample])
        sequence_counts[seq] += 1
    sorted_sequences = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in sorted_sequences[:top_k]]


def _get_modal_prediction(
    prob_predictions: List[List[Dict]],
    concept_name: str = 'Activity'
) -> List[Dict]:
    """
    Get the modal (most frequent) prediction from probabilistic samples.
    
    Args:
        prob_predictions: List of probabilistic prediction samples, each a list of events
        concept_name: Key used to extract activity labels from events
        
    Returns:
        The most frequently predicted sequence as a list of event dictionaries
    """
    if not prob_predictions:
        return []
    
    # Get the most frequent sequence (top-1)
    top_sequences = _get_top_k_sequences(prob_predictions, concept_name, top_k=1)
    if not top_sequences:
        return []
    
    modal_sequence = top_sequences[0]  # Most frequent sequence as tuple of activities
    
    # Find the first sample that matches this sequence and return it
    for sample in prob_predictions:
        sample_sequence = tuple([event.get(concept_name) for event in sample])
        if sample_sequence == modal_sequence:
            return sample
    
    # Fallback: return first sample if no match found (shouldn't happen)
    return prob_predictions[0]


def _calculate_dls(
    mean_prediction: List[Dict],
    true_suffix: List[Dict],
    concept_name: str = 'Activity'
) -> float:
    """
    Calculate the Damerau–Levenshtein Similarity (DLS) between predicted and true sequences.

    Args:
        mean_prediction: List of events representing the predicted mean suffix.
        true_suffix: List of events representing the true suffix.
        concept_name: Key used to extract activity labels from events.

    Returns:
        A float representing the DLS between the two sequences, in the range [0, 1].
    """
    predicted_sequence = [event.get(concept_name) for event in mean_prediction]
    true_sequence = [event.get(concept_name) for event in true_suffix]

    max_length = max(len(predicted_sequence), len(true_sequence))
    if max_length == 0:
        return 1.0

    distance = damerau_levenshtein_distance(predicted_sequence, true_sequence)
    similarity = 1 - (distance / max_length)
    return float(similarity)


def _calculate_dls_from_samples(
    prob_predictions: List[List[Dict]],
    true_suffix: List[Dict],
    concept_name: str = 'Activity'
) -> List[float]:
    """
    Calculate DLS for each probabilistic sample against the true suffix.
    
    Args:
        prob_predictions: List of probabilistic prediction samples, each a list of events
        true_suffix: List of events representing the true suffix
        concept_name: Key used to extract activity labels from events
        
    Returns:
        List of DLS values, one for each probabilistic sample
    """
    dls_values = []
    for sample in prob_predictions:
        dls = _calculate_dls(sample, true_suffix, concept_name)
        dls_values.append(dls)
    return dls_values


def _calculate_support_of_correct_prediction(
    prob_predictions: List[List[Dict]],
    true_suffix: List[Dict],
    concept_name: str = 'Activity',
) -> Tuple[float, int, int]:
    """
    Calculate the support of correct prediction (exact-match frequency).

    Returns:
        Tuple of (ratio, matches, total_samples).
    """
    if not prob_predictions or true_suffix is None:
        return 0.0, 0, 0

    true_sequence = tuple(event.get(concept_name) for event in true_suffix)
    matches = 0
    for sample in prob_predictions:
        sample_sequence = tuple(event.get(concept_name) for event in sample)
        if sample_sequence == true_sequence:
            matches += 1

    total_samples = len(prob_predictions)
    ratio = matches / total_samples if total_samples > 0 else 0.0
    return float(ratio), matches, total_samples


def _calculate_negative_log_likelihood(
    prob_predictions: List[List[Dict]],
    true_suffix: List[Dict],
    concept_name: str = 'Activity',
    epsilon: float = 1e-10,
) -> float:
    """
    Calculate negative log-likelihood (NLL) of the true sequence.
    
    The empirical probability is calculated as the frequency of the true sequence
    appearing in the probabilistic samples. If the probability is 0, it is clipped
    to epsilon to avoid -inf.
    
    Args:
        prob_predictions: List of probabilistic prediction samples, each a list of events
        true_suffix: List of events representing the true suffix
        concept_name: Key used to extract activity labels from events
        epsilon: Small value to clip probability when it's 0 (default: 1e-10)
        
    Returns:
        Negative log-likelihood value (float). Lower values indicate better performance.
    """
    if not prob_predictions or true_suffix is None:
        # Return a high NLL value when no predictions or true suffix
        return -np.log(epsilon)
    
    true_sequence = tuple(event.get(concept_name) for event in true_suffix)
    matches = 0
    for sample in prob_predictions:
        sample_sequence = tuple(event.get(concept_name) for event in sample)
        if sample_sequence == true_sequence:
            matches += 1
    
    total_samples = len(prob_predictions)
    if total_samples == 0:
        return -np.log(epsilon)
    
    probability = matches / total_samples
    # Clip probability to epsilon to avoid -inf when probability is 0
    clipped_probability = max(probability, epsilon)
    nll = -np.log(clipped_probability)
    return float(nll)


def _calculate_wasserstein_distance(
    prob_predictions_clean: List[List[Dict]],
    prob_predictions_perturbed: List[List[Dict]],
    concept_name: str = 'Activity',
) -> float:
    """
    Calculate Wasserstein distance between clean and perturbed probabilistic prediction distributions.
    
    The cost metric is normalized DLS: cost(seq1, seq2) = 1 - DLS(seq1, seq2)
    where DLS is the Damerau-Levenshtein Similarity.
    
    Args:
        prob_predictions_clean: List of probabilistic prediction samples (clean setting)
        prob_predictions_perturbed: List of probabilistic prediction samples (perturbed setting)
        concept_name: Key used to extract activity labels from events
        
    Returns:
        Wasserstein distance value (float). Higher values indicate greater distribution shift.
    """
    if not prob_predictions_clean or not prob_predictions_perturbed:
        return 0.0
    
    # Extract unique sequences and compute empirical probabilities
    clean_sequences = [tuple(event.get(concept_name) for event in sample) for sample in prob_predictions_clean]
    perturbed_sequences = [tuple(event.get(concept_name) for event in sample) for sample in prob_predictions_perturbed]
    
    # Count frequencies
    clean_counts = Counter(clean_sequences)
    perturbed_counts = Counter(perturbed_sequences)
    
    # Get all unique sequences from both distributions
    all_sequences = list(set(clean_sequences) | set(perturbed_sequences))
    
    if len(all_sequences) == 0:
        return 0.0
    
    # Compute empirical probabilities
    n_clean = len(prob_predictions_clean)
    n_perturbed = len(prob_predictions_perturbed)
    
    clean_probs = np.array([clean_counts.get(seq, 0) / n_clean for seq in all_sequences])
    perturbed_probs = np.array([perturbed_counts.get(seq, 0) / n_perturbed for seq in all_sequences])
    
    # Normalize to ensure they sum to 1 (handle floating point issues)
    clean_probs = clean_probs / clean_probs.sum() if clean_probs.sum() > 0 else clean_probs
    perturbed_probs = perturbed_probs / perturbed_probs.sum() if perturbed_probs.sum() > 0 else perturbed_probs
    
    # Build cost matrix: normalized DLS = 1 - DLS
    n = len(all_sequences)
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        seq1 = list(all_sequences[i])
        for j in range(n):
            seq2 = list(all_sequences[j])
            # Calculate DLS
            dls = _calculate_dls_from_sequences(seq1, seq2)
            # Normalized DLS (distance) = 1 - DLS (similarity)
            cost_matrix[i, j] = 1.0 - dls
    
    # Solve optimal transport problem
    if SCIPY_AVAILABLE and n <= 100:  # Use scipy for reasonable problem sizes
        try:
            # Flatten cost matrix for linprog
            c = cost_matrix.flatten()
            
            # Constraints: sum of transport from each source = clean_probs
            #              sum of transport to each sink = perturbed_probs
            A_eq = []
            b_eq = []
            
            # Row constraints (sum over columns = clean_probs)
            for i in range(n):
                row = np.zeros(n * n)
                row[i * n:(i + 1) * n] = 1.0
                A_eq.append(row)
                b_eq.append(clean_probs[i])
            
            # Column constraints (sum over rows = perturbed_probs)
            for j in range(n):
                col = np.zeros(n * n)
                for i in range(n):
                    col[i * n + j] = 1.0
                A_eq.append(col)
                b_eq.append(perturbed_probs[j])
            
            A_eq = np.array(A_eq)
            b_eq = np.array(b_eq)
            
            # Bounds: transport amounts must be non-negative
            bounds = [(0, None) for _ in range(n * n)]
            
            # Solve linear program
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if result.success:
                # Wasserstein distance is the optimal cost
                transport = result.x.reshape(n, n)
                wasserstein = np.sum(transport * cost_matrix)
                return float(wasserstein)
        except Exception:
            # Fall back to simple implementation if scipy fails
            pass
    
    # Fallback: Simple approximation using greedy matching or direct calculation
    # For small problems, we can compute directly
    if n <= 20:
        # Use Hungarian algorithm approximation or direct enumeration
        # For simplicity, use a greedy approach or compute exact for very small n
        wasserstein = _compute_wasserstein_simple(clean_probs, perturbed_probs, cost_matrix)
        return float(wasserstein)
    else:
        # For large problems, use a simpler approximation
        # Compute as weighted average of costs
        wasserstein = np.sum(clean_probs[:, np.newaxis] * perturbed_probs[np.newaxis, :] * cost_matrix)
        return float(wasserstein)


def _calculate_dls_from_sequences(seq1: List, seq2: List) -> float:
    """
    Calculate DLS between two sequences (as lists of activity names).
    
    Args:
        seq1: First sequence as list of activity names
        seq2: Second sequence as list of activity names
        
    Returns:
        DLS similarity value in [0, 1]
    """
    max_length = max(len(seq1), len(seq2))
    if max_length == 0:
        return 1.0
    
    distance = damerau_levenshtein_distance(seq1, seq2)
    similarity = 1 - (distance / max_length)
    return float(similarity)


def _compute_wasserstein_simple(
    p: np.ndarray,
    q: np.ndarray,
    cost_matrix: np.ndarray
) -> float:
    """
    Simple Wasserstein distance computation for small problems.
    Uses a greedy matching approach.
    
    Args:
        p: Source distribution probabilities
        q: Target distribution probabilities
        cost_matrix: Cost matrix between all pairs
        
    Returns:
        Approximate Wasserstein distance
    """
    n = len(p)
    wasserstein = 0.0
    p_remaining = p.copy()
    q_remaining = q.copy()
    
    # Greedy matching: repeatedly match highest probability pairs with lowest cost
    while np.sum(p_remaining) > 1e-10 and np.sum(q_remaining) > 1e-10:
        # Find the pair with minimum cost among remaining non-zero probabilities
        min_cost = np.inf
        min_i, min_j = -1, -1
        
        for i in range(n):
            if p_remaining[i] < 1e-10:
                continue
            for j in range(n):
                if q_remaining[j] < 1e-10:
                    continue
                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    min_i, min_j = i, j
        
        if min_i == -1 or min_j == -1:
            break
        
        # Transport as much as possible
        transport = min(p_remaining[min_i], q_remaining[min_j])
        wasserstein += transport * cost_matrix[min_i, min_j]
        p_remaining[min_i] -= transport
        q_remaining[min_j] -= transport
    
    return wasserstein


def _longest_common_subsequence_length(seq1: List, seq2: List) -> int:
    """
    Calculate the length of the longest common subsequence (LCS) between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Length of the LCS
    """
    if not seq1 or not seq2:
        return 0
    
    m, n = len(seq1), len(seq2)
    # Create a 2D table to store LCS lengths
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def _calculate_rouge_l(
    predicted: List[Dict],
    reference: List[Dict],
    concept_name: str = 'Activity',
) -> float:
    """
    Calculate ROUGE-L score between predicted and reference sequences.
    
    ROUGE-L = (2 * LCS_length) / (len(predicted) + len(reference))
    
    Args:
        predicted: List of events representing the predicted sequence
        reference: List of events representing the reference (true) sequence
        concept_name: Key used to extract activity labels from events
        
    Returns:
        ROUGE-L score in the range [0, 1]
    """
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    
    pred_sequence = [event.get(concept_name) for event in predicted]
    ref_sequence = [event.get(concept_name) for event in reference]
    
    lcs_length = _longest_common_subsequence_length(pred_sequence, ref_sequence)
    
    total_length = len(pred_sequence) + len(ref_sequence)
    if total_length == 0:
        return 1.0
    
    rouge_l = (2.0 * lcs_length) / total_length
    return float(rouge_l)


def _get_char_ngrams(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def _calculate_chrf(
    predicted: List[Dict],
    reference: List[Dict],
    concept_name: str = 'Activity',
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """
    Calculate chrF score between predicted and reference sequences.
    """
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0

    pred_sequence = ' '.join(event.get(concept_name, '') or '' for event in predicted)
    ref_sequence = ' '.join(event.get(concept_name, '') or '' for event in reference)

    if not pred_sequence and not ref_sequence:
        return 1.0
    if not pred_sequence or not ref_sequence:
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    effective_orders = 0

    for order in range(1, max_order + 1):
        pred_ngrams = _get_char_ngrams(pred_sequence, order)
        ref_ngrams = _get_char_ngrams(ref_sequence, order)

        if not pred_ngrams and not ref_ngrams:
            continue

        overlap = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
        precision = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

        total_precision += precision
        total_recall += recall
        effective_orders += 1

    if effective_orders == 0:
        return 0.0

    avg_precision = total_precision / effective_orders
    avg_recall = total_recall / effective_orders

    if avg_precision == 0 and avg_recall == 0:
        return 0.0

    beta_sq = beta ** 2
    chrf = (1 + beta_sq) * (avg_precision * avg_recall) / (
        (beta_sq * avg_precision) + avg_recall
    )
    return float(chrf)


def _aggregate_mean_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate mean prediction metrics."""
    return {
        'activity_match_rate': np.mean([m['activity_sequence_match'] for m in metrics_list]),
        'length_match_rate': np.mean([m['length_match'] for m in metrics_list])
    }


def _aggregate_prob_metrics(metrics_list: List[Dict], top_k: int = 3) -> Dict:
    """Aggregate probabilistic prediction metrics."""
    if not metrics_list:
        return {
            'top_k': top_k,
            'top_k_activity_match_rate': 0.0,
            'avg_length_original': 0.0,
            'avg_length_perturbed': 0.0,
            'var_length_original': 0.0,
            'var_length_perturbed': 0.0
        }

    top_k_value = metrics_list[0].get('top_k', top_k)

    return {
        'top_k': top_k_value,
        'top_k_activity_match_rate': np.mean([m.get('top_k_activity_match_rate', 0.0) for m in metrics_list]),
        'avg_length_original': np.mean([m.get('avg_length_original', 0.0) for m in metrics_list]),
        'avg_length_perturbed': np.mean([m.get('avg_length_perturbed', 0.0) for m in metrics_list]),
        'var_length_original': np.mean([m.get('var_length_original', 0.0) for m in metrics_list]),
        'var_length_perturbed': np.mean([m.get('var_length_perturbed', 0.0) for m in metrics_list])
    }


def _aggregate_by_prefix_length(
    metrics_list: List[Dict],
    concept_name: str = 'Activity',
    top_k: int = 3
) -> Dict:
    """Aggregate metrics by prefix length."""
    by_prefix = defaultdict(list)
    
    for metrics in metrics_list:
        prefix_len = metrics['prefix_len']
        by_prefix[prefix_len].append(metrics)
    
    result = {}
    for prefix_len, metrics_group in by_prefix.items():
        mean_metrics = [m['mean_prediction'] for m in metrics_group]
        prob_metrics = [m.get('probabilistic_prediction', {}) for m in metrics_group]
        
        # Filter out empty probabilistic predictions
        prob_metrics = [m for m in prob_metrics if m]
        
        result[prefix_len] = {
            'count': len(metrics_group),
            'mean_prediction': _aggregate_mean_metrics(mean_metrics),
            'probabilistic_prediction': _aggregate_prob_metrics(prob_metrics, top_k) if prob_metrics else {}
        }
    
    return result


def save_chunk(results, i, output_dir):
    """Helper function to save intermediate results."""
    chunk_number = (i + 1)
    filename = os.path.join(output_dir, f'robustness_results_part_{chunk_number:03d}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} results to {filename}")
