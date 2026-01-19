"""
Adversarial Sample Selector Module

This module provides functionality to identify and analyze the most effective
adversarial samples based on prediction shift metrics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def get_best_adversarial_prefixes(
    dataset: Dict[Tuple[str, int], Dict[str, Any]],
    top_n: int = 20,
    concept_name: str = 'Activity'
) -> List[Tuple]:
    """
    Identify and return the most effective adversarial prefix-suffix pairs.
    
    This function sorts the dataset by prediction_shift_dls (ascending) to find
    the adversarial samples that caused the biggest shift in model predictions.
    Lower DLS values indicate larger prediction shifts (more adversarial effect).
    
    Args:
        dataset: Dictionary mapping (case_name, prefix_len) to result entries.
                 Expected to have been processed by prepare_robustness_results().
        top_n: Number of top adversarial samples to return (default: 20)
        concept_name: Name of the concept to analyze (default: 'Activity')
    
    Returns:
        List of tuples, each containing:
            - case_name (str): Case identifier
            - prefix_len (int): Length of the prefix
            - prediction_shift_dls (float): DLS between clean and perturbed predictions
            - prefix_orig (List[Dict]): Original prefix events
            - prefix_pert (List[Dict]): Perturbed prefix events
            - mean_orig (List[Dict]): Original (clean) prediction
            - mean_pert (List[Dict]): Perturbed prediction
            - perturbations (List[Dict]): Identified perturbations in the prefix
    """
    # Extract all entries with prediction_shift_dls
    adversarial_samples = []
    
    for (case_name, prefix_len), entry in dataset.items():
        prediction_shift_dls = entry.get('prediction_shift_dls')
        
        # Skip entries without the required metric
        if prediction_shift_dls is None:
            continue
        
        # Extract prefix and prediction data
        prefix_orig = None
        prefix_pert = None
        mean_orig = None
        mean_pert = None
        
        if 'original' in entry:
            try:
                prefix_orig, _, mean_orig, _ = entry['original']
            except (ValueError, TypeError):
                continue
        
        if 'perturbed' in entry:
            try:
                prefix_pert, _, mean_pert, _ = entry['perturbed']
            except (ValueError, TypeError):
                continue
        
        # Skip if we don't have complete data
        if prefix_orig is None or prefix_pert is None or mean_orig is None or mean_pert is None:
            continue
        
        # Identify perturbations in the prefix
        perturbations = _identify_perturbations(prefix_orig, prefix_pert, concept_name)
        
        adversarial_samples.append({
            'case_name': case_name,
            'prefix_len': prefix_len,
            'prediction_shift_dls': prediction_shift_dls,
            'prefix_orig': prefix_orig,
            'prefix_pert': prefix_pert,
            'mean_orig': mean_orig,
            'mean_pert': mean_pert,
            'perturbations': perturbations
        })
    
    # Sort by prediction_shift_dls ascending (lower = more adversarial)
    adversarial_samples.sort(key=lambda x: x['prediction_shift_dls'])
    
    # Take top N
    top_samples = adversarial_samples[:top_n]
    
    # Print formatted results
    _print_adversarial_analysis(top_samples, concept_name)
    
    # Return as list of tuples for easy unpacking
    result = [
        (
            sample['case_name'],
            sample['prefix_len'],
            sample['prediction_shift_dls'],
            sample['prefix_orig'],
            sample['prefix_pert'],
            sample['mean_orig'],
            sample['mean_pert'],
            sample['perturbations']
        )
        for sample in top_samples
    ]
    
    return result


def _identify_perturbations(
    prefix_orig: List[Dict],
    prefix_pert: List[Dict],
    concept_name: str = 'Activity'
) -> List[Dict]:
    """
    Identify what was perturbed in the prefix.
    
    Compares the original and perturbed prefixes to find differences
    in activities, attributes, and other event properties.
    
    Args:
        prefix_orig: Original prefix events
        prefix_pert: Perturbed prefix events
        concept_name: Name of the activity concept
    
    Returns:
        List of perturbation descriptions, each as a dict with:
            - event_index: Index of the perturbed event
            - changes: List of specific changes made
    """
    perturbations = []
    
    # Handle length differences
    min_len = min(len(prefix_orig), len(prefix_pert))
    max_len = max(len(prefix_orig), len(prefix_pert))
    
    if len(prefix_orig) != len(prefix_pert):
        perturbations.append({
            'event_index': 'Length',
            'changes': [f"Length changed: {len(prefix_orig)} → {len(prefix_pert)}"]
        })
    
    # Compare each event
    for i in range(min_len):
        event_orig = prefix_orig[i]
        event_pert = prefix_pert[i]
        changes = []
        
        # Get all keys from both events
        all_keys = set(event_orig.keys()) | set(event_pert.keys())
        
        for key in all_keys:
            val_orig = event_orig.get(key)
            val_pert = event_pert.get(key)
            
            # Check if values differ
            if val_orig != val_pert:
                # Format the change description
                if key == concept_name:
                    changes.append(f"{key}: '{val_orig}' → '{val_pert}'")
                elif isinstance(val_orig, (int, float)) and isinstance(val_pert, (int, float)):
                    # For numeric values, show the change
                    changes.append(f"{key}: {val_orig:.2f} → {val_pert:.2f}")
                else:
                    changes.append(f"{key}: {val_orig} → {val_pert}")
        
        if changes:
            perturbations.append({
                'event_index': i,
                'changes': changes
            })
    
    # Handle extra events (if lengths differ)
    if len(prefix_pert) > len(prefix_orig):
        for i in range(min_len, len(prefix_pert)):
            perturbations.append({
                'event_index': i,
                'changes': [f"Event added: {prefix_pert[i].get(concept_name, 'Unknown')}"]
            })
    elif len(prefix_orig) > len(prefix_pert):
        for i in range(min_len, len(prefix_orig)):
            perturbations.append({
                'event_index': i,
                'changes': [f"Event removed: {prefix_orig[i].get(concept_name, 'Unknown')}"]
            })
    
    return perturbations


def _extract_activity_sequence(events: List[Dict], concept_name: str = 'Activity') -> List[str]:
    """
    Extract activity sequence from events.
    
    Args:
        events: List of event dictionaries
        concept_name: Name of the activity concept
    
    Returns:
        List of activity names
    """
    return [event.get(concept_name, 'Unknown') for event in events]


def _print_adversarial_analysis(samples: List[Dict], concept_name: str = 'Activity') -> None:
    """
    Print formatted analysis of adversarial samples.
    
    Args:
        samples: List of adversarial sample dictionaries
        concept_name: Name of the activity concept
    """
    if not samples:
        print("No adversarial samples found.")
        return
    
    print("\n" + "=" * 80)
    print(f"Top {len(samples)} Most Effective Adversarial Samples")
    print("=" * 80)
    print(f"\nLower DLS = Bigger prediction shift = More adversarial effect")
    print("=" * 80 + "\n")
    
    for rank, sample in enumerate(samples, 1):
        case_name = sample['case_name']
        prefix_len = sample['prefix_len']
        dls = sample['prediction_shift_dls']
        perturbations = sample['perturbations']
        
        # Extract activity sequences
        clean_pred = _extract_activity_sequence(sample['mean_orig'], concept_name)
        pert_pred = _extract_activity_sequence(sample['mean_pert'], concept_name)
        
        print(f"#{rank}. Case: {case_name}, Prefix Length: {prefix_len}")
        print(f"    Prediction Shift DLS: {dls:.4f}")
        print(f"")
        
        # Print perturbations
        if perturbations:
            print(f"    Perturbations Applied:")
            for pert in perturbations:
                event_idx = pert['event_index']
                changes = pert['changes']
                if event_idx == 'Length':
                    print(f"      • {changes[0]}")
                else:
                    print(f"      • Event {event_idx}:")
                    for change in changes:
                        print(f"          - {change}")
        else:
            print(f"    Perturbations Applied: None detected")
        
        print(f"")
        
        # Print predictions
        print(f"    Clean Prediction:     [{', '.join(clean_pred[:5])}" + 
              (f", ... (+{len(clean_pred)-5} more)" if len(clean_pred) > 5 else "") + "]")
        print(f"    Perturbed Prediction: [{', '.join(pert_pred[:5])}" + 
              (f", ... (+{len(pert_pred)-5} more)" if len(pert_pred) > 5 else "") + "]")
        
        print("\n" + "-" * 80 + "\n")
    
    # Summary statistics
    dls_values = [s['prediction_shift_dls'] for s in samples]
    print(f"Summary Statistics:")
    print(f"  Mean DLS: {np.mean(dls_values):.4f}")
    print(f"  Median DLS: {np.median(dls_values):.4f}")
    print(f"  Min DLS: {np.min(dls_values):.4f}")
    print(f"  Max DLS: {np.max(dls_values):.4f}")
    print("\n" + "=" * 80 + "\n")
