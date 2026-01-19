import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional


def _compare_prefixes(
    clean_prefix: pd.DataFrame,
    perturbed_prefix: pd.DataFrame,
    properties: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Compare two prefix dataframes and return differences.
    
    Args:
        clean_prefix: Clean prefix DataFrame
        perturbed_prefix: Perturbed prefix DataFrame
        properties: Properties dictionary for filtering time columns
    
    Returns:
        List of dictionaries, each containing:
        - 'event_idx': Index of the event (row) with differences
        - 'changes': Dictionary mapping column names to (old_value, new_value) tuples
    """
    differences = []
    
    # Get all columns from both dataframes
    all_columns = set(clean_prefix.columns) | set(perturbed_prefix.columns)
    
    # Filter out time-related columns for comparison
    time_columns = [
        properties.get('timestamp_name'),
        properties.get('time_since_case_start_column'),
        properties.get('time_since_last_event_column'),
        properties.get('day_in_week_column'),
        properties.get('seconds_in_day_column'),
    ]
    time_columns = [col for col in time_columns if col is not None]
    
    # Get columns to compare (all non-time columns)
    columns_to_compare = [col for col in all_columns if col not in time_columns]
    
    # Determine the minimum length to compare
    min_len = min(len(clean_prefix), len(perturbed_prefix))
    
    # Compare row by row
    for row_idx in range(min_len):
        event_changes = {}
        
        for col in columns_to_compare:
            clean_val = clean_prefix.iloc[row_idx][col] if col in clean_prefix.columns else None
            perturbed_val = perturbed_prefix.iloc[row_idx][col] if col in perturbed_prefix.columns else None
            
            # Handle NaN/None values
            clean_is_na = pd.isna(clean_val) if clean_val is not None else True
            perturbed_is_na = pd.isna(perturbed_val) if perturbed_val is not None else True
            
            # Check if values are different
            if clean_is_na and perturbed_is_na:
                continue  # Both are NaN, no difference
            elif clean_is_na != perturbed_is_na:
                # One is NaN, other is not
                event_changes[col] = (clean_val, perturbed_val)
            else:
                # Both are not NaN, compare values
                if clean_val != perturbed_val:
                    # For numeric values, check if they're close (to handle floating point issues)
                    if isinstance(clean_val, (int, float)) and isinstance(perturbed_val, (int, float)):
                        if not np.isclose(clean_val, perturbed_val, rtol=1e-9, atol=1e-9):
                            event_changes[col] = (clean_val, perturbed_val)
                    else:
                        event_changes[col] = (clean_val, perturbed_val)
        
        if event_changes:
            differences.append({
                'event_idx': row_idx,
                'changes': event_changes
            })
    
    # Check if one prefix is longer than the other
    if len(clean_prefix) != len(perturbed_prefix):
        max_len = max(len(clean_prefix), len(perturbed_prefix))
        for row_idx in range(min_len, max_len):
            differences.append({
                'event_idx': row_idx,
                'changes': {'_prefix_length_mismatch': (
                    f"Clean has {len(clean_prefix)} events",
                    f"Perturbed has {len(perturbed_prefix)} events"
                )}
            })
    
    return differences


def _format_comparison_output(
    case_id: str,
    prefix_len: int,
    differences: List[Dict[str, Any]]
) -> str:
    """
    Format the output for a single case comparison.
    
    Args:
        case_id: Case identifier
        prefix_len: Prefix length
        differences: List of difference dictionaries from _compare_prefixes
    
    Returns:
        Formatted string for this case
    """
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Case: {case_id}, Prefix Length: {prefix_len}")
    output_lines.append(f"{'='*80}")
    
    if not differences:
        output_lines.append("  ✓ No changes detected")
    else:
        output_lines.append(f"  [CHANGED] Found {len(differences)} event(s) with differences")
        output_lines.append("")
        
        for diff in differences:
            event_idx = diff['event_idx']
            changes = diff['changes']
            
            # Check if this is a length mismatch
            if '_prefix_length_mismatch' in changes:
                old_val, new_val = changes['_prefix_length_mismatch']
                output_lines.append(f"  Event {event_idx}: {old_val} → {new_val}")
                continue
            
            output_lines.append(f"  Event {event_idx}:")
            for col, (old_val, new_val) in changes.items():
                # Format values for display
                old_str = str(old_val) if old_val is not None else "None/NaN"
                new_str = str(new_val) if new_val is not None else "None/NaN"
                
                # Truncate long values
                max_len = 50
                if len(old_str) > max_len:
                    old_str = old_str[:max_len] + "..."
                if len(new_str) > max_len:
                    new_str = new_str[:max_len] + "..."
                
                output_lines.append(f"    {col}:")
                output_lines.append(f"      Clean:    {old_str}")
                output_lines.append(f"      Perturbed: {new_str} [CHANGED]")
            output_lines.append("")
    
    return "\n".join(output_lines)


def _print_summary(
    total_cases: int,
    cases_with_changes: int,
    cases_without_changes: int
) -> None:
    """
    Print summary statistics.
    
    Args:
        total_cases: Total number of cases compared
        cases_with_changes: Number of cases with differences
        cases_without_changes: Number of cases without differences
    """
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cases compared: {total_cases}")
    print(f"Cases with changes: {cases_with_changes} ({100*cases_with_changes/max(total_cases,1):.1f}%)")
    print(f"Cases without changes: {cases_without_changes} ({100*cases_without_changes/max(total_cases,1):.1f}%)")
    print("="*80)


def highlight_feature_attack_impact(
    clean_data_path: str,
    perturbed_data_path: str,
    properties: Dict[str, Any]
) -> None:
    """
    Compare clean and perturbed test datasets, identify differences in prefixes,
    and print a formatted comparison showing all cases.
    
    Args:
        clean_data_path: String path to clean test dataset pickle file
        perturbed_data_path: String path to perturbed test dataset pickle file
        properties: Properties dictionary (for column names, activity column, etc.)
    """
    # Load both datasets
    print(f"Loading clean dataset from: {clean_data_path}")
    clean_data = torch.load(clean_data_path, weights_only=False)
    
    print(f"Loading perturbed dataset from: {perturbed_data_path}")
    perturbed_data = torch.load(perturbed_data_path, weights_only=False)
    
    print(f"\nClean dataset has {len(clean_data)} cases")
    print(f"Perturbed dataset has {len(perturbed_data)} cases")
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Get all keys (union of both datasets)
    all_keys = set(clean_data.keys()) | set(perturbed_data.keys())
    
    total_cases = len(all_keys)
    cases_with_changes = 0
    cases_without_changes = 0
    
    # Iterate through all keys
    for key in sorted(all_keys):
        case_id, prefix_len = key
        
        # Check if key exists in both datasets
        if key not in clean_data:
            print(f"\n{'='*80}")
            print(f"Case: {case_id}, Prefix Length: {prefix_len}")
            print(f"{'='*80}")
            print("  ⚠ WARNING: Key exists only in perturbed dataset")
            cases_with_changes += 1
            continue
        
        if key not in perturbed_data:
            print(f"\n{'='*80}")
            print(f"Case: {case_id}, Prefix Length: {prefix_len}")
            print(f"{'='*80}")
            print("  ⚠ WARNING: Key exists only in clean dataset")
            cases_with_changes += 1
            continue
        
        # Extract prefix dataframes
        clean_prefix, _ = clean_data[key]
        perturbed_prefix, _ = perturbed_data[key]
        
        # Compare prefixes
        differences = _compare_prefixes(clean_prefix, perturbed_prefix, properties)
        
        # Format and print output
        output = _format_comparison_output(case_id, prefix_len, differences)
        print(output)
        
        # Update statistics
        if differences:
            cases_with_changes += 1
        else:
            cases_without_changes += 1
    
    # Print summary
    _print_summary(total_cases, cases_with_changes, cases_without_changes)


def _format_structural_attack_output(
    case_id: str,
    prefix_len: int,
    clean_prefix: pd.DataFrame,
    perturbed_prefix: pd.DataFrame,
    clean_suffix: pd.DataFrame,
    perturbed_suffix: pd.DataFrame,
    properties: Dict[str, Any]
) -> str:
    """
    Format the output for a single case structural attack comparison.
    
    Args:
        case_id: Case identifier
        prefix_len: Prefix length
        clean_prefix: Clean prefix DataFrame
        perturbed_prefix: Perturbed prefix DataFrame
        clean_suffix: Clean suffix DataFrame
        perturbed_suffix: Perturbed suffix DataFrame
        properties: Properties dictionary for column names
    
    Returns:
        Formatted string for this case
    """
    output_lines = []
    output_lines.append(f"\n{'='*80}")
    output_lines.append(f"Case: {case_id}, Prefix Length: {prefix_len}")
    output_lines.append(f"{'='*70}")
    
    # Get activity column name
    activity_column = properties.get('concept_name', 'Activity')
    
    # Get time column names
    time_columns = {
        'time_since_case_start': properties.get('time_since_case_start_column'),
        'time_since_last_event': properties.get('time_since_last_event_column'),
        'day_in_week': properties.get('day_in_week_column'),
        'seconds_in_day': properties.get('seconds_in_day_column'),
    }
    # Filter out None values
    time_columns = {k: v for k, v in time_columns.items() if v is not None}
    
    # Extract activity sequences
    def get_activity_sequence(df):
        if df is None or len(df) == 0:
            return []
        if activity_column in df.columns:
            return df[activity_column].tolist()
        return []
    
    # Extract time attributes
    def get_time_attributes(df):
        if df is None or len(df) == 0:
            return {}
        time_attrs = {}
        for attr_name, col_name in time_columns.items():
            if col_name in df.columns:
                time_attrs[attr_name] = df[col_name].tolist()
        return time_attrs
    
    # Prefix section
    output_lines.append("Prefix")
    clean_prefix_activities = get_activity_sequence(clean_prefix)
    perturbed_prefix_activities = get_activity_sequence(perturbed_prefix)
    output_lines.append(f"Activity_seq_clean = {clean_prefix_activities}")
    output_lines.append(f"Activity_seq_pert = {perturbed_prefix_activities}")
    
    clean_prefix_times = get_time_attributes(clean_prefix)
    perturbed_prefix_times = get_time_attributes(perturbed_prefix)
    
    # Print time attributes
    for attr_name, col_name in time_columns.items():
        clean_vals = clean_prefix_times.get(attr_name, [])
        perturbed_vals = perturbed_prefix_times.get(attr_name, [])
        # Capitalize first letter for display
        if col_name and len(col_name) > 0:
            display_name = col_name[0].upper() + col_name[1:]
        else:
            display_name = attr_name if attr_name else col_name
        output_lines.append(f"{display_name}_clean = {clean_vals}")
        output_lines.append(f"{display_name}_pert = {perturbed_vals}")
    
    # Suffix section
    output_lines.append("")
    output_lines.append("===")
    output_lines.append("suffix")
    clean_suffix_activities = get_activity_sequence(clean_suffix)
    perturbed_suffix_activities = get_activity_sequence(perturbed_suffix)
    output_lines.append(f"Activity_seq_clean = {clean_suffix_activities}")
    output_lines.append(f"Activity_seq_pert = {perturbed_suffix_activities}")
    
    clean_suffix_times = get_time_attributes(clean_suffix)
    perturbed_suffix_times = get_time_attributes(perturbed_suffix)
    
    # Print time attributes
    for attr_name, col_name in time_columns.items():
        clean_vals = clean_suffix_times.get(attr_name, [])
        perturbed_vals = perturbed_suffix_times.get(attr_name, [])
        # Capitalize first letter for display
        if col_name and len(col_name) > 0:
            display_name = col_name[0].upper() + col_name[1:]
        else:
            display_name = attr_name if attr_name else col_name
        output_lines.append(f"{display_name}_clean = {clean_vals}")
        output_lines.append(f"{display_name}_pert = {perturbed_vals}")
    
    return "\n".join(output_lines)


def highlight_structural_attack_impact(
    clean_data_path: str,
    perturbed_data_path: str,
    properties: Dict[str, Any]
) -> None:
    """
    Compare clean and perturbed test datasets for structural attacks,
    focusing on activity sequences and time attributes for both prefix and suffix.
    
    Args:
        clean_data_path: String path to clean test dataset pickle file
        perturbed_data_path: String path to perturbed test dataset pickle file
        properties: Properties dictionary (for column names, activity column, etc.)
    """
    # Load both datasets
    print(f"Loading clean dataset from: {clean_data_path}")
    clean_data = torch.load(clean_data_path, weights_only=False)
    
    print(f"Loading perturbed dataset from: {perturbed_data_path}")
    perturbed_data = torch.load(perturbed_data_path, weights_only=False)
    
    print(f"\nClean dataset has {len(clean_data)} cases")
    print(f"Perturbed dataset has {len(perturbed_data)} cases")
    print("\n" + "="*80)
    print("STRUCTURAL ATTACK COMPARISON RESULTS")
    print("="*80)
    
    # Get all keys (union of both datasets)
    all_keys = set(clean_data.keys()) | set(perturbed_data.keys())
    
    total_cases = len(all_keys)
    
    # Iterate through all keys
    for key in sorted(all_keys):
        case_id, prefix_len = key
        
        # Check if key exists in both datasets
        if key not in clean_data:
            print(f"\n{'='*80}")
            print(f"Case: {case_id}, Prefix Length: {prefix_len}")
            print(f"{'='*70}")
            print("  ⚠ WARNING: Key exists only in perturbed dataset")
            continue
        
        if key not in perturbed_data:
            print(f"\n{'='*80}")
            print(f"Case: {case_id}, Prefix Length: {prefix_len}")
            print(f"{'='*70}")
            print("  ⚠ WARNING: Key exists only in clean dataset")
            continue
        
        # Extract prefix and suffix dataframes
        clean_prefix, clean_suffix = clean_data[key]
        perturbed_prefix, perturbed_suffix = perturbed_data[key]
        
        # Format and print output
        output = _format_structural_attack_output(
            case_id, prefix_len,
            clean_prefix, perturbed_prefix,
            clean_suffix, perturbed_suffix,
            properties
        )
        print(output)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total cases compared: {total_cases}")
    print("="*80)
