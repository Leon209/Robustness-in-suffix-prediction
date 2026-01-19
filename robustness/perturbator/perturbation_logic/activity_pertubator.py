import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Iterable, Literal

import pandas as pd
import pickle
from datetime import datetime
import numpy as np
import torch

from ml_models.event_log_loader.new_event_log_loader import CSV2EventLog
from perturbation_logic.structural_attacks import _update_day_and_seconds_features



@dataclass
class LoopConfig:
    loop_probability: float = 0.35
    min_segment_length: int = 2
    max_segment_length: int = 5
    min_loops_per_case: int = 1
    max_loops_per_case: int = 1
    timestamp_increment_seconds: float = 60.0


def build_readable_event_log(
    csv_path: str,
    properties_path: str,
    select_columns: Optional[Iterable[str]] = None,
    sort_by: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Use the stored event log properties + CSV2EventLog to build a readable dataframe
    that already contains engineered columns (case_elapsed_time, etc.) and EOS rows.
    """
    # with open(properties_path, "rb") as f:
    #     properties = pickle.load(f)

    # evlog = CSV2EventLog(
    #     event_log_dir=csv_path,
    #     timestamp_name=properties["timestamp_name"],
    #     case_name=properties["case_name"],
    #     categorical_columns=properties["categorical_columns"],
    #     continuous_columns=properties["continuous_columns"],
    #     continuous_positive_columns=properties.get("continuous_positive_columns", []),
    #     time_since_case_start_column=properties.get("time_since_case_start_column"),
    #     time_since_last_event_column=properties.get("time_since_last_event_column"),
    #     day_in_week_column=properties.get("day_in_week_column"),
    #     seconds_in_day_column=properties.get("seconds_in_day_column"),
    #     date_format=properties.get("date_format", "%Y-%m-%d %H:%M:%S.%f"),
    #     min_suffix_size=properties.get("min_suffix_size", 1),
    # )

    properties = {
    'case_name' : 'case:concept:name',
    'concept_name' : 'concept:name',
    'timestamp_name' : 'time:timestamp',
    'time_since_case_start_column' : 'case_elapsed_time',
    'time_since_last_event_column' : 'event_elapsed_time',
    'day_in_week_column' : 'day_in_week',
    'seconds_in_day_column' : 'seconds_in_day',
    'min_suffix_size' : 5,
    'train_validation_size' : 0.15,
    'test_validation_size' : 0.2,
    'window_size' : 'auto',
    'categorical_columns' : ['concept:name', 'InfectionSuspected', 'org:group', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'lifecycle:transition',
                             'Diagnose', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG'],
    'continuous_columns' : ['case_elapsed_time', 'event_elapsed_time', 'day_in_week', 'seconds_in_day',
                            'Age', 'Leucocytes', 'CRP', 'LacticAcid'],
    'continuous_positive_columns' : []
    }

    evlog = CSV2EventLog(
        event_log_dir=csv_path,
        timestamp_name=properties["timestamp_name"],
        case_name=properties["case_name"],
        categorical_columns=properties["categorical_columns"],
        continuous_columns=properties["continuous_columns"],
        continuous_positive_columns=properties.get("continuous_positive_columns", []),
        time_since_case_start_column=properties.get("time_since_case_start_column"),
        time_since_last_event_column=properties.get("time_since_last_event_column"),
        day_in_week_column=properties.get("day_in_week_column"),
        seconds_in_day_column=properties.get("seconds_in_day_column"),
        date_format=properties.get("date_format", "%Y-%m-%d %H:%M:%S.%f"),
        min_suffix_size=properties.get("min_suffix_size", 1),
    )

    df = evlog.df.copy()
    if sort_by:
        df = df.sort_values(sort_by).reset_index(drop=True)

    if select_columns:
        keep_cols = [properties["case_name"]]
        keep_cols += [col for col in select_columns if col != properties["case_name"]]
        keep_cols = [col for col in keep_cols if col in df.columns]
        df = df[keep_cols]

    return df, properties


def insert_realistic_loops(
    df: pd.DataFrame,
    properties: Dict[str, Any],
    config: Optional[LoopConfig] = None,
    eos_value: str = "EOS",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Duplicate realistic activity segments within cases to create loops.

    Args:
        df: Readable dataframe with EOS rows (output of build_readable_event_log).
        properties: Event log properties dict.
        config: LoopConfig describing probabilities/lengths.
        eos_value: Value identifying EOS rows.
        random_state: Optional RNG seed for reproducibility.
    """
    config = config or LoopConfig()
    rng = random.Random(random_state)

    case_col = properties["case_name"]
    activity_col = properties["concept_name"]

    timestamp_col = properties["timestamp_name"]
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    augmented_cases = []
    for case_id, group in df.groupby(case_col, sort=False):
        augmented_case = _insert_loops_into_case(
            group.reset_index(drop=True),
            properties=properties,
            config=config,
            rng=rng,
            activity_col=activity_col,
            eos_value=eos_value,
        )
        augmented_cases.append(augmented_case)

    return pd.concat(augmented_cases, ignore_index=True)


def split_prefix_suffix_readable(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    min_suffix_size: int,
    eos_value: str = "EOS",
) -> Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split a readable dataframe (with EOS rows) into prefix/suffix DataFrames.
    
    The minimum suffix size only counts non-EOS rows. The suffix includes:
    - At least min_suffix_size non-EOS rows
    - All EOS rows at the end of the case
    
    Args:
        df: DataFrame with event log data including EOS rows
        case_column: Name of the case ID column
        activity_column: Name of the activity column (used to identify EOS rows)
        min_suffix_size: Minimum number of non-EOS rows required in suffix
        eos_value: Value that identifies EOS rows (default: "EOS")
    
    Returns:
        Dictionary keyed by (case_id, prefix_len) -> (prefix_df, suffix_df).
        prefix_len counts only non-EOS rows.
    """
    result: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for case_id, group in df.groupby(case_column, sort=False):
        group = group.reset_index(drop=True)
        
        # Separate EOS and non-EOS rows
        non_eos_mask = group[activity_column] != eos_value
        non_eos_df = group[non_eos_mask].reset_index(drop=True)
        
        # Need at least min_suffix_size non-EOS rows
        if len(non_eos_df) < min_suffix_size:
            continue
        
        # Get the positions (indices in reset group) of non-EOS rows
        non_eos_positions = [i for i, is_non_eos in enumerate(non_eos_mask) if is_non_eos]
        
        # Generate all possible prefix/suffix splits
        # prefix_len counts only non-EOS rows
        max_prefix_len = len(non_eos_df) - min_suffix_size
        
        for prefix_len in range(1, max_prefix_len + 1):
            # Find the split point: after prefix_len non-EOS rows
            # The split should be after the prefix_len-th non-EOS row
            # non_eos_positions[prefix_len - 1] is the position of the last non-EOS row in prefix
            # We want to split after this row, so split_idx = position + 1
            split_idx = non_eos_positions[prefix_len - 1] + 1
            
            # Split: prefix is everything before split_idx, suffix is everything from split_idx
            prefix_df = group.iloc[:split_idx].copy()
            suffix_df = group.iloc[split_idx:].copy()
            
            # Verify suffix has at least min_suffix_size non-EOS rows
            suffix_non_eos_count = (suffix_df[activity_column] != eos_value).sum()
            if suffix_non_eos_count >= min_suffix_size:
                result[(str(case_id), prefix_len)] = (prefix_df, suffix_df)

    return result


###Start LOOP LEARNING AND MATCHING ###

def _detect_loops_in_sequence(
    activity_sequence: List[str],
) -> List[Tuple[int, int]]:
    """
    Detect all loops in an activity sequence.
    A loop is a subsequence where the first and last activities are the same.
    Single-activity repeats (e.g., B->B) are excluded.
    
    Args:
        activity_sequence: List of activity names
        
    Returns:
        List of (start_index, end_index) tuples for each loop found.
        end_index is inclusive.
    """
    loops = []
    n = len(activity_sequence)
    
    # Check all possible subsequences
    for start in range(n):
        start_activity = activity_sequence[start]
        # Look for matching activity from start+1 onwards
        for end in range(start + 1, n):
            if activity_sequence[end] == start_activity:
                # Found a loop: from start to end (inclusive)
                # Exclude single-activity loops (start == end-1 means only one activity in between)
                if end - start >= 2:  # At least 2 activities: start_act -> ... -> start_act
                    loops.append((start, end))
    
    return loops


def learn_realistic_loops(
    df: pd.DataFrame,
    properties: Dict[str, Any],
    activity_column: str = None,
    eos_value: str = "EOS",
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Learn all realistic loops from a dataset.
    Loops are sequences within the same case that start and end with the same activity.
    
    Args:
        df: DataFrame from build_readable_event_log (with all columns)
        properties: Event log properties dict
        activity_column: Name of activity column (defaults to properties["concept_name"])
        eos_value: Value identifying EOS rows
        
    Returns:
        List of tuples: [(starting_activity, loop_dataframe), ...]
        Each loop_dataframe contains all original columns for the loop events.
    """
    case_col = properties["case_name"]
    if activity_column is None:
        activity_column = properties["concept_name"]
    
    learned_loops = []
    
    for case_id, group in df.groupby(case_col, sort=False):
        group = group.reset_index(drop=True)
        
        # Separate EOS and non-EOS rows
        non_eos_mask = group[activity_column] != eos_value
        non_eos_df = group[non_eos_mask].reset_index(drop=True)
        
        if len(non_eos_df) < 2:
            continue
        
        # Extract activity sequence
        activity_sequence = non_eos_df[activity_column].tolist()
        
        # Detect all loops in this case
        loop_indices = _detect_loops_in_sequence(activity_sequence)
        
        # For each loop, extract the loop data
        for start_idx, end_idx in loop_indices:
            # Get the loop DataFrame (inclusive of end)
            loop_data = non_eos_df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
            starting_activity = activity_sequence[start_idx]  # First (and last) activity of loop
            
            learned_loops.append((starting_activity, loop_data))
    
    return learned_loops


def _get_time_deltas_from_loop(
    loop_df: pd.DataFrame,
    properties: Dict[str, Any],
) -> List[float]:
    """
    Extract time deltas between consecutive events in a loop DataFrame.
    
    Args:
        loop_df: DataFrame containing the loop events
        properties: Event log properties dict
        
    Returns:
        List of time deltas in seconds between consecutive events in the loop.
        First delta is 0.0 (no delta before first event).
        Returns deltas for transitions: [0.0, delta_1->2, delta_2->3, ..., delta_n-1->n]
    """
    event_elapsed_col = properties.get("time_since_last_event_column")
    timestamp_col = properties.get("timestamp_name")
    
    if len(loop_df) < 2:
        return [0.0] * len(loop_df)
    
    deltas = [0.0]  # First event has no previous delta
    
    if event_elapsed_col and event_elapsed_col in loop_df.columns:
        # Use existing time_since_last_event column
        elapsed_values = loop_df[event_elapsed_col].fillna(0.0).astype(float).tolist()
        # Skip first value (it's the delta from previous event, not relevant for loop)
        deltas.extend(elapsed_values[1:])
    elif timestamp_col and timestamp_col in loop_df.columns:
        # Compute from timestamps
        timestamps = pd.to_datetime(loop_df[timestamp_col], errors="coerce")
        for i in range(1, len(timestamps)):
            delta = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds()
            deltas.append(max(0.0, delta))  # Ensure non-negative
    else:
        # Default: assume 60 seconds between events
        deltas.extend([60.0] * (len(loop_df) - 1))
    
    return deltas


def _insert_loop_into_case(
    case_df: pd.DataFrame,
    loop_data: pd.DataFrame,
    loop_time_deltas: List[float],
    matching_activity: str,
    properties: Dict[str, Any],
    activity_column: str,
    eos_value: str = "EOS",
) -> Tuple[pd.DataFrame, bool, int]:
    """
    Insert a learned loop into a case DataFrame after a matching activity.
    
    Args:
        case_df: Case DataFrame (with all columns)
        loop_data: Loop DataFrame to insert (with all columns)
        loop_time_deltas: Time deltas between consecutive events in loop
        matching_activity: Activity to match and insert loop after
        properties: Event log properties dict
        activity_column: Name of activity column
        eos_value: Value identifying EOS rows
        
    Returns:
        Tuple of:
            - Updated case DataFrame with loop inserted and timestamps adjusted.
            - Boolean flag indicating whether insertion occurred.
            - Number of non-EOS events inserted.
    """
    timestamp_col = properties.get("timestamp_name")
    case_elapsed_col = properties.get("time_since_case_start_column")
    event_elapsed_col = properties.get("time_since_last_event_column")
    day_col = properties.get("day_in_week_column")
    seconds_col = properties.get("seconds_in_day_column")
    
    case_df = case_df.copy().reset_index(drop=True)
    
    # Separate EOS and non-EOS rows
    non_eos_mask = case_df[activity_column] != eos_value
    non_eos_df = case_df[non_eos_mask].reset_index(drop=True)
    eos_rows = case_df[~non_eos_mask].reset_index(drop=True)
    
    # Find first matching activity (excluding EOS)
    match_idx = None
    for idx, row in non_eos_df.iterrows():
        if row[activity_column] == matching_activity:
            match_idx = idx
            break
    
    if match_idx is None:
        # No match found, return original case
        return case_df, False, 0
    
    # Ensure timestamps are datetime
    if timestamp_col and timestamp_col in non_eos_df.columns:
        non_eos_df[timestamp_col] = pd.to_datetime(non_eos_df[timestamp_col], errors="coerce")
    
    # Get the timestamp of the matching activity (this will be kept)
    match_timestamp = None
    if timestamp_col and timestamp_col in non_eos_df.columns:
        match_timestamp = non_eos_df.iloc[match_idx][timestamp_col]
        if pd.isna(match_timestamp):
            match_timestamp = pd.Timestamp(datetime.utcnow())
    else:
        match_timestamp = pd.Timestamp(datetime.utcnow())
    
    # Split case into: before_match (includes matching activity), after_match
    before_match = non_eos_df.iloc[:match_idx + 1].copy()  # Includes matching activity
    after_match = non_eos_df.iloc[match_idx + 1:].copy() if match_idx + 1 < len(non_eos_df) else pd.DataFrame()
    
    # Create loop copy with adjusted timestamps
    # The loop includes the starting activity at both ends (e.g., B->C->F->B)
    # When inserting, we insert only the rest of the loop (C->F->B) after the matching activity
    # to avoid duplicating the starting activity
    loop_copy = loop_data.iloc[1:].copy().reset_index(drop=True)  # Skip first activity (already in sequence)

    if loop_copy.empty:
        return case_df, False, 0
    
    # Get time deltas for the rest of the loop (skip first delta which is 0)
    loop_rest_deltas = loop_time_deltas[1:] if len(loop_time_deltas) > 1 else [60.0] * (len(loop_copy))
    
    # Set timestamps for loop events starting from match_timestamp + first delta
    if timestamp_col and timestamp_col in loop_copy.columns and len(loop_copy) > 0:
        # First event of the loop rest starts after match_timestamp + first delta
        # loop_rest_deltas[0] is the delta from B to C (first event in loop_rest)
        current_ts = match_timestamp
        for i in range(len(loop_copy)):
            if i < len(loop_rest_deltas):
                delta_seconds = loop_rest_deltas[i]
            else:
                delta_seconds = loop_rest_deltas[-1] if loop_rest_deltas else 60.0
            current_ts = current_ts + pd.to_timedelta(delta_seconds, unit="s")
            loop_copy.iloc[i, loop_copy.columns.get_loc(timestamp_col)] = current_ts
    
    # Calculate total loop duration (sum of all deltas from loop_time_deltas)
    # This represents the total time taken for the full loop
    total_loop_duration = sum(loop_time_deltas[1:]) if len(loop_time_deltas) > 1 else (loop_time_deltas[0] if loop_time_deltas else 60.0)
    
    # Shift remaining events forward by total loop duration
    if not after_match.empty and timestamp_col and timestamp_col in after_match.columns:
        after_match[timestamp_col] = pd.to_datetime(after_match[timestamp_col], errors="coerce") + pd.to_timedelta(total_loop_duration, unit="s")
    
    # Combine: before_match + loop_copy + after_match + eos_rows
    result_parts = [before_match, loop_copy]
    if not after_match.empty:
        result_parts.append(after_match)
    augmented_non_eos = pd.concat(result_parts, ignore_index=True)
    
    # Recompute temporal columns for the entire case
    case_col = properties["case_name"]
    if case_col in augmented_non_eos.columns:
        case_id_value = augmented_non_eos[case_col].iloc[0] if len(augmented_non_eos) > 0 else None
    else:
        case_id_value = case_df[case_col].iloc[0] if case_col in case_df.columns else None
    
    # Combine with EOS rows
    if not eos_rows.empty:
        augmented_case = pd.concat([augmented_non_eos, eos_rows], ignore_index=True)
    else:
        augmented_case = augmented_non_eos
    
    if case_col in augmented_case.columns and case_id_value:
        augmented_case[case_col] = case_id_value
    
    # Recompute all temporal columns
    if timestamp_col and timestamp_col in augmented_case.columns:
        augmented_case[timestamp_col] = pd.to_datetime(augmented_case[timestamp_col], errors="coerce")
        
        base_ts = augmented_case[timestamp_col].iloc[0]
        if pd.isna(base_ts):
            base_ts = pd.Timestamp(datetime.utcnow())
        
        # Recompute timestamps from event_elapsed_time if available
        if event_elapsed_col and event_elapsed_col in augmented_case.columns:
            deltas = augmented_case[event_elapsed_col].fillna(60.0).astype(float).tolist()
            if deltas:
                deltas[0] = 0.0
            deltas = [max(0.0, d) if d > 0 else 60.0 for d in deltas]
        else:
            # Compute deltas from timestamps
            timestamps = augmented_case[timestamp_col]
            deltas = [0.0]
            for i in range(1, len(timestamps)):
                if pd.notna(timestamps.iloc[i]) and pd.notna(timestamps.iloc[i-1]):
                    delta = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds()
                    deltas.append(max(0.0, delta))
                else:
                    deltas.append(60.0)
        
        # Rebuild timestamps from deltas
        new_timestamps = [base_ts]
        for delta in deltas[1:]:
            new_timestamps.append(new_timestamps[-1] + pd.to_timedelta(delta, unit="s"))
        augmented_case[timestamp_col] = new_timestamps
        
        # Recompute case_elapsed_time
        if case_elapsed_col and case_elapsed_col in augmented_case.columns:
            cumulative = 0.0
            case_elapsed_values = [cumulative]
            for delta in deltas[1:]:
                cumulative += delta
                case_elapsed_values.append(cumulative)
            augmented_case[case_elapsed_col] = case_elapsed_values
        
        # Update event_elapsed_time
        if event_elapsed_col and event_elapsed_col in augmented_case.columns:
            augmented_case[event_elapsed_col] = deltas
        
        # Update day and seconds columns
        _update_day_and_seconds_features(augmented_case, timestamp_col, day_col, seconds_col)
    
    inserted_non_eos = int((loop_copy[activity_column] != eos_value).sum())
    if inserted_non_eos == 0:
        return case_df, False, 0

    return augmented_case.reset_index(drop=True), True, inserted_non_eos


def _split_case_by_prefix_length(
    case_df: pd.DataFrame,
    activity_column: str,
    eos_value: str,
    prefix_len_non_eos: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a case dataframe into prefix/suffix given a prefix length (counting non-EOS rows).
    """
    case_df = case_df.reset_index(drop=True)
    if prefix_len_non_eos <= 0:
        return case_df.iloc[0:0].copy(), case_df.copy()

    non_eos_mask = case_df[activity_column] != eos_value
    non_eos_indices = [i for i, is_non_eos in enumerate(non_eos_mask) if is_non_eos]

    if prefix_len_non_eos > len(non_eos_indices):
        return case_df.copy(), case_df.iloc[0:0].copy()

    split_idx = non_eos_indices[prefix_len_non_eos - 1] + 1
    prefix_df = case_df.iloc[:split_idx].copy()
    suffix_df = case_df.iloc[split_idx:].copy()
    return prefix_df, suffix_df


def match_loops_greedy(
    data_new: pd.DataFrame,
    learned_loops: List[Tuple[str, pd.DataFrame]],
    properties: Dict[str, Any],
    min_suffix_size: int,
    activity_column: str = None,
    eos_value: str = "EOS",
) -> Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Greedily match and insert learned loops into a new dataset, then split into prefixes/suffixes.
    
    Loops are inserted greedily: take the first loop, find first matching starting activity,
    insert it, then move to next loop. After insertion, split into prefixes/suffixes where:
    - Loop must be entirely in prefix
    - At least one more activity after loop completes
    - Suffix has at least min_suffix_size non-EOS activities
    
    Args:
        data_new: New dataset DataFrame (with all columns)
        learned_loops: List of (starting_activity, loop_dataframe) tuples from learn_realistic_loops
        properties: Event log properties dict
        min_suffix_size: Minimum number of non-EOS activities required in suffix
        activity_column: Name of activity column (defaults to properties["concept_name"])
        eos_value: Value identifying EOS rows
        
    Returns:
        Dictionary keyed by (case_id, prefix_len) -> (prefix_df, suffix_df).
        prefix_len counts only non-EOS rows.
    """
    case_col = properties["case_name"]
    if activity_column is None:
        activity_column = properties["concept_name"]

    # Precompute loop metadata (skip loops without sufficient length)
    prepared_loops: List[Tuple[str, pd.DataFrame, List[float]]] = []
    for starting_activity, loop_df in learned_loops:
        if loop_df is None or loop_df.empty or len(loop_df) < 2:
            continue
        loop_time_deltas = _get_time_deltas_from_loop(loop_df, properties)
        prepared_loops.append((starting_activity, loop_df, loop_time_deltas))

    if not prepared_loops:
        return {}

    # First split the dataset into prefix/suffix pairs
    base_pairs = split_prefix_suffix_readable(
        data_new,
        case_column=case_col,
        activity_column=activity_column,
        min_suffix_size=min_suffix_size,
        eos_value=eos_value,
    )

    result: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for (case_id, prefix_len), (prefix_df, suffix_df) in base_pairs.items():
        combined_case = pd.concat([prefix_df, suffix_df], ignore_index=True)
        prefix_matched = False

        for starting_activity, loop_data, loop_time_deltas in prepared_loops:
            if starting_activity not in prefix_df[activity_column].values:
                continue

            updated_case, inserted, inserted_non_eos = _insert_loop_into_case(
                combined_case,
                loop_data,
                loop_time_deltas,
                starting_activity,
                properties,
                activity_column,
                eos_value,
            )

            if not inserted or inserted_non_eos == 0:
                continue

            new_prefix_len = prefix_len + inserted_non_eos
            new_prefix_df, new_suffix_df = _split_case_by_prefix_length(
                updated_case,
                activity_column=activity_column,
                eos_value=eos_value,
                prefix_len_non_eos=new_prefix_len,
            )

            suffix_non_eos_count = (new_suffix_df[activity_column] != eos_value).sum()
            if suffix_non_eos_count < min_suffix_size:
                continue

            result[(str(case_id), new_prefix_len)] = (new_prefix_df, new_suffix_df)
            prefix_matched = True
            break  # Greedy: stop after first successful loop insertion

        # Prefixes without matches are skipped automatically

    return result


# Re-export for backward compatibility
from perturbation_logic.structural_attacks import redo_last_activity_of_prefix


###Start LOOP AUGMENTATION ###

def _insert_loops_into_case(
    case_df: pd.DataFrame,
    properties: Dict[str, Any],
    config: LoopConfig,
    rng: random.Random,
    activity_col: str,
    eos_value: str,
) -> pd.DataFrame:
    timestamp_col = properties["timestamp_name"]
    case_col = properties["case_name"]

    non_eos = case_df[case_df[activity_col] != eos_value].reset_index(drop=True)
    eos_rows = case_df[case_df[activity_col] == eos_value].reset_index(drop=True)

    if non_eos.empty or rng.random() > config.loop_probability:
        return case_df

    seg_len_max = min(config.max_segment_length, len(non_eos))
    seg_len_min = min(config.min_segment_length, seg_len_max)

    if seg_len_min <= 0 or seg_len_max <= 0:
        return case_df

    loop_count = rng.randint(config.min_loops_per_case, config.max_loops_per_case)
    segments: List[Tuple[int, int]] = []

    for _ in range(loop_count):
        seg_len = rng.randint(seg_len_min, seg_len_max)
        max_start = len(non_eos) - seg_len
        if max_start <= 0:
            break
        start_idx = rng.randint(0, max_start)
        segments.append((start_idx, seg_len))

    if not segments:
        return case_df

    events = non_eos.to_dict("records")
    for start, seg_len in sorted(segments, key=lambda x: x[0], reverse=True):
        segment_copy = [row.copy() for row in events[start : start + seg_len]]
        events[start + seg_len : start + seg_len] = segment_copy

    augmented_non_eos = pd.DataFrame(events)
    augmented_case = pd.concat([augmented_non_eos, eos_rows], ignore_index=True)
    augmented_case[case_col] = case_df[case_col].iloc[0]

    return _recompute_case_temporal_columns(
        augmented_case,
        properties=properties,
        timestamp_col=timestamp_col,
        config=config,
    )


def _recompute_case_temporal_columns(
    case_df: pd.DataFrame,
    properties: Dict[str, Any],
    timestamp_col: str,
    config: LoopConfig,
) -> pd.DataFrame:
    """
    Ensure temporal columns stay consistent after loop insertion.
    """
    case_df = case_df.reset_index(drop=True).copy()
    case_df[timestamp_col] = pd.to_datetime(case_df[timestamp_col], errors="coerce")

    base_ts = case_df[timestamp_col].iloc[0]
    if pd.isna(base_ts):
        base_ts = pd.Timestamp(datetime.utcnow())

    event_elapsed_col = properties.get("time_since_last_event_column")
    case_elapsed_col = properties.get("time_since_case_start_column")
    day_col = properties.get("day_in_week_column")
    seconds_col = properties.get("seconds_in_day_column")

    default_delta = config.timestamp_increment_seconds
    deltas: List[float] = []

    if event_elapsed_col and event_elapsed_col in case_df.columns:
        deltas = case_df[event_elapsed_col].fillna(default_delta).astype(float).tolist()
        if deltas:
            deltas[0] = 0.0
        deltas = [delta if delta > 0 else default_delta for delta in deltas]
    else:
        deltas = [0.0] + [default_delta] * (len(case_df) - 1)

    new_timestamps = [base_ts]
    for delta in deltas[1:]:
        new_timestamps.append(new_timestamps[-1] + pd.to_timedelta(delta, unit="s"))
    case_df[timestamp_col] = new_timestamps

    if case_elapsed_col:
        cumulative = 0.0
        case_elapsed_values = []
        for delta in deltas:
            case_elapsed_values.append(cumulative)
            cumulative += delta
        case_df[case_elapsed_col] = case_elapsed_values

    if event_elapsed_col:
        case_df[event_elapsed_col] = deltas

    if day_col:
        case_df[day_col] = case_df[timestamp_col].dt.weekday

    if seconds_col:
        ts = case_df[timestamp_col]
        case_df[seconds_col] = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second

    return case_df


def get_train_val_test(
    csv_path: str,
    train_size: float | int = 0.7,
    val_size: float | int = 0.15,
    test_size: float | int = 0.15,
    output_dir: Optional[str] = None,
    file_prefix: str = "helpdesk_",
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    """
    Deterministically split the rows of a CSV into train/val/test subsets.
    The split preserves the original row order (no randomness).
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path).reset_index(drop=True)
    total_rows = len(df)
    assert total_rows > 0, "No rows found in dataset"

    def interpret_size(value: float | int, available: int) -> int:
        if value <= 1.0:
            return min(available, int(round(available * value)))
        return min(available, int(value))

    train_count = interpret_size(train_size, total_rows)
    remaining_after_train = total_rows - train_count
    val_count = interpret_size(val_size, remaining_after_train)
    remaining_after_val = remaining_after_train - val_count
    test_count = interpret_size(test_size, remaining_after_val)

    assigned = train_count + val_count + test_count
    if assigned < total_rows:
        test_count += total_rows - assigned

    splits = {
        "train": df.iloc[:train_count].reset_index(drop=True),
        "val": df.iloc[train_count:train_count + val_count].reset_index(drop=True),
        "test": df.iloc[train_count + val_count:train_count + val_count + test_count].reset_index(drop=True),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for split_name, split_df in splits.items():
            path = os.path.join(output_dir, f"{file_prefix}{split_name}.csv")
            split_df.to_csv(path, index=False)

    return splits


def encode_single_dataframe(df, encoder_decoder, case_name_col, case_id_value):
    """
    Encode a single DataFrame (prefix or suffix) into tensor format.
    
    Args:
        df: DataFrame to encode (prefix or suffix)
        encoder_decoder: TensorEncoderDecoder instance
        case_name_col: Name of the case column
        case_id_value: Value for the case ID (to ensure proper grouping)
    
    Returns:
        Tuple of (categorical_tensors, numerical_tensors) where each is a list
        of tensors with shape (1, window_size)
    """
    # Ensure the DataFrame has the case_name column for proper encoding
    df_copy = df.copy()
    if case_name_col not in df_copy.columns:
        df_copy[case_name_col] = case_id_value
    else:
        # Ensure all rows have the same case_id
        df_copy[case_name_col] = case_id_value
    
    # Encode categorical columns
    cat_tensors = []
    for col in encoder_decoder.categorical_columns:
        if col not in df_copy.columns:
            # Create zero tensor if column missing
            cat_tensors.append(torch.zeros((1, encoder_decoder.window_size), dtype=torch.long))
            continue
            
        # Get values for this column
        case_values = np.array(df_copy[[col]], dtype=object)
        # Transform using the fitted encoder
        case_values_enc = encoder_decoder.categorical_encoders[col].transform(case_values) + 1
        
        # Pad to window_size
        padded = encoder_decoder.pad_to_window_size(case_values_enc)
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(padded, dtype=torch.long).squeeze(-1)  # shape: (window_size,)
        tensor = tensor.unsqueeze(0)  # shape: (1, window_size)
        cat_tensors.append(tensor)
    
    # Encode continuous columns
    num_tensors = []
    for col in encoder_decoder.continuous_columns + encoder_decoder.continuous_positive_columns:
        if col not in df_copy.columns:
            # Create zero tensor if column missing
            num_tensors.append(torch.zeros((1, encoder_decoder.window_size), dtype=torch.float32))
            continue
            
        # Get values for this column
        case_values = df_copy[[col]].values  # shape (n, 1)
        # Impute and transform
        case_values_imputed = encoder_decoder.continuous_imputers[col].transform(case_values)
        case_values_enc = encoder_decoder.continuous_encoders[col].transform(case_values_imputed)
        
        # Pad to window_size
        padded = encoder_decoder.pad_to_window_size(case_values_enc)
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(padded, dtype=torch.float32).squeeze(-1)  # shape: (window_size,)
        tensor = tensor.unsqueeze(0)  # shape: (1, window_size)
        num_tensors.append(tensor)
    
    return (cat_tensors, num_tensors)