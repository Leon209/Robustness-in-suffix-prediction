import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


def _update_day_and_seconds_features(
    df: pd.DataFrame,
    timestamp_col: Optional[str],
    day_col: Optional[str],
    seconds_col: Optional[str],
) -> None:
    """Update day-of-week and seconds-in-day columns based on timestamps."""
    if timestamp_col and timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if day_col and day_col in df.columns:
            df[day_col] = ts.dt.weekday
        if seconds_col and seconds_col in df.columns:
            df[seconds_col] = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second


def _convert_numpy_to_python_types(event: pd.Series) -> pd.Series:
    """
    Convert numpy numeric types in a Series to Python native types.
    This ensures consistency and prevents issues with np.float32 vs float.
    Also normalizes all NaN values (including np.float32(nan)) to standard np.nan.
    
    Args:
        event: Series with potentially numpy numeric types
        
    Returns:
        Series with Python native numeric types and normalized NaN values
    """
    for col in event.index:
        val = event[col]
        if pd.isna(val):
            # Normalize all NaN types (np.float32(nan), np.float64(nan), etc.) 
            # to standard np.nan for consistency
            event[col] = np.nan
        else:
            # Convert numpy scalar types to Python native types
            # np.generic is the base class for all numpy scalars (float32, float64, int32, etc.)
            if isinstance(val, np.generic):
                event[col] = val.item()
    return event


def _calculate_time_features(
    current_event: pd.Series,
    previous_event: pd.Series,
    properties: Dict[str, Any],
) -> pd.Series:
    """
    Calculate time features for an event based on the previous event.
    
    Skips time calculation if the current event is an EOS row (concept_name == 'EOS')
    or if time-related values are NaN.
    
    Args:
        current_event: Series representing the current event (may be modified in place).
        previous_event: Series representing the previous event (reference point).
        properties: Event log properties dict containing column names.
    
    Returns:
        Series with updated time features (or unchanged if EOS/NaN).
    """
    # Check if current event is an EOS row
    concept_name_col = properties.get("concept_name")
    is_eos = False
    if concept_name_col and concept_name_col in current_event.index:
        is_eos = current_event[concept_name_col] == 'EOS'
    
    # Check if event_elapsed_col is NaN (indicates EOS or invalid time data)
    event_elapsed_col = properties.get("time_since_last_event_column")
    has_nan_time = False
    if event_elapsed_col and event_elapsed_col in current_event.index:
        has_nan_time = pd.isna(current_event[event_elapsed_col])
    
    # Skip time calculation for EOS rows or when time values are NaN
    if is_eos or has_nan_time:
        return current_event
    
    case_elapsed_col = properties.get("time_since_case_start_column")
    day_col = properties.get("day_in_week_column")
    seconds_col = properties.get("seconds_in_day_column")
    
    # time_since_last_event_column remains unchanged (keep current event's value)
    # No action needed here as we're not modifying this column
    
    # Calculate time_since_case_start_column
    if case_elapsed_col and case_elapsed_col in current_event.index and case_elapsed_col in previous_event.index:
        if event_elapsed_col and event_elapsed_col in current_event.index:
            # Check for NaN values in previous event as well
            if not pd.isna(previous_event[case_elapsed_col]) and not pd.isna(current_event[event_elapsed_col]):
                # Convert to Python float to avoid numpy float32 dtype
                current_event[case_elapsed_col] = float(previous_event[case_elapsed_col] + current_event[event_elapsed_col])
    
    # Calculate seconds_in_day
    if seconds_col and seconds_col in current_event.index and seconds_col in previous_event.index:
        if event_elapsed_col and event_elapsed_col in current_event.index:
            # Check for NaN values
            if not pd.isna(previous_event[seconds_col]) and not pd.isna(current_event[event_elapsed_col]):
                # Convert to Python float to avoid numpy float32 dtype
                current_event[seconds_col] = float((previous_event[seconds_col] + current_event[event_elapsed_col]) % 86400)
    
    # Calculate day_in_week_column
    if day_col and day_col in current_event.index and day_col in previous_event.index:
        if event_elapsed_col and event_elapsed_col in current_event.index and seconds_col and seconds_col in previous_event.index:
            # Check for NaN values
            if not pd.isna(previous_event[seconds_col]) and not pd.isna(current_event[event_elapsed_col]) and not pd.isna(previous_event[day_col]):
                days_to_add = round((previous_event[seconds_col] + current_event[event_elapsed_col]) / 86400.0)
                # Convert to Python float to avoid numpy float32 dtype
                current_event[day_col] = float((previous_event[day_col] + days_to_add) % 7)
    
    return current_event


def redo_last_activity_of_prefix(
    prefix_df: pd.DataFrame,
    suffix_df: pd.DataFrame,
    properties: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Duplicate the last activity of a prefix (with identical attributes) and
    recalculate the temporal columns for the duplicated event and all subsequent
    suffix events.

    The repeated event (x') has the same time_since_last_event_column as the original
    last event (x), and time features are recalculated based on the previous event.
    Suffix events are also recalculated using the same logic recursively.

    Args:
        prefix_df: DataFrame representing the prefix (readable form).
        suffix_df: DataFrame representing the suffix (readable form).
        properties: Event log properties dict (same used in build_readable_event_log).

    Returns:
        Tuple (updated_prefix_df, updated_suffix_df).
    """
    if prefix_df.empty:
        return prefix_df.copy(), suffix_df.copy()

    timestamp_col = properties.get("timestamp_name")
    event_elapsed_col = properties.get("time_since_last_event_column")

    updated_prefix = prefix_df.copy().reset_index(drop=True)
    updated_suffix = suffix_df.copy().reset_index(drop=True)

    # Ensure timestamp columns are datetime for calculations
    for frame in (updated_prefix, updated_suffix):
        if timestamp_col and timestamp_col in frame.columns:
            frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")

    # Get the last prefix event (x)
    last_prefix_event = updated_prefix.iloc[-1].copy()
    
    # Duplicate last prefix event (x' is a copy of x)
    duplicated_event = last_prefix_event.copy()
    
    # Convert numpy numeric types to Python native types to ensure consistency
    # This prevents issues with np.float32 vs float
    duplicated_event = _convert_numpy_to_python_types(duplicated_event)
    
    # Check if duplicated event is EOS or has NaN time values
    concept_name_col = properties.get("concept_name")
    is_eos_duplicated = False
    if concept_name_col and concept_name_col in duplicated_event.index:
        is_eos_duplicated = duplicated_event[concept_name_col] == 'EOS'
    
    has_nan_time_duplicated = False
    if event_elapsed_col and event_elapsed_col in duplicated_event.index:
        has_nan_time_duplicated = pd.isna(duplicated_event[event_elapsed_col])
    
    # Only calculate time features if not EOS and no NaN
    if not is_eos_duplicated and not has_nan_time_duplicated:
        # For x': time_since_last_event_column stays the same (already copied from x)
        # Calculate other time features based on x (previous event)
        _calculate_time_features(duplicated_event, last_prefix_event, properties)
        
        # Update timestamp if available (based on the time features)
        if timestamp_col and timestamp_col in duplicated_event.index:
            # Recalculate timestamp from case_start + case_elapsed_time if possible
            # This is a fallback - ideally timestamp should be derived from case_start
            # For now, we'll update it based on the elapsed time difference
            if event_elapsed_col and event_elapsed_col in duplicated_event.index:
                if not pd.isna(duplicated_event[event_elapsed_col]) and not pd.isna(last_prefix_event[timestamp_col]):
                    duplicated_event[timestamp_col] = last_prefix_event[timestamp_col] + pd.to_timedelta(
                        duplicated_event[event_elapsed_col], unit="s"
                    )

    # Append duplicated event to prefix
    updated_prefix = pd.concat([updated_prefix, duplicated_event.to_frame().T], ignore_index=True)

    # Recalculate time features for all suffix events recursively
    if not updated_suffix.empty:
        # First suffix event uses x' as previous event
        previous_event = duplicated_event
        
        for idx in range(len(updated_suffix)):
            current_event = updated_suffix.iloc[idx].copy()
            
            # Check if current event is EOS or has NaN time values
            is_eos_current = False
            if concept_name_col and concept_name_col in current_event.index:
                is_eos_current = current_event[concept_name_col] == 'EOS'
            
            has_nan_time_current = False
            if event_elapsed_col and event_elapsed_col in current_event.index:
                has_nan_time_current = pd.isna(current_event[event_elapsed_col])
            
            # Only calculate time features if not EOS and no NaN
            if not is_eos_current and not has_nan_time_current:
                # time_since_last_event_column remains unchanged
                # Calculate other time features based on previous event
                _calculate_time_features(current_event, previous_event, properties)
                
                # Update timestamp
                if timestamp_col and timestamp_col in current_event.index:
                    if event_elapsed_col and event_elapsed_col in current_event.index:
                        if not pd.isna(current_event[event_elapsed_col]) and timestamp_col in previous_event.index and not pd.isna(previous_event[timestamp_col]):
                            current_event[timestamp_col] = previous_event[timestamp_col] + pd.to_timedelta(
                                current_event[event_elapsed_col], unit="s"
                            )
            
            # Convert numpy numeric types to Python native types before updating DataFrame
            current_event = _convert_numpy_to_python_types(current_event)
            
            # Update the suffix DataFrame with the recalculated event
            updated_suffix.iloc[idx] = current_event
            
            # Update previous_event for next iteration (even if it's EOS, we need it for chain)
            previous_event = current_event

    return updated_prefix, updated_suffix


# --- Loop augmentation ---

def _detect_loops_in_sequence(
    activity_sequence: List[str],
) -> List[Tuple[int, int]]:
    """
    Detect all loops in an activity sequence.
    A loop is a subsequence where the first and last activities are the same,
    with at least one event in between (e.g. A,B,C,B is a loop; A,B,B,D is not).

    Returns:
        List of (start_index, end_index) tuples, end_index inclusive.
    """
    loops = []
    n = len(activity_sequence)
    for start in range(n):
        start_activity = activity_sequence[start]
        for end in range(start + 1, n):
            if activity_sequence[end] == start_activity and end - start >= 2:
                loops.append((start, end))
    return loops


def _apply_first_inserted_event_time(
    c_event: pd.Series,
    b_event: pd.Series,
    properties: Dict[str, Any],
) -> None:
    """
    Set time columns for the first inserted event C (after anchor B) per user spec.
    Modifies c_event in place.
    """
    event_elapsed_col = properties.get("time_since_last_event_column")
    case_elapsed_col = properties.get("time_since_case_start_column")
    seconds_col = properties.get("seconds_in_day_column")
    day_col = properties.get("day_in_week_column")
    if not event_elapsed_col or event_elapsed_col not in b_event.index:
        return
    b_elapsed = b_event[event_elapsed_col]
    if pd.isna(b_elapsed):
        return
    if event_elapsed_col in c_event.index:
        c_event[event_elapsed_col] = b_elapsed
    if case_elapsed_col and case_elapsed_col in c_event.index and case_elapsed_col in b_event.index:
        if not pd.isna(b_event[case_elapsed_col]):
            c_event[case_elapsed_col] = float(b_event[case_elapsed_col] + b_elapsed)
    if seconds_col and seconds_col in c_event.index and seconds_col in b_event.index:
        b_sec = b_event[seconds_col] if not pd.isna(b_event.get(seconds_col)) else 0.0
        c_event[seconds_col] = float((b_sec + b_elapsed) % 86400)
    if day_col and day_col in c_event.index and day_col in b_event.index and seconds_col in b_event.index:
        b_sec = b_event[seconds_col] if not pd.isna(b_event.get(seconds_col)) else 0.0
        b_day = b_event[day_col] if not pd.isna(b_event.get(day_col)) else 0
        days_to_add = round((b_sec + b_elapsed) / 86400.0)
        c_event[day_col] = float((b_day + days_to_add) % 7)


def generate_loop_augmentation(
    df: pd.DataFrame,
    properties: Dict[str, Any],
    *,
    min_suffix_size: int = 2,
    max_matches_per_loop: int = 1,
    save_path: Optional[str] = None,
    save_every_n: Optional[int] = None,
    eos_value: str = "EOS",
) -> Tuple[
    Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
    Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
]:
    """
    Find loops in the first half of the df (by cases), greedily match and insert
    them into matching cases in the second half (up to max_matches_per_loop per loop),
    and fill val_loops_clean and val_loops_pert. Time shifting is applied only to val_loops_pert.

    Args:
        df: Readable dataframe with EOS rows (output of get_train_test_val_datasets).
        properties: Event log properties dict.
        min_suffix_size: Minimum non-EOS events required in suffix after insertion.
        max_matches_per_loop: Maximum number of cases in the second half each loop is matched to
            (e.g. 10 means each loop is matched with up to 10 event logs found in the second half).
        save_path: If set with save_every_n, save checkpoint dicts to this directory.
        save_every_n: Save val_loops_clean and val_loops_pert every N successful matches.
        eos_value: Value identifying EOS rows.

    Returns:
        (val_loops_clean, val_loops_pert), each mapping (case_id, prefix_len) -> (prefix_df, suffix_df).
    """
    case_col = properties.get("case_name")
    activity_col = properties.get("concept_name")
    event_elapsed_col = properties.get("time_since_last_event_column")
    timestamp_col = properties.get("timestamp_name")
    if not case_col or not activity_col:
        return {}, {}

    cases = df[case_col].unique()
    n_cases = len(cases)
    half = n_cases // 2
    first_half_ids = set(cases[:half])
    second_half_ids = set(cases[half:])

    first_half_df = df[df[case_col].isin(first_half_ids)].copy()
    second_half_df = df[df[case_col].isin(second_half_ids)].copy()

    # Build list of (anchor_activity, loop_body_df) from first half, in order
    loops_from_first: List[Tuple[str, pd.DataFrame]] = []
    for case_id, group in first_half_df.groupby(case_col, sort=False):
        group = group.reset_index(drop=True)
        non_eos_mask = group[activity_col] != eos_value
        non_eos_df = group[non_eos_mask].reset_index(drop=True)
        if len(non_eos_df) < 2:
            continue
        activity_sequence = non_eos_df[activity_col].tolist()
        loop_indices = _detect_loops_in_sequence(activity_sequence)
        for start_idx, end_idx in loop_indices:
            # Loop segment: start..end inclusive. Body to insert: segment without first event.
            loop_segment = non_eos_df.iloc[start_idx : end_idx + 1].copy()
            loop_body = loop_segment.iloc[1:].copy().reset_index(drop=True)
            if loop_body.empty:
                continue
            anchor = activity_sequence[start_idx]
            loops_from_first.append((anchor, loop_body))

    if not loops_from_first:
        return {}, {}

    val_loops_clean: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]] = {}
    val_loops_pert: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]] = {}
    match_count = 0
    loop_index = 0
    second_half_case_list = list(second_half_df.groupby(case_col, sort=False))

    while loop_index < len(loops_from_first):
        anchor, loop_body = loops_from_first[loop_index]
        loop_index += 1
        matches_for_this_loop = 0

        # Up to max_matches_per_loop cases in second half where anchor appears and insertion is valid
        for case_id, target_group in second_half_case_list:
            if matches_for_this_loop >= max_matches_per_loop:
                break
            target_group = target_group.reset_index(drop=True)
            non_eos_mask = target_group[activity_col] != eos_value
            non_eos_indices = [i for i, m in enumerate(non_eos_mask) if m]
            non_eos_df = target_group[non_eos_mask].reset_index(drop=True)
            if len(non_eos_df) < min_suffix_size + 1:
                continue
            # First occurrence of anchor in non-EOS rows
            match_pos = None
            for i, idx in enumerate(non_eos_indices):
                if target_group.iloc[idx][activity_col] == anchor:
                    match_pos = i
                    break
            if match_pos is None:
                continue
            # prefix_len_clean = number of non-EOS events up to and including anchor
            prefix_len_clean = match_pos + 1
            # Suffix after anchor (non-EOS count) must be >= min_suffix_size
            suffix_non_eos_count = len(non_eos_indices) - prefix_len_clean
            if suffix_non_eos_count < min_suffix_size:
                continue
            # Avoid overwriting: first match wins
            key = (str(case_id), prefix_len_clean)
            if key in val_loops_clean:
                continue

            # Split target: before anchor (inclusive), after anchor
            last_prefix_idx = non_eos_indices[prefix_len_clean - 1]
            split_idx = last_prefix_idx + 1
            prefix_clean_df = target_group.iloc[:split_idx].copy()
            suffix_df = target_group.iloc[split_idx:].copy()

            # Build perturbed: prefix_clean + loop_body + suffix, with time updates on body and suffix for pert only
            loop_body_copy = loop_body.copy()
            anchor_row = target_group.iloc[last_prefix_idx]
            # All variables stay from extracted loop; only time columns updated from anchor B
            first_inserted = loop_body_copy.iloc[0].copy()
            _convert_numpy_to_python_types(first_inserted)
            _apply_first_inserted_event_time(first_inserted, anchor_row, properties)
            loop_body_copy.iloc[0] = first_inserted

            # Chain time for rest of loop body and suffix (pert only)
            prev = first_inserted
            for i in range(1, len(loop_body_copy)):
                cur = loop_body_copy.iloc[i].copy()
                _convert_numpy_to_python_types(cur)
                _calculate_time_features(cur, prev, properties)
                if timestamp_col and timestamp_col in cur.index and event_elapsed_col in cur.index:
                    if not pd.isna(prev.get(timestamp_col)) and not pd.isna(cur[event_elapsed_col]):
                        cur[timestamp_col] = prev[timestamp_col] + pd.to_timedelta(
                            cur[event_elapsed_col], unit="s"
                        )
                loop_body_copy.iloc[i] = cur
                prev = cur

            # Shift suffix events (pert)
            suffix_pert = suffix_df.copy()
            if not suffix_pert.empty:
                for idx in range(len(suffix_pert)):
                    cur = suffix_pert.iloc[idx].copy()
                    _convert_numpy_to_python_types(cur)
                    _calculate_time_features(cur, prev, properties)
                    if timestamp_col and timestamp_col in cur.index and event_elapsed_col in cur.index:
                        if not pd.isna(prev.get(timestamp_col)) and not pd.isna(cur[event_elapsed_col]):
                            cur[timestamp_col] = prev[timestamp_col] + pd.to_timedelta(
                                cur[event_elapsed_col], unit="s"
                            )
                    suffix_pert.iloc[idx] = cur
                    prev = cur

            prefix_pert_df = pd.concat(
                [prefix_clean_df, loop_body_copy],
                ignore_index=True,
            )

            val_loops_clean[key] = (prefix_clean_df, suffix_df)
            val_loops_pert[key] = (prefix_pert_df, suffix_pert)
            match_count += 1
            matches_for_this_loop += 1

            if save_path and save_every_n is not None and match_count % save_every_n == 0:
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "val_loops_clean.pkl"), "wb") as f:
                    pickle.dump(val_loops_clean, f)
                with open(os.path.join(save_path, "val_loops_pert.pkl"), "wb") as f:
                    pickle.dump(val_loops_pert, f)
        # Continue to next loop (loop_index already advanced)

    return val_loops_clean, val_loops_pert
