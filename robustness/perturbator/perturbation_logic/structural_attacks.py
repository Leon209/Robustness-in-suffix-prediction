import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional


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
