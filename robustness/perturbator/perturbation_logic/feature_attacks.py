import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import random


def _filter_time_columns(features: List[str], properties: Dict[str, Any]) -> List[str]:
    """
    Filter out time-related columns from the list of features.
    
    Args:
        features: List of feature names
        properties: Properties dictionary containing time column names
    
    Returns:
        Filtered list of features excluding time-related columns
    """
    time_columns = [
        properties.get('timestamp_name'),
        properties.get('time_since_case_start_column'),
        properties.get('time_since_last_event_column'),
        properties.get('day_in_week_column'),
        properties.get('seconds_in_day_column'),
    ]
    # Filter out None values and remove from features
    time_columns = [col for col in time_columns if col is not None]
    return [f for f in features if f not in time_columns]


def _perturb_numerical_feature(value: float, min_val: float, max_val: float, magnitude: float) -> float:
    """
    Apply magnitude-based perturbation to a numerical feature value.
    
    Args:
        value: Current feature value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        magnitude: Magnitude of perturbation (0.0 to 1.0)
    
    Returns:
        Perturbed value clipped to [min_val, max_val]
    """
    if pd.isna(value):
        return value
    
    # Calculate the range
    feature_range = max_val - min_val
    if feature_range <= 0:
        return value
    
    # Calculate change amount: magnitude * range
    change_amount = magnitude * feature_range
    
    # Random direction: -1 or +1
    direction = random.choice([-1, 1])
    change = change_amount * direction
    
    # Apply change and clip to range
    new_value = value + change
    new_value = max(min_val, min(max_val, new_value))
    
    return new_value


def _perturb_categorical_feature(categories: List[str]) -> str:
    """
    Randomly select a category from the available categories.
    
    Args:
        categories: List of available categories
    
    Returns:
        Randomly selected category
    """
    if not categories:
        return None
    return random.choice(categories)


def last_event_attack(
    data: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
    properties: Dict[str, Any],
    feature_info: Dict[str, Any],
    attackable_features: List[str],
    num_of_features_to_attack: int,
    magnitude: float,
    feature_range_scope: str = 'global',
    random_seed: int = None,
) -> Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Attack the last event of each prefix by perturbing selected features.
    
    Args:
        data: Dictionary mapping (case_id, prefix_len) to (prefix_df, suffix_df)
        properties: Properties dictionary with column definitions (for time column filtering and activity column)
        feature_info: Dictionary with 'global'/'local' keys (new format) or 'categorical'/'continuous' keys (old format for backward compatibility)
        attackable_features: List of feature names that could be attacked
        num_of_features_to_attack: Number of features to attack per last event (1 to |attackable_features|)
        magnitude: Magnitude of perturbation for numerical features (0.0 to 1.0)
        feature_range_scope: 'global' or 'local' - determines whether to use global or activity-specific feature ranges
        random_seed: Random seed for reproducibility
    
    Returns:
        Modified data dictionary with perturbed prefixes
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Filter out time-related columns
    filtered_features = _filter_time_columns(attackable_features, properties)
    
    if not filtered_features:
        return data.copy()
    
    # Ensure num_of_features_to_attack is within valid range
    num_of_features_to_attack = max(1, min(num_of_features_to_attack, len(filtered_features)))
    
    # Get activity column name for EOS detection
    activity_col = properties.get('concept_name', 'Activity')
    eos_value = 'EOS'
    
    # Determine feature info structure (backward compatibility)
    if 'global' in feature_info:
        # New format with 'global' and 'local' keys
        use_local = (feature_range_scope == 'local')
        if use_local:
            local_info = feature_info.get('local', {})
        global_info = feature_info.get('global', {})
        categorical_info = global_info.get('categorical', {})
        continuous_info = global_info.get('continuous', {})
    else:
        # Old format (backward compatibility)
        use_local = False
        categorical_info = feature_info.get('categorical', {})
        continuous_info = feature_info.get('continuous', {})
    
    # Create modified data dictionary
    modified_data = {}
    
    for key, (prefix_df, suffix_df) in data.items():
        # Make a copy of the prefix to modify
        modified_prefix = prefix_df.copy()
        
        # Find the last non-EOS event
        non_eos_mask = modified_prefix[activity_col] != eos_value
        non_eos_indices = modified_prefix[non_eos_mask].index
        
        if len(non_eos_indices) == 0:
            # No non-EOS events, skip
            modified_data[key] = (modified_prefix, suffix_df.copy())
            continue
        
        # Get the last non-EOS event index
        last_event_idx = non_eos_indices[-1]
        
        # Get activity name for local feature info lookup
        activity_name = None
        if use_local:
            activity_name = modified_prefix.at[last_event_idx, activity_col]
            # If activity not found in local info, skip perturbation for this event
            if activity_name not in local_info:
                modified_data[key] = (modified_prefix, suffix_df.copy())
                continue
        
        # Select feature info based on scope
        if use_local and activity_name in local_info:
            activity_info = local_info[activity_name]
            local_categorical_info = activity_info.get('categorical', {})
            local_continuous_info = activity_info.get('continuous', {})
        else:
            local_categorical_info = {}
            local_continuous_info = {}
        
        # Randomly select features to attack
        features_to_attack = random.sample(filtered_features, num_of_features_to_attack)
        
        # Attack each selected feature
        for feature in features_to_attack:
            if feature not in modified_prefix.columns:
                continue
            
            # Determine which feature info to use (local if available, otherwise global)
            if use_local and activity_name in local_info:
                # Try local first, fall back to global if feature not in local
                feature_categorical_info = local_categorical_info if feature in local_categorical_info else categorical_info
                feature_continuous_info = local_continuous_info if feature in local_continuous_info else continuous_info
            else:
                # Use global feature info
                feature_categorical_info = categorical_info
                feature_continuous_info = continuous_info
            
            # Determine if feature is categorical or continuous
            if feature in feature_categorical_info and feature_categorical_info[feature]:
                # Categorical feature: randomly select a category
                categories = feature_categorical_info[feature]
                new_value = _perturb_categorical_feature(categories)
                if new_value is not None:
                    modified_prefix.at[last_event_idx, feature] = new_value
            elif feature in feature_continuous_info:
                # Continuous feature: apply magnitude-based perturbation
                range_info = feature_continuous_info[feature]
                min_val = range_info['min']
                max_val = range_info['max']
                current_value = modified_prefix.at[last_event_idx, feature]
                
                if pd.notna(current_value):
                    new_value = _perturb_numerical_feature(current_value, min_val, max_val, magnitude)
                    modified_prefix.at[last_event_idx, feature] = new_value
        
        modified_data[key] = (modified_prefix, suffix_df.copy())
    
    return modified_data


def random_event_attack(
    data: Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]],
    properties: Dict[str, Any],
    feature_info: Dict[str, Any],
    attackable_features: List[str],
    num_of_features_to_attack: int,
    event_attack_probability: float,
    magnitude: float = 0.5,
    feature_range_scope: str = 'global',
    random_seed: int = None,
) -> Dict[Tuple[str, int], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Attack random events in each prefix with a given probability.
    
    Args:
        data: Dictionary mapping (case_id, prefix_len) to (prefix_df, suffix_df)
        properties: Properties dictionary with column definitions (for time column filtering and activity column)
        feature_info: Dictionary with 'global'/'local' keys (new format) or 'categorical'/'continuous' keys (old format for backward compatibility)
        attackable_features: List of feature names that could be attacked
        num_of_features_to_attack: Number of features to attack per event (1 to |attackable_features|)
        event_attack_probability: Probability of attacking each event (0.0 to 1.0)
        magnitude: Magnitude of perturbation for numerical features (0.0 to 1.0), default 0.5
        feature_range_scope: 'global' or 'local' - determines whether to use global or activity-specific feature ranges
        random_seed: Random seed for reproducibility
    
    Returns:
        Modified data dictionary with perturbed prefixes
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Filter out time-related columns
    filtered_features = _filter_time_columns(attackable_features, properties)
    
    if not filtered_features:
        return data.copy()
    
    # Ensure num_of_features_to_attack is within valid range
    num_of_features_to_attack = max(1, min(num_of_features_to_attack, len(filtered_features)))
    
    # Get activity column name for EOS detection
    activity_col = properties.get('concept_name', 'Activity')
    eos_value = 'EOS'
    
    # Determine feature info structure (backward compatibility)
    if 'global' in feature_info:
        # New format with 'global' and 'local' keys
        use_local = (feature_range_scope == 'local')
        if use_local:
            local_info = feature_info.get('local', {})
        global_info = feature_info.get('global', {})
        categorical_info = global_info.get('categorical', {})
        continuous_info = global_info.get('continuous', {})
    else:
        # Old format (backward compatibility)
        use_local = False
        categorical_info = feature_info.get('categorical', {})
        continuous_info = feature_info.get('continuous', {})
    
    # Create modified data dictionary
    modified_data = {}
    
    for key, (prefix_df, suffix_df) in data.items():
        # Make a copy of the prefix to modify
        modified_prefix = prefix_df.copy()
        
        # Get all non-EOS event indices
        non_eos_mask = modified_prefix[activity_col] != eos_value
        non_eos_indices = modified_prefix[non_eos_mask].index.tolist()
        
        # For each non-EOS event, independently decide to attack with probability p
        for event_idx in non_eos_indices:
            if random.random() < event_attack_probability:
                # This event will be attacked
                # Get activity name for local feature info lookup
                activity_name = None
                if use_local:
                    activity_name = modified_prefix.at[event_idx, activity_col]
                    # If activity not found in local info, skip perturbation for this event
                    if activity_name not in local_info:
                        continue
                
                # Select feature info based on scope
                if use_local and activity_name in local_info:
                    activity_info = local_info[activity_name]
                    local_categorical_info = activity_info.get('categorical', {})
                    local_continuous_info = activity_info.get('continuous', {})
                else:
                    local_categorical_info = {}
                    local_continuous_info = {}
                
                # Randomly select features to attack
                features_to_attack = random.sample(filtered_features, num_of_features_to_attack)
                
                # Attack each selected feature
                for feature in features_to_attack:
                    if feature not in modified_prefix.columns:
                        continue
                    
                    # Determine which feature info to use (local if available, otherwise global)
                    if use_local and activity_name in local_info:
                        # Try local first, fall back to global if feature not in local
                        feature_categorical_info = local_categorical_info if feature in local_categorical_info else categorical_info
                        feature_continuous_info = local_continuous_info if feature in local_continuous_info else continuous_info
                    else:
                        # Use global feature info
                        feature_categorical_info = categorical_info
                        feature_continuous_info = continuous_info
                    
                    # Determine if feature is categorical or continuous
                    if feature in feature_categorical_info and feature_categorical_info[feature]:
                        # Categorical feature: randomly select a category
                        categories = feature_categorical_info[feature]
                        new_value = _perturb_categorical_feature(categories)
                        if new_value is not None:
                            modified_prefix.at[event_idx, feature] = new_value
                    elif feature in feature_continuous_info:
                        # Continuous feature: apply magnitude-based perturbation
                        range_info = feature_continuous_info[feature]
                        min_val = range_info['min']
                        max_val = range_info['max']
                        current_value = modified_prefix.at[event_idx, feature]
                        
                        if pd.notna(current_value):
                            new_value = _perturb_numerical_feature(
                                current_value, min_val, max_val, magnitude
                            )
                            modified_prefix.at[event_idx, feature] = new_value
        
        modified_data[key] = (modified_prefix, suffix_df.copy())
    
    return modified_data
