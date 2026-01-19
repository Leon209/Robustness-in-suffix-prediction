"""
Helpdesk Dataset Perturbation Functions

This module provides functions to add perturbations to the helpdesk dataset
for robustness evaluation of trained models.

The perturbations are designed to work with encoded categorical and numerical
features, allowing for systematic evaluation of model robustness.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import copy


class HelpdeskPerturbator:
    """
    A class to handle perturbations on the helpdesk dataset.
    
    This class provides methods to:
    1. Load and encode the helpdesk dataset
    2. Apply various types of perturbations
    3. Save perturbed datasets for robustness evaluation
    """
    
    def __init__(self, csv_path: str = '../data/helpdesk.csv'):
        """
        Initialize the perturbator with the helpdesk dataset.
        
        Args:
            csv_path: Path to the helpdesk CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.encoded_df = None
        self.categorical_encoders = {}
        self.categorical_columns = [
            'Activity', 'Resource', 'Variant index', 'seriousness', 
            'customer', 'product', 'responsible_section', 'seriousness_2', 
            'service_level', 'service_type', 'support_section', 'workgroup'
        ]
        self.case_id_column = 'Case ID'
        
        # Load and prepare the dataset
        self._load_dataset()
        self._encode_categorical_features()
    
    def _load_dataset(self):
        """Load the helpdesk dataset from CSV."""
        print(f"Loading dataset from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} events from {self.df[self.case_id_column].nunique()} cases")
    
    def _encode_categorical_features(self):
        """
        Encode categorical features to numerical values for perturbation.
        
        Creates mappings from categorical values to numerical indices.
        """
        print("Encoding categorical features...")
        self.encoded_df = self.df.copy()
        
        for col in self.categorical_columns:
            if col in self.df.columns:
                # Get unique values and create mapping
                unique_values = self.df[col].unique()
                # Sort to ensure consistent encoding
                unique_values = sorted([v for v in unique_values if pd.notna(v)])
                
                # Create encoder mapping
                self.categorical_encoders[col] = {
                    'values': unique_values,
                    'value_to_idx': {val: idx for idx, val in enumerate(unique_values)},
                    'idx_to_value': {idx: val for idx, val in enumerate(unique_values)}
                }
                
                # Encode the column
                self.encoded_df[f'{col}_encoded'] = self.df[col].map(
                    self.categorical_encoders[col]['value_to_idx']
                )
                
                print(f"  {col}: {len(unique_values)} unique values")
    
    def get_case_last_events(self) -> Dict[str, int]:
        """
        Get the index of the last event for each case.
        
        Returns:
            Dictionary mapping case_id to the index of its last event
        """
        case_last_events = {}
        for case_id in self.df[self.case_id_column].unique():
            case_events = self.df[self.df[self.case_id_column] == case_id]
            last_event_idx = case_events.index[-1]
            case_last_events[case_id] = last_event_idx
        
        return case_last_events
    
    def change_categorical_feature_last_event(
        self, 
        feature_name: str, 
        perturbation_type: str = 'random',
        perturbation_value: Optional[Union[str, int]] = None,
        cases_to_perturb: Optional[List[str]] = None,
        perturbation_probability: float = 1.0
    ) -> pd.DataFrame:
        """
        Change a categorical feature of the last event in traces.
        
        Args:
            feature_name: Name of the categorical feature to perturb
            perturbation_type: Type of perturbation ('random', 'specific', 'swap')
            perturbation_value: Specific value to set (for 'specific' type)
            cases_to_perturb: List of case IDs to perturb (None for all cases)
            perturbation_probability: Probability of applying perturbation to each case
            
        Returns:
            DataFrame with perturbed data
        """
        if feature_name not in self.categorical_columns:
            raise ValueError(f"Feature '{feature_name}' is not a categorical column")
        
        print(f"Applying categorical perturbation to '{feature_name}'")
        print(f"Perturbation type: {perturbation_type}")
        print(f"Perturbation probability: {perturbation_probability}")
        
        # Create a copy of the encoded dataframe
        perturbed_df = self.encoded_df.copy()
        
        # Get last events for each case
        case_last_events = self.get_case_last_events()
        
        # Determine which cases to perturb
        if cases_to_perturb is None:
            cases_to_perturb = list(case_last_events.keys())
        
        # Filter cases based on perturbation probability
        if perturbation_probability < 1.0:
            num_cases = len(cases_to_perturb)
            num_perturb = int(num_cases * perturbation_probability)
            cases_to_perturb = random.sample(cases_to_perturb, num_perturb)
        
        print(f"Perturbing {len(cases_to_perturb)} cases")
        
        # Get available values for the feature
        available_values = self.categorical_encoders[feature_name]['values']
        available_indices = list(range(len(available_values)))
        
        perturbations_applied = 0
        
        for case_id in cases_to_perturb:
            last_event_idx = case_last_events[case_id]
            
            # Get current value
            current_value = perturbed_df.loc[last_event_idx, feature_name]
            current_encoded = perturbed_df.loc[last_event_idx, f'{feature_name}_encoded']
            
            # Determine new value based on perturbation type
            if perturbation_type == 'random':
                # Randomly select a different value
                other_values = [v for v in available_values if v != current_value]
                if other_values:
                    new_value = random.choice(other_values)
                else:
                    continue  # Skip if no other values available
            
            elif perturbation_type == 'specific':
                # Use specific perturbation value
                if perturbation_value is None:
                    raise ValueError("perturbation_value must be provided for 'specific' type")
                if perturbation_value not in available_values:
                    raise ValueError(f"perturbation_value '{perturbation_value}' not found in feature values")
                new_value = perturbation_value
            
            elif perturbation_type == 'swap':
                # Swap with another case's value (if available)
                other_cases = [c for c in cases_to_perturb if c != case_id]
                if other_cases:
                    other_case = random.choice(other_cases)
                    other_last_idx = case_last_events[other_case]
                    new_value = perturbed_df.loc[other_last_idx, feature_name]
                else:
                    continue
            
            else:
                raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
            
            # Apply the perturbation
            if new_value != current_value:
                perturbed_df.loc[last_event_idx, feature_name] = new_value
                perturbed_df.loc[last_event_idx, f'{feature_name}_encoded'] = \
                    self.categorical_encoders[feature_name]['value_to_idx'][new_value]
                perturbations_applied += 1
        
        print(f"Applied {perturbations_applied} perturbations")
        return perturbed_df
    
    def save_perturbed_dataset(
        self, 
        perturbed_df: pd.DataFrame, 
        output_path: str,
        include_encoded_columns: bool = False
    ):
        """
        Save the perturbed dataset to a CSV file.
        
        Args:
            perturbed_df: The perturbed DataFrame
            output_path: Path where to save the CSV file
            include_encoded_columns: Whether to include encoded columns in output
        """
        # Prepare output dataframe
        if include_encoded_columns:
            output_df = perturbed_df
        else:
            # Remove encoded columns for cleaner output
            encoded_cols = [col for col in perturbed_df.columns if col.endswith('_encoded')]
            output_df = perturbed_df.drop(columns=encoded_cols)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        print(f"Saved perturbed dataset to {output_path}")
        print(f"Dataset shape: {output_df.shape}")
    
    def get_feature_statistics(self, feature_name: str) -> Dict:
        """
        Get statistics about a categorical feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature statistics
        """
        if feature_name not in self.categorical_columns:
            raise ValueError(f"Feature '{feature_name}' is not a categorical column")
        
        encoder = self.categorical_encoders[feature_name]
        value_counts = self.df[feature_name].value_counts()
        
        return {
            'feature_name': feature_name,
            'total_unique_values': len(encoder['values']),
            'values': encoder['values'],
            'value_counts': value_counts.to_dict(),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
        }
    
    def analyze_perturbation_impact(
        self, 
        original_df: pd.DataFrame, 
        perturbed_df: pd.DataFrame,
        feature_name: str
    ) -> Dict:
        """
        Analyze the impact of perturbations on a feature.
        
        Args:
            original_df: Original DataFrame
            perturbed_df: Perturbed DataFrame
            feature_name: Name of the perturbed feature
            
        Returns:
            Dictionary with perturbation analysis
        """
        # Get last events
        case_last_events = self.get_case_last_events()
        
        changes = 0
        total_cases = len(case_last_events)
        
        for case_id, last_event_idx in case_last_events.items():
            original_value = original_df.loc[last_event_idx, feature_name]
            perturbed_value = perturbed_df.loc[last_event_idx, feature_name]
            
            if original_value != perturbed_value:
                changes += 1
        
        return {
            'feature_name': feature_name,
            'total_cases': total_cases,
            'cases_changed': changes,
            'change_percentage': (changes / total_cases) * 100 if total_cases > 0 else 0
        }
    
    def change_multiple_categorical_features_last_event(
        self,
        feature_configs: List[Dict],
        perturbation_strategy: str = 'independent'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Change multiple categorical features of the last event in traces.
        
        Args:
            feature_configs: List of dictionaries with feature configuration
                Each dict should contain:
                - 'feature_name': Name of the categorical feature
                - 'perturbation_type': Type of perturbation ('random', 'specific', 'swap')
                - 'perturbation_probability': Probability of applying perturbation
                - 'perturbation_value': Specific value to set (for 'specific' type, optional)
            perturbation_strategy: Strategy for applying perturbations
                - 'independent': Each feature perturbed independently
                - 'correlated': Same cases perturbed for all features
                - 'cascading': Primary feature affects secondary features
        
        Returns:
            Tuple of (perturbed_dataframe, impact_analysis_dict)
        """
        print(f"Applying multi-feature perturbations with {perturbation_strategy} strategy")
        print(f"Features to perturb: {[config['feature_name'] for config in feature_configs]}")
        
        # Create a copy of the encoded dataframe
        perturbed_df = self.encoded_df.copy()
        impacts = {}
        
        if perturbation_strategy == 'independent':
            # Each feature perturbed independently
            for config in feature_configs:
                feature_name = config['feature_name']
                if feature_name in self.categorical_columns:
                    print(f"Perturbing {feature_name} independently...")
                    
                    feature_perturbed_df = self.change_categorical_feature_last_event(
                        feature_name=feature_name,
                        perturbation_type=config.get('perturbation_type', 'random'),
                        perturbation_value=config.get('perturbation_value', None),
                        perturbation_probability=config['perturbation_probability']
                    )
                    
                    # Update the main dataframe
                    case_last_events = self.get_case_last_events()
                    for case_id in self.df['Case ID'].unique():
                        last_event_idx = case_last_events[case_id]
                        if (self.encoded_df.loc[last_event_idx, feature_name] != 
                            feature_perturbed_df.loc[last_event_idx, feature_name]):
                            perturbed_df.loc[last_event_idx, feature_name] = feature_perturbed_df.loc[last_event_idx, feature_name]
                            perturbed_df.loc[last_event_idx, f'{feature_name}_encoded'] = feature_perturbed_df.loc[last_event_idx, f'{feature_name}_encoded']
                    
                    # Analyze impact
                    impact = self.analyze_perturbation_impact(
                        self.encoded_df, perturbed_df, feature_name
                    )
                    impacts[feature_name] = impact
                    print(f"  {feature_name}: {impact['change_percentage']:.2f}% cases changed")
        
        elif perturbation_strategy == 'correlated':
            # Same cases perturbed for all features
            all_cases = list(self.df['Case ID'].unique())
            
            # Use the first feature's probability to determine case selection
            primary_config = feature_configs[0]
            num_cases_to_perturb = int(len(all_cases) * primary_config['perturbation_probability'])
            cases_to_perturb = random.sample(all_cases, num_cases_to_perturb)
            
            print(f"Perturbing {len(cases_to_perturb)} cases across {len(feature_configs)} features")
            
            for config in feature_configs:
                feature_name = config['feature_name']
                if feature_name in self.categorical_columns:
                    print(f"Perturbing {feature_name} on same cases...")
                    
                    feature_perturbed_df = self.change_categorical_feature_last_event(
                        feature_name=feature_name,
                        perturbation_type=config.get('perturbation_type', 'random'),
                        perturbation_value=config.get('perturbation_value', None),
                        cases_to_perturb=cases_to_perturb,
                        perturbation_probability=1.0  # Apply to all selected cases
                    )
                    
                    # Update the main dataframe
                    case_last_events = self.get_case_last_events()
                    for case_id in cases_to_perturb:
                        last_event_idx = case_last_events[case_id]
                        perturbed_df.loc[last_event_idx, feature_name] = feature_perturbed_df.loc[last_event_idx, feature_name]
                        perturbed_df.loc[last_event_idx, f'{feature_name}_encoded'] = feature_perturbed_df.loc[last_event_idx, f'{feature_name}_encoded']
                    
                    # Analyze impact
                    impact = self.analyze_perturbation_impact(
                        self.encoded_df, perturbed_df, feature_name
                    )
                    impacts[feature_name] = impact
                    print(f"  {feature_name}: {impact['change_percentage']:.2f}% cases changed")
        
        elif perturbation_strategy == 'cascading':
            # Primary feature affects secondary features
            primary_config = feature_configs[0]
            secondary_configs = feature_configs[1:]
            
            primary_feature = primary_config['feature_name']
            print(f"Primary corruption: {primary_feature} ({primary_config['perturbation_probability']*100:.1f}%)")
            
            # Step 1: Apply primary perturbation
            primary_perturbed_df = self.change_categorical_feature_last_event(
                feature_name=primary_feature,
                perturbation_type=primary_config.get('perturbation_type', 'random'),
                perturbation_value=primary_config.get('perturbation_value', None),
                perturbation_probability=primary_config['perturbation_probability']
            )
            
            # Get cases affected by primary perturbation
            case_last_events = self.get_case_last_events()
            primary_affected_cases = []
            
            for case_id in self.df['Case ID'].unique():
                last_event_idx = case_last_events[case_id]
                if (self.encoded_df.loc[last_event_idx, primary_feature] != 
                    primary_perturbed_df.loc[last_event_idx, primary_feature]):
                    primary_affected_cases.append(case_id)
            
            print(f"Primary perturbation affected {len(primary_affected_cases)} cases")
            
            # Update main dataframe with primary perturbation
            for case_id in primary_affected_cases:
                last_event_idx = case_last_events[case_id]
                perturbed_df.loc[last_event_idx, primary_feature] = primary_perturbed_df.loc[last_event_idx, primary_feature]
                perturbed_df.loc[last_event_idx, f'{primary_feature}_encoded'] = primary_perturbed_df.loc[last_event_idx, f'{primary_feature}_encoded']
            
            # Step 2: Apply cascade perturbations
            for config in secondary_configs:
                feature_name = config['feature_name']
                cascade_probability = config['perturbation_probability']
                
                if feature_name in self.categorical_columns:
                    num_cascade_cases = int(len(primary_affected_cases) * cascade_probability)
                    cascade_cases = random.sample(primary_affected_cases, num_cascade_cases)
                    
                    print(f"Cascading to {feature_name} ({len(cascade_cases)} cases)")
                    
                    cascade_perturbed_df = self.change_categorical_feature_last_event(
                        feature_name=feature_name,
                        perturbation_type=config.get('perturbation_type', 'random'),
                        perturbation_value=config.get('perturbation_value', None),
                        cases_to_perturb=cascade_cases,
                        perturbation_probability=1.0
                    )
                    
                    # Update main dataframe
                    for case_id in cascade_cases:
                        last_event_idx = case_last_events[case_id]
                        perturbed_df.loc[last_event_idx, feature_name] = cascade_perturbed_df.loc[last_event_idx, feature_name]
                        perturbed_df.loc[last_event_idx, f'{feature_name}_encoded'] = cascade_perturbed_df.loc[last_event_idx, f'{feature_name}_encoded']
            
            # Analyze impacts for all features
            all_features = [primary_feature] + [config['feature_name'] for config in secondary_configs]
            for feature_name in all_features:
                if feature_name in self.categorical_columns:
                    impact = self.analyze_perturbation_impact(
                        self.encoded_df, perturbed_df, feature_name
                    )
                    impacts[feature_name] = impact
                    print(f"  {feature_name}: {impact['change_percentage']:.2f}% cases changed")
        
        else:
            raise ValueError(f"Unknown perturbation_strategy: {perturbation_strategy}")
        
        print(f"Multi-feature perturbation completed with {perturbation_strategy} strategy")
        return perturbed_df, impacts
    
    def get_case_first_events(self) -> Dict[str, int]:
        """
        Get the index of the first event for each case.
        
        Returns:
            Dictionary mapping case_id to the index of its first event
        """
        case_first_events = {}
        for case_id in self.df['Case ID'].unique():
            case_events = self.df[self.df['Case ID'] == case_id]
            first_event_idx = case_events.index[0]  # First event, not last
            case_first_events[case_id] = first_event_idx
        
        return case_first_events
    
    def change_multiple_features_of_first_event(
        self,
        num_features_to_perturb: int,
        perturbable_features: List[str],
        perturbation_probability: float = 1.0,
        perturbation_type: str = 'random',
        cases_to_perturb: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Randomly perturb multiple features of the first event in traces.
        
        Args:
            num_features_to_perturb: Number of features to perturb per case
            perturbable_features: List of features that can be perturbed
            perturbation_probability: Probability of applying perturbation to each case
            perturbation_type: Type of perturbation ('random', 'specific', 'swap')
            cases_to_perturb: List of case IDs to perturb (None for all cases)
            
        Returns:
            Tuple of (perturbed_dataframe, impact_analysis_dict)
        """
        print(f"Applying first-event perturbations to {num_features_to_perturb} features per case")
        print(f"Perturbable features: {perturbable_features}")
        print(f"Perturbation type: {perturbation_type}")
        print(f"Perturbation probability: {perturbation_probability}")
        
        # Input validation
        for feature in perturbable_features:
            if feature not in self.categorical_columns:
                raise ValueError(f"Feature '{feature}' not in categorical columns")
        
        if num_features_to_perturb > len(perturbable_features):
            raise ValueError("num_features_to_perturb cannot exceed perturbable_features length")
        if num_features_to_perturb < 1:
            raise ValueError("num_features_to_perturb must be at least 1")
        
        # Create a copy of the encoded dataframe
        perturbed_df = self.encoded_df.copy()
        
        # Get all cases or filter by cases_to_perturb
        all_cases = cases_to_perturb if cases_to_perturb else list(self.df['Case ID'].unique())
        
        # Apply perturbation probability
        if perturbation_probability < 1.0:
            num_cases = int(len(all_cases) * perturbation_probability)
            selected_cases = random.sample(all_cases, num_cases)
        else:
            selected_cases = all_cases
        
        print(f"Selected {len(selected_cases)} cases for perturbation")
        
        # Get first events for each case
        case_first_events = self.get_case_first_events()
        
        # Track perturbation details
        perturbation_details = {}
        perturbations_applied = 0
        
        # For each selected case, randomly choose features to perturb
        for case_id in selected_cases:
            # Randomly select features for this case
            features_to_perturb = random.sample(
                perturbable_features, 
                num_features_to_perturb
            )
            perturbation_details[case_id] = features_to_perturb
            
            first_event_idx = case_first_events[case_id]
            
            # Apply perturbations to selected features
            for feature_name in features_to_perturb:
                # Get current value
                current_value = perturbed_df.loc[first_event_idx, feature_name]
                
                # Determine new value based on perturbation type
                available_values = self.categorical_encoders[feature_name]['values']
                
                if perturbation_type == 'random':
                    # Randomly select a different value
                    other_values = [v for v in available_values if v != current_value]
                    if other_values:
                        new_value = random.choice(other_values)
                    else:
                        continue  # Skip if no other values available
                
                elif perturbation_type == 'specific':
                    # Use a specific perturbation value (first available value)
                    if len(available_values) > 1:
                        new_value = available_values[1] if available_values[1] != current_value else available_values[0]
                    else:
                        continue
                
                elif perturbation_type == 'swap':
                    # Swap with another case's value (if available)
                    other_cases = [c for c in selected_cases if c != case_id]
                    if other_cases:
                        other_case = random.choice(other_cases)
                        other_first_idx = case_first_events[other_case]
                        new_value = perturbed_df.loc[other_first_idx, feature_name]
                    else:
                        continue
                
                else:
                    raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
                
                # Apply the perturbation
                if new_value != current_value:
                    perturbed_df.loc[first_event_idx, feature_name] = new_value
                    perturbed_df.loc[first_event_idx, f'{feature_name}_encoded'] = \
                        self.categorical_encoders[feature_name]['value_to_idx'][new_value]
                    perturbations_applied += 1
        
        print(f"Applied {perturbations_applied} perturbations across {len(selected_cases)} cases")
        
        # Analyze impact
        impact_analysis = self._analyze_first_event_perturbation_impact(
            perturbation_details, case_first_events
        )
        
        return perturbed_df, impact_analysis
    
    def _analyze_first_event_perturbation_impact(
        self, 
        perturbation_details: Dict[str, List[str]], 
        case_first_events: Dict[str, int]
    ) -> Dict:
        """
        Analyze the impact of first-event perturbations.
        
        Args:
            perturbation_details: Dictionary mapping case_id to list of perturbed features
            case_first_events: Dictionary mapping case_id to first event index
            
        Returns:
            Dictionary with perturbation analysis
        """
        total_cases_affected = len(perturbation_details)
        
        # Count features perturbed per case
        features_per_case = [len(features) for features in perturbation_details.values()]
        
        # Count total perturbations per feature
        feature_counts = {}
        for features in perturbation_details.values():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Calculate statistics
        avg_features_per_case = sum(features_per_case) / len(features_per_case) if features_per_case else 0
        
        return {
            'total_cases_affected': total_cases_affected,
            'total_perturbations_applied': sum(features_per_case),
            'avg_features_per_case': avg_features_per_case,
            'features_perturbed_per_case': {
                'min': min(features_per_case) if features_per_case else 0,
                'max': max(features_per_case) if features_per_case else 0,
                'distribution': features_per_case
            },
            'perturbation_summary': feature_counts,
            'case_level_details': perturbation_details
        }

    def all_events_attack(
        self,
        num_features_to_perturb: int,
        perturbable_features: List[str],
        perturbation_probability: float = 1.0,
        perturbation_type: str = 'random',
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Randomly perturb multiple features for all events in each selected case.
        
        Args:
            num_features_to_perturb: Number of features to perturb per event
            perturbable_features: List of feature names eligible for perturbation
            perturbation_probability: Probability to select a case for attack
            perturbation_type: 'random' | 'specific' | 'swap'
        
        Returns:
            Tuple of (perturbed_dataframe, impact_analysis_dict)
        """
        print(f"Applying all-events attack: {num_features_to_perturb} features per event")
        print(f"Perturbable features: {perturbable_features}")
        print(f"Perturbation type: {perturbation_type}")
        print(f"Perturbation probability (case-level): {perturbation_probability}")
        
        # Input validation
        for feature in perturbable_features:
            if feature not in self.categorical_columns:
                raise ValueError(f"Feature '{feature}' not in categorical columns")
        if num_features_to_perturb > len(perturbable_features):
            raise ValueError("num_features_to_perturb cannot exceed perturbable_features length")
        if num_features_to_perturb < 1:
            raise ValueError("num_features_to_perturb must be at least 1")
        
        # Work on a copy
        perturbed_df = self.encoded_df.copy()
        
        # Select cases
        all_cases = list(self.df['Case ID'].unique())
        if perturbation_probability < 1.0:
            num_cases = max(0, int(len(all_cases) * perturbation_probability))
            selected_cases = random.sample(all_cases, num_cases)
        else:
            selected_cases = all_cases
        print(f"Selected {len(selected_cases)} cases for all-events attack")
        
        # Track details per case and event
        perturbation_details: Dict[str, Dict[int, List[str]]] = {}
        perturbations_applied = 0
        events_affected = 0
        
        for case_id in selected_cases:
            case_events_idx = self.df[self.df['Case ID'] == case_id].index.tolist()
            if not case_events_idx:
                continue
            perturbation_details[case_id] = {}
            
            for event_idx in case_events_idx:
                # Choose features for this event
                features_to_perturb = random.sample(perturbable_features, num_features_to_perturb)
                changed_any = False
                
                for feature_name in features_to_perturb:
                    current_value = perturbed_df.loc[event_idx, feature_name]
                    available_values = self.categorical_encoders[feature_name]['values']
                    
                    # Determine new value
                    if perturbation_type == 'random':
                        other_values = [v for v in available_values if v != current_value]
                        if not other_values:
                            continue
                        new_value = random.choice(other_values)
                    elif perturbation_type == 'specific':
                        if len(available_values) > 1:
                            new_value = available_values[1] if available_values[1] != current_value else available_values[0]
                        else:
                            continue
                    elif perturbation_type == 'swap':
                        # Swap with a random event (could be from any case)
                        other_event_idx = random.choice(self.df.index.tolist())
                        if other_event_idx == event_idx:
                            continue
                        new_value = perturbed_df.loc[other_event_idx, feature_name]
                    else:
                        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")
                    
                    if new_value != current_value:
                        perturbed_df.loc[event_idx, feature_name] = new_value
                        perturbed_df.loc[event_idx, f'{feature_name}_encoded'] = \
                            self.categorical_encoders[feature_name]['value_to_idx'][new_value]
                        perturbations_applied += 1
                        changed_any = True
                        
                        # Record detail
                        if event_idx not in perturbation_details[case_id]:
                            perturbation_details[case_id][event_idx] = []
                        perturbation_details[case_id][event_idx].append(feature_name)
                
                if changed_any:
                    events_affected += 1
        
        print(f"Applied {perturbations_applied} perturbations across {events_affected} events in {len(selected_cases)} cases")
        
        # Build impact analysis
        impact = self._analyze_all_events_attack_impact(perturbation_details)
        return perturbed_df, impact
    
    def _analyze_all_events_attack_impact(
        self,
        perturbation_details: Dict[str, Dict[int, List[str]]]
    ) -> Dict:
        """
        Summarize impact of all-events attack.
        
        Returns statistics over cases, events, and feature perturbations.
        """
        total_cases_affected = len(perturbation_details)
        total_events_affected = sum(len(events) for events in perturbation_details.values())
        total_perturbations = sum(len(features) for events in perturbation_details.values() for features in events.values())
        
        features_counter: Dict[str, int] = {}
        per_event_distribution: List[int] = []
        for events in perturbation_details.values():
            for features in events.values():
                per_event_distribution.append(len(features))
                for f in features:
                    features_counter[f] = features_counter.get(f, 0) + 1
        
        avg_features_per_event = (sum(per_event_distribution) / len(per_event_distribution)) if per_event_distribution else 0.0
        
        return {
            'total_cases_affected': total_cases_affected,
            'total_events_affected': total_events_affected,
            'total_perturbations_applied': total_perturbations,
            'avg_features_per_event': avg_features_per_event,
            'features_perturbed_counts': features_counter,
            'per_event_feature_count_distribution': per_event_distribution,
            'case_event_details': perturbation_details,
        }

