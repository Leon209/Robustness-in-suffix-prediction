"""
Camargo LSTM Evaluation Module

This module provides functions to evaluate the Camargo LSTM model for robustness analysis.
Extracted from Helpdesk_gn.ipynb to enable reusable evaluation functionality.
"""

import torch
import random
from typing import Optional, Tuple, List, Dict, Any
from tqdm.notebook import tqdm


# Global placeholders for multiprocessing workers
global_model = None
global_samples_per_case = None
global_cat_categories = None
global_scaler_params = None
global_dict_cat_class_id = None


def init_worker(model,
                samples_per_case: int,
                cat_categories,
                scaler_params,
                dict_cat_class_ids,
                ):
    """
    Initializer for each worker process, setting global variables.
    
    Args:
        model: The Camargo LSTM model
        samples_per_case: Number of samples to generate per case
        cat_categories: Categorical categories information
        scaler_params: Scaler parameters for denormalization
        dict_cat_class_ids: Dictionary mapping class names to indices
    """
    global global_model, global_samples_per_case, global_cat_categories, global_scaler_params, global_dict_cat_class_id
    
    # Models have already been moved to CPU before forking
    model.eval()
    
    global_model = model
    global_samples_per_case = samples_per_case
    global_cat_categories = cat_categories
    global_scaler_params = scaler_params
    global_dict_cat_class_id = dict_cat_class_ids


@torch.no_grad()
def iterate_case(full_case: Tuple[List[torch.Tensor], List[torch.Tensor]],
                 concept_name_id: int,
                 min_suffix_size: int):
    """
    Iterate through a case generating prefix-suffix pairs for evaluation.
    
    Args:
        full_case: Tuple containing categorical and numerical tensors
        concept_name_id: ID of the concept name (Activity) in categorical features
        min_suffix_size: Minimum suffix size
        
    Yields:
        Tuple of (prefix_length, (cats_prefix, nums_prefix))
    """
    cats_full, nums_full, _ = full_case
    seq_len = cats_full[0].size(0)
    window_size = seq_len - min_suffix_size

    # Initialize with all‐zero padding, batch dim = 1
    cats_prefix: List[torch.Tensor] = [torch.zeros((1, window_size), dtype=cat.dtype) for cat in cats_full]
    nums_prefix: List[torch.Tensor] = [torch.zeros((1, window_size), dtype=num.dtype) for num in nums_full]

    prefix_length = 0

    # Slide the window one event at a time
    for i in range(window_size):
        # Roll left by 1 and insert the new event at the rightmost slot
        for j, cat_stream in enumerate(cats_full):
            cats_prefix[j][0] = torch.roll(cats_prefix[j][0], shifts=-1, dims=0)
            cats_prefix[j][0, -1] = cat_stream[i]

        for j, num_stream in enumerate(nums_full):
            nums_prefix[j][0] = torch.roll(nums_prefix[j][0], shifts=-1, dims=0)
            nums_prefix[j][0, -1] = num_stream[i]

        # Only start yielding once we've seen at least one real "activity" token
        if prefix_length > 0 or cats_prefix[concept_name_id][0, -1] != 0:
            prefix_length += 1
            
            yield prefix_length, (cats_prefix, nums_prefix)


@torch.no_grad()
def _evaluate_case(case_name: str,
                   full_case: Tuple[List[torch.Tensor], List[torch.Tensor], str],
                   concept_name_id: int,
                   min_suffix_size: int,
                   ):
    """
    Evaluate a single case generating predictions for all prefix lengths.
    
    Args:
        case_name: Name of the case
        full_case: Tuple containing categorical and numerical tensors
        concept_name_id: ID of the concept name (Activity) in categorical features
        min_suffix_size: Minimum suffix size
        
    Returns:
        List of results for each prefix length
    """
    # List of tensors for test samples:
    cats_full, nums_full, _ = full_case
    # Denormalization values for numerical variables:
    mean_s, std_s = global_scaler_params
    
    act2idx, res2idx = global_dict_cat_class_id  # expect tuple of two dicts
    # Invert them:
    idx2act = {ix: name for name, ix in act2idx.items()}
    idx2res = {ix: name for name, ix in res2idx.items()}

    results = []
    
    # iterate_case already defined elsewhere
    for prefix_length, (cats_pref, nums_pref) in iterate_case(full_case, concept_name_id, min_suffix_size):

        # prefix_prep
        acts = cats_pref[0][0].tolist()
        ress = cats_pref[1][0].tolist()
        times = nums_pref[0][0].tolist()
        # Build the prefix
        prefix_prep = [{"Activity": idx2act[a], "Resource": idx2res[r], "case_elapsed_time": t * std_s + mean_s} 
                      for a, r, t in zip(acts, ress, times) if a != 0]

        # true target: Get from the activity full tensor all indices of the last n values which are not zero
        non_zero_ids = (cats_full[0] != 0).nonzero(as_tuple=True)[0]
        
        # Get the activity ids without the EOS:
        true_acts = cats_full[0][(non_zero_ids[0]+prefix_length):-1].tolist()
        true_ress = cats_full[1][(non_zero_ids[0]+prefix_length):-1].tolist()
        true_nums = nums_full[0][(non_zero_ids[0]+prefix_length):-1].tolist()
        
        # Build target as list of dicts:
        target = [{"Activity": idx2act[a]} for a in true_acts if idx2act[a] != "EOS"]
        if target == []:
            continue

        # MOST LIKELY
        cats_pref_clone = [t.clone() for t in cats_pref]
        nums_pref_clone = [t.clone() for t in nums_pref]
        ml_list = []
        
        max_iterations = len(cats_pref[0][0]) - prefix_length
        
        # Iterate through window size - pref len:
        for i in range(max_iterations):
            # Predictions
            act_probs = global_model((cats_pref_clone, nums_pref_clone))
            # Index of most likely prediction
            index_act = act_probs.argmax(dim=-1).item()
            
            # NaN is predicted new value at position 0
            if index_act == 0:
                act = 'NaN'
            
            # Stop the suffix creation if EOS is predicted
            elif idx2act[index_act] == 'EOS':
                break
            
            else:
                act = idx2act[index_act]
            
            # Add to Most-likely:
            ml_list.append({"Activity": act})
                        
            # Update Prefix Most Likely
            cats_pref_clone[0] = torch.cat([cats_pref_clone[0][:, 1:], torch.tensor([[index_act]])], dim=1)
            if i < len(true_acts):
                cats_pref_clone[1] = torch.cat([cats_pref_clone[1][:, 1:], torch.tensor([[true_ress[i]]])], dim=1)
                nums_pref_clone[0] = torch.cat([nums_pref_clone[0][:, 1:], torch.tensor([[true_nums[i]]])], dim=1)
                
        most_likely = ml_list
        
        # RANDOM SAMPLING
        samples_lists = []
        for _ in range(global_samples_per_case):
            cats_pref_clone_samples = [t.clone() for t in cats_pref]
            nums_pref_clone_samples = [t.clone() for t in nums_pref]
            # Iterate through window size - pref len:
            samples = []
            for i in range(len(cats_pref[0][0])-prefix_length):
                # Predictions
                act_probs_sample = global_model((cats_pref_clone_samples, nums_pref_clone_samples)).squeeze(0)              
                # Random Sampling:
                random_index_act = torch.multinomial(act_probs_sample, num_samples=1).item()    
                
                # NaN is predicted new value at position 0
                if random_index_act == 0:
                   act = 'NaN'
                
                # Stop the suffix creation if EOS is predicted
                elif idx2act[random_index_act] == 'EOS':
                    break
                
                else:
                    act = idx2act[random_index_act]
                
                samples.append({"Activity": act})
                
                # Update Prefix Most Likely
                cats_pref_clone_samples[0] = torch.cat([cats_pref_clone_samples[0][:, 1:], torch.tensor([[random_index_act]])], dim=1)
                if i < len(true_acts):
                    cats_pref_clone_samples[1] = torch.cat([cats_pref_clone_samples[1][:, 1:], torch.tensor([[true_ress[i]]])], dim=1)
                    nums_pref_clone_samples[0] = torch.cat([nums_pref_clone_samples[0][:, 1:], torch.tensor([[true_nums[i]]])], dim=1)
            
            samples_lists.append(samples)
            
        random_suffixes = samples_lists

        results.append((case_name, prefix_length, prefix_prep, random_suffixes, target, most_likely))

    return results


@torch.no_grad()
def _evaluate_predefined_pair(case_name: str,
                               prefix_len: int,
                               prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                               suffix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                               concept_name_id: int,
                               ):
    """
    Evaluate a single predefined prefix-suffix pair.
    
    Args:
        case_name: Name of the case
        prefix_len: Length of the prefix
        prefix: Tuple of (cats_prefix, nums_prefix) - lists of tensors with shape (1, window_size)
        suffix: Tuple of (cats_suffix, nums_suffix) - lists of tensors containing the true suffix
        concept_name_id: ID of the concept name (Activity) in categorical features
        
    Returns:
        Tuple of (case_name, prefix_length, prefix_prep, random_suffixes, target, most_likely) or None if invalid
    """
    cats_pref, nums_pref = prefix
    cats_suffix, nums_suffix = suffix
    
    # Denormalization values for numerical variables:
    mean_s, std_s = global_scaler_params
    
    act2idx, res2idx = global_dict_cat_class_id  # expect tuple of two dicts
    # Invert them:
    idx2act = {ix: name for name, ix in act2idx.items()}
    idx2res = {ix: name for name, ix in res2idx.items()}

    # prefix_prep
    acts = cats_pref[0][0].tolist()
    ress = cats_pref[1][0].tolist()
    times = nums_pref[0][0].tolist()
    # Build the prefix
    prefix_prep = [{"Activity": idx2act[a], "Resource": idx2res[r], "case_elapsed_time": t * std_s + mean_s} 
                  for a, r, t in zip(acts, ress, times) if a != 0]

    # Extract true target from suffix
    # The suffix tensors have shape (1, window_size), we need to extract non-zero values
    cats_suffix_act = cats_suffix[0]  # Activity tensor from suffix, shape (1, window_size)
    
    # Get non-zero activity indices from suffix
    # Flatten to 1D for easier processing
    suffix_act_flat = cats_suffix_act[0]  # shape (window_size,)
    non_zero_ids = (suffix_act_flat != 0).nonzero(as_tuple=True)[0]
    
    if len(non_zero_ids) == 0:
        return None  # Skip if no valid activities in suffix
    
    # Extract true activities, resources, and times from suffix
    # The suffix might contain the full window, so we take only the non-zero parts
    true_acts = suffix_act_flat[non_zero_ids].tolist()
    
    # Get resources and times from suffix if available
    if len(cats_suffix) > 1:
        suffix_res_flat = cats_suffix[1][0]  # shape (window_size,)
        true_ress = suffix_res_flat[non_zero_ids].tolist()
    else:
        true_ress = []
    
    if len(nums_suffix) > 0:
        suffix_num_flat = nums_suffix[0][0]  # shape (window_size,)
        true_nums = suffix_num_flat[non_zero_ids].tolist()
    else:
        true_nums = []
    
    # Build target as list of dicts (exclude EOS)
    target = [{"Activity": idx2act[a]} for a in true_acts if idx2act.get(a, "") != "EOS"]
    if target == []:
        return None

    # MOST LIKELY
    cats_pref_clone = [t.clone() for t in cats_pref]
    nums_pref_clone = [t.clone() for t in nums_pref]
    ml_list = []
    
    max_iterations = len(cats_pref[0][0]) - prefix_len
    
    # Iterate through window size - pref len:
    for i in range(max_iterations):
        # Predictions
        act_probs = global_model((cats_pref_clone, nums_pref_clone))
        # Index of most likely prediction
        index_act = act_probs.argmax(dim=-1).item()
        
        # NaN is predicted new value at position 0
        if index_act == 0:
            act = 'NaN'
        
        # Stop the suffix creation if EOS is predicted
        elif idx2act[index_act] == 'EOS':
            break
        
        else:
            act = idx2act[index_act]
        
        # Add to Most-likely:
        ml_list.append({"Activity": act})
                    
        # Update Prefix Most Likely
        cats_pref_clone[0] = torch.cat([cats_pref_clone[0][:, 1:], torch.tensor([[index_act]])], dim=1)
        if i < len(true_acts):
            cats_pref_clone[1] = torch.cat([cats_pref_clone[1][:, 1:], torch.tensor([[true_ress[i]]])], dim=1)
            nums_pref_clone[0] = torch.cat([nums_pref_clone[0][:, 1:], torch.tensor([[true_nums[i]]])], dim=1)
            
    most_likely = ml_list
    
    # RANDOM SAMPLING
    samples_lists = []
    for _ in range(global_samples_per_case):
        cats_pref_clone_samples = [t.clone() for t in cats_pref]
        nums_pref_clone_samples = [t.clone() for t in nums_pref]
        # Iterate through window size - pref len:
        samples = []
        for i in range(len(cats_pref[0][0])-prefix_len):
            # Predictions
            act_probs_sample = global_model((cats_pref_clone_samples, nums_pref_clone_samples)).squeeze(0)              
            # Random Sampling:
            random_index_act = torch.multinomial(act_probs_sample, num_samples=1).item()    
            
            # NaN is predicted new value at position 0
            if random_index_act == 0:
               act = 'NaN'
            
            # Stop the suffix creation if EOS is predicted
            elif idx2act[random_index_act] == 'EOS':
                break
            
            else:
                act = idx2act[random_index_act]
            
            samples.append({"Activity": act})
            
            # Update Prefix Most Likely
            cats_pref_clone_samples[0] = torch.cat([cats_pref_clone_samples[0][:, 1:], torch.tensor([[random_index_act]])], dim=1)
            if i < len(true_acts):
                cats_pref_clone_samples[1] = torch.cat([cats_pref_clone_samples[1][:, 1:], torch.tensor([[true_ress[i]]])], dim=1)
                nums_pref_clone_samples[0] = torch.cat([nums_pref_clone_samples[0][:, 1:], torch.tensor([[true_nums[i]]])], dim=1)
        
        samples_lists.append(samples)
        
    random_suffixes = samples_lists

    return (case_name, prefix_len, prefix_prep, random_suffixes, target, most_likely)


def evaluate_with_predefined_prefixes(model,
                                      dataset,
                                      predefined_pairs: Dict[Tuple[str, int], Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]], 
                                                                                    Tuple[List[torch.Tensor], List[torch.Tensor]]]],
                                      device,
                                      samples_per_case: int = 20,
                                      random_order: Optional[bool] = False,
                                      ):
    """
    Evaluate using predefined prefix-suffix pairs instead of generating them from cases.
    
    Args:
        model: The Camargo LSTM model
        dataset: The dataset (needed for encoder_decoder and categories)
        predefined_pairs: Dictionary mapping (case_name, prefix_len) -> (prefix, suffix)
                         where prefix and suffix are tuples of (cats, nums) - lists of tensors
                         Each tensor has shape (1, window_size)
        device: Device to run evaluation on
        samples_per_case: Number of samples to generate per case (default=20)
        random_order: Whether to randomize case order
        
    Yields:
        Tuple of (case_name, prefix_length, prefix, sampled_suffixes, target, mean_prediction)
    """
    
    # Move models to CPU
    model.to('cpu')
    
    # Category names and ids
    concept_name = 'Activity'
    # Id of activity in cat list
    concept_name_id = [i for i, cat in enumerate(dataset.all_categories[0]) if cat[0] == concept_name][0]
    
    # Dict with key: act class, value: index position
    act_classes_id = dataset.all_categories[0][0][2]
    # Dict with key: res class, value: index position
    res_classes_id = dataset.all_categories[0][1][2]
    
    # Tuple of category (e.g., Activity) with amount classes, dict with class and index
    cat_categories, _ = model.data_set_categories
    
    # Scaler used in dataset to normalize/ denormalize the numerical attributes:
    scaler = dataset.encoder_decoder.continuous_encoders['case_elapsed_time']
    scaler_params = (scaler.mean_.item(), scaler.scale_.item())
    
    # Initialize globals for identical logic
    init_worker(model, samples_per_case, cat_categories, scaler_params, (act_classes_id, res_classes_id))
    
    # Get items from predefined pairs
    items = list(predefined_pairs.items())
    if random_order:
        items = random.sample(items, len(items))
    
    for (case_name, prefix_len), (prefix, suffix) in tqdm(items, total=len(predefined_pairs)):
        result = _evaluate_predefined_pair(
            case_name=case_name,
            prefix_len=prefix_len,
            prefix=prefix,
            suffix=suffix,
            concept_name_id=concept_name_id,
        )
        
        if result is not None:
            yield result


def evaluate_seq_processing(model,
                            dataset,
                            device,
                            samples_per_case: int = 20,
                            random_order: Optional[bool] = False,
                            ):
    """
    Sequential evaluation yielding tuples per case and prefix length.
    
    Args:
        model: The Camargo LSTM model
        dataset: The dataset to evaluate
        device: Device to run evaluation on
        samples_per_case: Number of samples to generate per case (default=20)
        random_order: Whether to randomize case order
        
    Yields:
        Tuple of (case_name, prefix_length, prefix, sampled_suffixes, target, mean_prediction)
    """
    
    # Move models to CPU
    model.to('cpu')
    
    # Category names and ids
    concept_name = 'Activity'
    # Id of activity in cat list
    concept_name_id = [i for i, cat in enumerate(dataset.all_categories[0]) if cat[0] == concept_name][0]
    
    # Dict with key: act class, value: index position
    act_classes_id = dataset.all_categories[0][0][2]
    # Dict with key: res class, value: index position
    res_classes_id = dataset.all_categories[0][1][2]
    
    # Id of EOS token in activity
    eos_value = 'EOS'
    # index of EOS value in activity dict:
    eos_id = [v for k, v in dataset.all_categories[0][concept_name_id][2].items() if k == eos_value][0]
    
    cases = {}
    for event in dataset:
        # Get suffix being the last 
        suffix = event[0][concept_name_id][-dataset.encoder_decoder.min_suffix_size:]
        if torch.all(suffix == eos_id).item():
            cases[event[2]] = event
            
    case_items = list(cases.items())
    if random_order:
        case_items = random.sample(case_items, len(case_items))
    
    # Tuple of category (e.g., Activity) with amount classes, dict with class and index
    cat_categories, _ = model.data_set_categories
    
    # Scaler used in dataset to normalize/ denormalize the numerical attributes:
    scaler = dataset.encoder_decoder.continuous_encoders['case_elapsed_time']
    scaler_params = (scaler.mean_.item(), scaler.scale_.item())
    
    # Initialize globals for identical logic
    init_worker(model, samples_per_case, cat_categories, scaler_params, (act_classes_id, res_classes_id))
    
    for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(cases)):
        
        # Get a list with the results for all cases of one case:
        results = _evaluate_case(case_name=case_name,
                                full_case=full_case,
                                concept_name_id=concept_name_id,
                                min_suffix_size=dataset.encoder_decoder.min_suffix_size)
        
        # Return the results for inserting:
        for res in results:
            yield res
