"""
Weytjens LSTM Remaining-Time Evaluation Module

Provides functions to evaluate the Weytjens StochasticLSTM model for robustness
analysis. Adapted from camargo_evaluation.py to support remaining-time prediction
instead of suffix activity prediction.

The model predicts case_elapsed_time (remaining time to case completion).
Two model instances are used:
- model (p_fix=0.05): for Monte Carlo sampling (uncertainty estimation)
- model_without_drop (p_fix=0): deterministic prediction (all neurons active)
"""

import torch
import random
from typing import Optional, Tuple, List, Dict, Any
from tqdm.notebook import tqdm


# Global placeholders for worker state
global_model = None
global_model_without_drop = None
global_samples_per_case = None
global_scaler_params = None   # (mean, std) for case_elapsed_time denormalization
global_act_categories = None  # cat_categories from model.data_set_categories[0]


def init_worker(model,
                model_without_drop,
                samples_per_case: int,
                act_categories,
                scaler_params: Tuple[float, float]):
    """
    Initialize global state for evaluation (mirrors multiprocessing init pattern).

    Args:
        model: StochasticLSTMWeytjens with dropout (p_fix=0.05) for MC sampling
        model_without_drop: StochasticLSTMWeytjens with p_fix=0 for deterministic prediction
        samples_per_case: Number of MC samples to draw per prefix
        act_categories: List of categorical feature metadata tuples from model.data_set_categories[0]
        scaler_params: (mean, std) from the case_elapsed_time StandardScaler
    """
    global global_model, global_model_without_drop, global_samples_per_case
    global global_scaler_params, global_act_categories

    model.eval()
    model_without_drop.eval()

    global_model = model
    global_model_without_drop = model_without_drop
    global_samples_per_case = samples_per_case
    global_scaler_params = scaler_params
    global_act_categories = act_categories


@torch.no_grad()
def _evaluate_predefined_pair(case_name: str,
                               prefix_len: int,
                               prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                               suffix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                               concept_name_id: int,
                               ) -> Optional[Tuple]:
    """
    Evaluate a single predefined prefix-suffix pair for remaining-time prediction.

    The Weytjens model predicts case_elapsed_time (total remaining time) directly
    from the prefix — no autoregressive generation needed. The target is the true
    case_elapsed_time at the end of the case (last non-padding value in suffix nums).

    Args:
        case_name: Case identifier
        prefix_len: Number of real events in the prefix (excluding padding)
        prefix: (cats_prefix, nums_prefix) — lists of tensors, shape (1, window_size)
        suffix: (cats_suffix, nums_suffix) — lists of tensors for the true suffix
        concept_name_id: Index of concept:name in the cat feature list (always 0)

    Returns:
        Tuple (case_name, prefix_len, prefix_prep, mc_samples, target, most_likely)
        or None if the suffix contains no valid activities
    """
    cats_pref, nums_pref = prefix
    cats_suffix, nums_suffix = suffix

    mean_s, std_s = global_scaler_params

    # Build human-readable prefix: one dict per non-padding event
    act_categories = global_act_categories[0][2]  # {name: idx} for concept:name
    prefix_prep = []
    seq_len = cats_pref[0].shape[1]
    for pos in range(seq_len):
        cat_val = cats_pref[concept_name_id][0, pos].item()
        if cat_val == 0:
            continue
        act_name = next((k for k, v in act_categories.items() if v == cat_val), None)
        num_val = nums_pref[0][0, pos].item() * std_s + mean_s
        prefix_prep.append({'concept:name': act_name, 'case_elapsed_time': num_val})

    # Extract true suffix activities (to verify there is a valid suffix)
    suffix_act_flat = cats_suffix[concept_name_id][0]
    non_zero_ids = (suffix_act_flat != 0).nonzero(as_tuple=True)[0]

    if len(non_zero_ids) == 0:
        return None

    true_acts = suffix_act_flat[non_zero_ids].tolist()
    # Build target activity list (excluding EOS) — needed by robustness_metrics
    eos_idx = next((v for k, v in act_categories.items() if k == 'EOS'), None)
    target_acts = [{'concept:name': next((k for k, v in act_categories.items() if v == a), 'NaN')}
                   for a in true_acts if a != eos_idx]

    # True remaining time: last non-zero value in suffix numerics (the EOS row holds the
    # final case_elapsed_time, which equals the total case duration)
    suffix_nums_flat = nums_suffix[0][0]  # shape (window_size,)
    non_zero_num_ids = (suffix_nums_flat != 0).nonzero(as_tuple=True)[0]
    if len(non_zero_num_ids) == 0:
        return None
    raw_target = suffix_nums_flat[non_zero_num_ids[-1]].item()
    target_cet = raw_target * std_s + mean_s
    target = [{'case_elapsed_time': target_cet}]

    # Monte Carlo samples using the stochastic model (dropout kept on)
    mc_samples = []
    for _ in range(global_samples_per_case):
        mean_pred, logvar_pred = global_model(input=prefix)
        mean_pred = mean_pred.squeeze(0)
        std_pred = torch.exp(0.5 * logvar_pred).squeeze(0)
        sample = torch.normal(mean=mean_pred, std=std_pred)
        sample_val = torch.clamp(sample * std_s + mean_s, min=0.0).item()
        mc_samples.append([{'case_elapsed_time': sample_val}])

    # Deterministic prediction using model without dropout
    mean_det, _ = global_model_without_drop(input=prefix)
    mean_det = mean_det.squeeze(0)
    det_val = torch.clamp(mean_det * std_s + mean_s, min=0.0).item()
    most_likely = [{'case_elapsed_time': det_val}]

    return (case_name, prefix_len, prefix_prep, mc_samples, target, most_likely)


def evaluate_with_predefined_prefixes(model,
                                      model_without_drop,
                                      dataset,
                                      predefined_pairs: Dict[Tuple[str, int],
                                                             Tuple[Tuple[List[torch.Tensor], List[torch.Tensor]],
                                                                   Tuple[List[torch.Tensor], List[torch.Tensor]]]],
                                      device,
                                      samples_per_case: int = 100,
                                      random_order: Optional[bool] = False,
                                      concept_name: str = 'concept:name',
                                      ):
    """
    Evaluate the Weytjens model using predefined prefix-suffix pairs.

    Mirrors the interface of camargo_evaluation.evaluate_with_predefined_prefixes so
    that robustness notebooks can follow the same pattern. The result 6-tuple format
    is identical, but predictions contain remaining-time values instead of activity
    sequences.

    Args:
        model: StochasticLSTMWeytjens with p_fix=0.05 (used for MC sampling)
        model_without_drop: StochasticLSTMWeytjens with p_fix=0 (deterministic)
        dataset: EventLogDataset (needed for encoder_decoder and categories)
        predefined_pairs: Dict mapping (case_name, prefix_len) -> (prefix, suffix)
                          where prefix/suffix are (cats, nums) tuples of tensor lists
        device: Torch device (evaluation always runs on CPU)
        samples_per_case: Number of MC samples per prefix (default=100)
        random_order: Whether to shuffle evaluation order
        concept_name: Name of the activity column (default='concept:name')

    Yields:
        Tuple (case_name, prefix_length, prefix_prep, mc_samples, target, most_likely)
    """
    # Move both models to CPU
    model.to('cpu')
    model_without_drop.to('cpu')

    # Resolve concept:name position in cat feature list
    cat_feat_names = model.model_feat[0]
    concept_name_id = cat_feat_names.index(concept_name)

    # Categorical metadata for decoding activity names
    cat_categories, _ = model.data_set_categories

    # Scaler for denormalizing case_elapsed_time
    scaler = dataset.encoder_decoder.continuous_encoders['case_elapsed_time']
    scaler_params = (scaler.mean_.item(), scaler.scale_.item())

    init_worker(model, model_without_drop, samples_per_case, cat_categories, scaler_params)

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
                            model_without_drop,
                            dataset,
                            device,
                            samples_per_case: int = 100,
                            random_order: Optional[bool] = False,
                            concept_name: str = 'concept:name',
                            ):
    """
    Sequential evaluation over all cases in a dataset, yielding one result per prefix.

    Iterates through every valid case (those ending with EOS in the suffix window)
    and every valid prefix length within each case.

    Args:
        model: StochasticLSTMWeytjens with p_fix=0.05 (MC sampling)
        model_without_drop: StochasticLSTMWeytjens with p_fix=0 (deterministic)
        dataset: EventLogDataset
        device: Torch device
        samples_per_case: Number of MC samples per prefix
        random_order: Whether to shuffle case order
        concept_name: Name of the activity column

    Yields:
        Tuple (case_name, prefix_length, prefix_prep, mc_samples, target, most_likely)
    """
    model.to('cpu')
    model_without_drop.to('cpu')

    cat_feat_names = model.model_feat[0]
    concept_name_id = cat_feat_names.index(concept_name)

    # Resolve EOS token id for filtering valid cases
    dataset_cat_map = {cat[0]: cat[2] for cat in dataset.all_categories[0]}
    eos_id = dataset_cat_map[concept_name]['EOS']

    # Filter to cases where the suffix window is all EOS (complete cases)
    cases = {}
    for event in dataset:
        suffix_window = event[0][concept_name_id][-dataset.encoder_decoder.min_suffix_size:]
        if torch.all(suffix_window == eos_id).item():
            cases[event[2]] = event

    case_items = list(cases.items())
    if random_order:
        case_items = random.sample(case_items, len(case_items))

    cat_categories, _ = model.data_set_categories
    scaler = dataset.encoder_decoder.continuous_encoders['case_elapsed_time']
    scaler_params = (scaler.mean_.item(), scaler.scale_.item())

    init_worker(model, model_without_drop, samples_per_case, cat_categories, scaler_params)

    mean_s, std_s = scaler_params
    act_categories = cat_categories[0][2]  # {name: idx} for concept:name
    eos_cat_idx = act_categories.get('EOS')

    for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(cases)):
        cats, nums, _ = full_case
        seq_len = cats[0].shape[0]
        min_suffix_size = dataset.encoder_decoder.min_suffix_size

        # True remaining time: case_elapsed_time at the EOS position
        raw_target = nums[0][-1 - min_suffix_size].item()
        target_cet = raw_target * std_s + mean_s
        target = [{'case_elapsed_time': target_cet}]

        # Build sliding prefix window
        current_prefix = (
            [torch.zeros_like(cat_attr).unsqueeze(0) for cat_attr in cats],
            [torch.zeros_like(num_attr).unsqueeze(0) for num_attr in nums],
        )

        prefix_length = 0
        for i in range(seq_len - min_suffix_size - 1):
            for j in range(len(current_prefix[0])):
                current_prefix[0][j][0] = torch.roll(current_prefix[0][j][0], -1)
                current_prefix[0][j][0, -1] = cats[j][i]
            for j in range(len(current_prefix[1])):
                current_prefix[1][j][0] = torch.roll(current_prefix[1][j][0], -1)
                current_prefix[1][j][0, -1] = nums[j][i]

            if prefix_length == 0 and cats[concept_name_id][i].item() == 0:
                continue
            prefix_length += 1

            # Build human-readable prefix
            prefix_prep = []
            for pos in range(current_prefix[0][0].shape[1]):
                cat_val = current_prefix[0][concept_name_id][0, pos].item()
                if cat_val == 0:
                    continue
                act_name = next((k for k, v in act_categories.items() if v == cat_val), None)
                num_val = current_prefix[1][0][0, pos].item() * std_s + mean_s
                prefix_prep.append({'concept:name': act_name, 'case_elapsed_time': num_val})

            # MC samples
            mc_samples = []
            for _ in range(samples_per_case):
                with torch.no_grad():
                    mean_pred, logvar_pred = global_model(input=current_prefix)
                mean_pred = mean_pred.squeeze(0)
                std_pred = torch.exp(0.5 * logvar_pred).squeeze(0)
                sample = torch.normal(mean=mean_pred, std=std_pred)
                sample_val = torch.clamp(sample * std_s + mean_s, min=0.0).item()
                mc_samples.append([{'case_elapsed_time': sample_val}])

            # Deterministic prediction
            with torch.no_grad():
                mean_det, _ = global_model_without_drop(input=current_prefix)
            mean_det = mean_det.squeeze(0)
            det_val = torch.clamp(mean_det * std_s + mean_s, min=0.0).item()
            most_likely = [{'case_elapsed_time': det_val}]

            yield (case_name, prefix_length, prefix_prep, mc_samples, target, most_likely)
