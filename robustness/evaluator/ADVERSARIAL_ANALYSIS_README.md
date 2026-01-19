# Adversarial Sample Analysis

## Overview

This enhancement adds the ability to identify and analyze the most effective adversarial samples in your robustness evaluation results. It measures how much the model's predictions change when prefixes are perturbed, helping you understand which attacks are most effective.

## What Was Added

### 1. New Metric: `prediction_shift_dls`

**File:** `robustness_metrics.py`

Added calculation of DLS (Damerau-Levenshtein Similarity) between clean and perturbed predictions:
- Measures how different the model's predictions are when the prefix is perturbed
- Lower DLS = bigger prediction shift = more adversarial effect
- Stored in each entry as `entry['prediction_shift_dls']`

### 2. New Module: `adversarial_sample_selector.py`

**Main Function:** `get_best_adversarial_prefixes()`

This function:
- Sorts all samples by `prediction_shift_dls` (ascending)
- Identifies the top N most adversarial samples
- Analyzes what perturbations were applied
- Prints detailed formatted output
- Returns structured data for further analysis

## Usage

### Basic Usage

```python
from adversarial_sample_selector import get_best_adversarial_prefixes
from robustness_metrics import load_results, prepare_robustness_results

# Load and prepare results
results = load_results('path/to/robustness_results.pkl')
results = prepare_robustness_results(results, save_path='path/to/robustness_results.pkl')

# Get top 10 most adversarial samples
best_adversaries = get_best_adversarial_prefixes(
    results, 
    top_n=10,
    concept_name='Activity'
)
```

### Output Format

The function prints a detailed analysis like this:

```
================================================================================
Top 10 Most Effective Adversarial Samples
================================================================================

Lower DLS = Bigger prediction shift = More adversarial effect
================================================================================

#1. Case: 3456, Prefix Length: 5
    Prediction Shift DLS: 0.1234
    
    Perturbations Applied:
      • Event 2:
          - Activity: 'Assign Seriousness' → 'Close Request'
      • Event 3:
          - case_elapsed_time: 3600.00 → 7200.00
    
    Clean Prediction:     [Take in Charge Ticket, Resolve (Phone), Close, ...]
    Perturbed Prediction: [Insert Ticket, Wait, Assign Seriousness, ...]

--------------------------------------------------------------------------------
...
```

### Return Value

The function returns a list of tuples, each containing:

```python
(
    case_name,              # str: Case identifier
    prefix_len,             # int: Length of prefix
    prediction_shift_dls,   # float: DLS between predictions
    prefix_orig,            # List[Dict]: Original prefix events
    prefix_pert,            # List[Dict]: Perturbed prefix events
    mean_orig,              # List[Dict]: Clean prediction
    mean_pert,              # List[Dict]: Perturbed prediction
    perturbations           # List[Dict]: Identified perturbations
)
```

## Integration with Existing Workflow

The new functionality integrates seamlessly with your existing robustness evaluation workflow:

1. **Run your robustness evaluation** (as before)
2. **Load results** using `load_results()`
3. **Prepare results** using `prepare_robustness_results()` - now includes `prediction_shift_dls`
4. **Analyze adversarial samples** using `get_best_adversarial_prefixes()`

## Example in Jupyter Notebook

See the new cells added to `compare_robustness_models.ipynb`:

```python
from adversarial_sample_selector import get_best_adversarial_prefixes

# Analyze U-ED-LSTM model
best_adversarial_samples = get_best_adversarial_prefixes(
    U_ED_LSTM_results, 
    top_n=10,
    concept_name='Activity'
)
```

## Understanding the Results

### Prediction Shift DLS

- **Range:** 0.0 to 1.0
- **Lower values** = More adversarial effect (bigger prediction change)
- **Higher values** = Less adversarial effect (predictions remain similar)
- **0.0** = Completely different predictions
- **1.0** = Identical predictions

### Perturbation Types

The analysis identifies several types of perturbations:
- **Activity changes:** When an event's activity is modified
- **Attribute changes:** When numerical attributes (like time) are changed
- **Structural changes:** When events are added or removed

## Advanced Usage

### Custom Analysis

You can process the returned data for custom analysis:

```python
best_adversaries = get_best_adversarial_prefixes(results, top_n=50)

# Extract only the DLS values
dls_values = [adv[2] for adv in best_adversaries]

# Find samples with specific perturbation types
for case_name, prefix_len, dls, _, _, _, _, perturbations in best_adversaries:
    for pert in perturbations:
        if any('Activity' in change for change in pert['changes']):
            print(f"Case {case_name}: Activity perturbation with DLS={dls:.4f}")
```

### Filtering by Prefix Length

```python
# Get all samples
all_adversaries = get_best_adversarial_prefixes(results, top_n=1000)

# Filter for specific prefix length
long_prefix_adversaries = [
    adv for adv in all_adversaries 
    if adv[1] >= 10  # prefix_len >= 10
]
```

## Files Modified/Created

1. **Modified:** `robustness/evaluator/robustness_metrics.py`
   - Added `prediction_shift_dls` calculation in `prepare_robustness_results()`

2. **Created:** `robustness/evaluator/adversarial_sample_selector.py`
   - `get_best_adversarial_prefixes()`: Main function
   - `_identify_perturbations()`: Compares original vs perturbed prefixes
   - `_extract_activity_sequence()`: Extracts activity sequences
   - `_print_adversarial_analysis()`: Formats and prints results

3. **Updated:** `robustness/evaluator/compare_robustness_models.ipynb`
   - Added example cells demonstrating the new functionality

## Benefits

1. **Identify weak points:** Find which perturbations most effectively fool your model
2. **Understand attack patterns:** See what types of changes cause the biggest prediction shifts
3. **Compare models:** Analyze which model is more robust to specific attack types
4. **Targeted improvements:** Focus defensive efforts on the most vulnerable scenarios

## Next Steps

- Run the analysis on your existing robustness results
- Compare adversarial effectiveness across different models
- Use insights to improve model robustness
- Investigate why certain prefixes are more susceptible to attacks
