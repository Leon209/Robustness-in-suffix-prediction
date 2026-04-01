# Robustness Evaluation Framework for Suffix Prediction of Business Processes

A framework for generating a range of attacks to empirically evaluate the robustness of a given model.

## Project Structure

```
├── data/                # Original datasets / event logs
├── perturbed_data/      # Train/val/test CSVs and perturbed test sets (BPIC17, BPIC20, helpdesk, sepsis)
├── encoded_data/        #  encoded inputs perturbed and clean data
├── evaluation_results/  # Saved predictions and experiment outputs
├── img/                 # Figures and outputs from experiments
├── ml_models/           # Benchmark models (U-ED-LSTM and Camargo LSTM)
│   ├── model/dropout_uncertainty_enc_dec_LSTM/
│   ├── notebooks/       # evaluation_run_notebooks/, training_variational_dropout/, encode_adv_dataset/, adv_retraining/, …
│   └── reimplemented_comparable_approaches/
│       ├── camargo_LSTM_suffix_pred/
│       └── weytjens_unc_rem_time/
├── robustness/          # Robustness evaluation framework
│   ├── perturbator/     # Generates perturbed datasets (Last Event Attack, All Events Attack, ...)
│   │   ├── perturbation_logic/  # activity_pertubator.py, structural_attacks.py, feature_attacks.py, …
│   │   ├── event_log_loader_service/event_log_loader.py
│   │   ├── helpdesk/generate_perturbations.ipynb  # also BPIC17/, BPIC20/, sepsis/
│   │   └── visualize_perturbations.ipynb
│   └── evaluator/       # Evaluates the evaluation_results
│       ├── automated_compare_robustness_models.ipynb          # Main notebook for the creation of robustness charts
│       ├── adversarial_retraining_asr.ipynb
│       ├── robustness_charts.py
│       ├── adversarial_sample_selector.py
│       └── robustness_metrics.py
```

## How to run the project

### Create the virtual environment

```bash
pipenv install
```

### Activate the virtual environment

```bash
pipenv shell
```

### Run the project

Inside the virtual environment you have the Python packages needed to run the notebooks and scripts.

1. **Download the datasets.** The download links are given in the thesis. Place the files in the `data/` folder.

2. **Generate attacks** with the robustness perturbator (e.g. `robustness/perturbator/<dataset>/generate_perturbations.ipynb`). Perturbed logs are written in an unencoded form under `perturbed_data/`.

3. **Encode the data** using the notebooks under `ml_models/notebooks/encode_adv_dataset/` so that train/validation/test and adversarial splits match what your models expect.

4. **Run evaluation** to produce predictions on clean and perturbed data—for example, for Helpdesk, open and run `ml_models/notebooks/evaluation_run_notebooks/normal_4layer/Helpdesk/robustness_evaluation.ipynb` (other logs have the same layout under `normal_4layer/<Dataset>/`).

5. **Visualize results** with `robustness/evaluator/automated_compare_robustness_models.ipynb`, which builds the robustness comparison charts from the saved outputs under `evaluation_results/`.

