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
│   ├── perturbator/     # Generates perturbed datasets
│   │   ├── perturbation_logic/  # activity_pertubator.py, structural_attacks.py, feature_attacks.py, …
│   │   ├── event_log_loader_service/event_log_loader.py
│   │   ├── helpdesk/generate_perturbations.ipynb  # also BPIC17/, BPIC20/, sepsis/
│   │   └── visualize_perturbations.ipynb
│   └── evaluator/       # Evaluates the evaluation_results
│       ├── compare_robustness_models.ipynb
│       ├── automated_compare_robustness_models.ipynb          # Main notebook for the creation of robustness charts
│       ├── adversarial_retraining_asr.ipynb
│       ├── robustness_charts.py
│       ├── adversarial_sample_selector.py
│       └── robustness_metrics.py
```





