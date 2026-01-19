# Robustness of deep learning models for suffix prediction 

A framework for generating a range of attacks to evaluate the robustness of a given model.

## Project Structure

```
.
├── backend/             # Flask API server
├── frontend/            # React application
├── robustness/          # Robustness evaluation framework
│   ├── perturbator/     # Generates Perturbed datasets
│   │   ├── activity_pertubator.py
│   │   ├── generate_loop_augmentations.ipynb
│   │   ├── generate_perturbations.ipynb
│   │   └── helpdesk_perturbations.py
│   └── evaluator/       # Evaluates the evaluation_results
│       ├── compare_robustness_models.ipynb
│       └── robustness_metrics.py
├── ml_models/           # Machine learning models and utilities
│   ├── model/
│   │   └── dropout_uncertainty_enc_dec_LSTM/
│   ├── evaluation/
│   │   ├── adversarial_attack.py  # For generating gradient attacks
│   ├── reimplemented_comparable_approaches/
│   │   ├── camargo_LSTM_suffix_pred/
│   │   └── weytjens_unc_rem_time/
│   └── notebooks/
├── data/                 # Original dataset
├── perturbed_data/       # Train/val/test set as csv and a test set with perturbations
├── encoded_data/         # Preprocessed data (debending on ML-Model)
├── evaluation_results/   # Predicitions for the input data

```



## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:3000`



