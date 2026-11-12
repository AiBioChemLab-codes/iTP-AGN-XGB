# iTP-AGN-XGB

**Therapeutic peptide identification using feature-space Gaussian noise augmentation and XGBoost**

iTP-AGN-XGB combines eight sequence descriptor groups with additive Gaussian noise (AGN) for eight peptide classification tasks. The primary model is a single AGN-enhanced XGBoost classifier; soft-voting ensembles serve as comparators.

This repository includes FASTA datasets, feature extraction, training code and pretrained XGBoost models for the INIT and AGN conditions.

## Installation

Python 3.10 is recommended. Create an environment in the project root and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The supplied models use XGBoost 3.0.5. Training and prediction run on CPU by default.

## Quick Start

Run the following commands from the project root.

### Pretrained Prediction

Evaluate the supplied models on the included test sets:

```bash
python predict.py
```

Outputs are saved to `runs/prediction/`. Select a task, condition and output directory:

```bash
python predict.py --tasks PBP --modes agn --output runs/PBP_agn
```

Use a custom FASTA file with a single task:

```bash
python predict.py --tasks PBP --modes agn --fasta data/fasta/PBP/Test.fasta --output runs/custom_prediction
```

The current prediction interface expects 0/1 labels and both classes, because it also computes evaluation metrics. The prediction output directory must not already exist.

### Training and Soft-voting Comparisons

Run feature extraction, cross-validation selection, final fitting and test evaluation:

```bash
python main.py
```

Outputs are saved to `runs/full/`. Specify another configuration or output directory:

```bash
python main.py --config configs/default.json --run-dir runs/experiment_01
```

A matching run resumes from verified checkpoints. Use a new output directory after changing inputs, configuration, code or dependencies.

`predict.py` loads the supplied models. `main.py` selects parameters and fits new models using the training data. Training outputs depend on the chosen configuration and environment.

## Data Format

`data/fasta/` contains AAP, ABP, ACP, AIP, AVP, CPP, PBP and QSP. Each task has `Train.fasta` and `Test.fasta`.

FASTA headers follow `>sample_id|label`, where `1` denotes the positive class and `0` the negative class:

```fasta
>peptide_001|1
ACDEFGHIK
>peptide_002|0
LMNPQRSTV
```

Sequences must contain at least three residues using the 20 standard amino-acid codes. Training and test files define the evaluation split; cross-validation uses training samples only.

## Training Configuration

Defaults and parameter search spaces are defined in [`configs/default.json`](configs/default.json).

| Setting | Default |
| --- | --- |
| Features | AAC, DPC, ASDC, AAIndex, PAAC, APAAC, CTD and BLOSUM62; 2,125 dimensions |
| Cross-validation | 10 folds; identical training sequences stay in the same fold |
| Selection metric | Mean validation ROC-AUC |
| Scaling | StandardScaler fitted on the current training fold |
| AGN | Zero-mean Gaussian noise with standard deviation 0.01 before scaling |
| Augmentation | 10 copies per training sample; unperturbed samples are not additionally retained |
| Random seed | 42 |
| Classification threshold | 0.5 |
| Single-model conditions | INIT and AGN XGBoost |
| Soft-voting comparisons | 3, 5 or 7 XGBoost members; equal-weight probability averaging |

INIT denotes training without AGN. Validation and test samples are not augmented. After selection, the scaler and model are fitted using all training samples before test evaluation.

## Outputs

Results are generated at runtime and saved under `runs/` by default.

| Command | Main outputs |
| --- | --- |
| Pretrained prediction | `metrics.csv`, `{task}_{mode}_predictions.csv` |
| Training | `models/`, `predictions/`, `cv_summary.csv`, `test_metrics.csv` |
| Comparisons and report | `voting_metrics.csv`, `classifier_equal_task_means.csv`, `figures/`, `REPORT.md` |

Prediction files contain `PID` (sample ID), `Target` (true label), `prob` (positive-class probability) and `pred` (predicted label). Metrics include ACC, MCC, SEN, SPE, PRE, F1, ROC-AUC and AP. AP is average precision; task means use equal weights.

## Repository Structure

```text
.
├── main.py                 # Training and comparisons
├── predict.py              # Pretrained prediction
├── requirements.txt
├── configs/default.json    # Configuration and parameter search spaces
├── data/fasta/             # Task-specific training and test data
├── pretrained/             # XGBoost models and scaling parameters
└── itp_agn_xgb/             # Data processing, descriptors, models and evaluation
```

Display command-line options:

```bash
python main.py --help
python predict.py --help
```
