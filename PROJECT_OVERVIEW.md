# FaceAnonEval Project Overview

## Project Purpose
FaceAnonEval is a comprehensive framework for generating and evaluating privacy-preserving neural face synthesis algorithms. The project implements various face anonymization mechanisms and evaluates their effectiveness against facial recognition and demographic classification systems.

## Core Architecture

### 1. Dataset Processing Pipeline
- **Input**: Raw face datasets (CelebA, LFW)
- **Processing**: Apply privacy mechanisms via `process_dataset.py`
- **Output**: Anonymized datasets stored in `Anonymized Datasets/`

### 2. Privacy Mechanisms (`src/privacy_mechanisms/`)
- **dtheta_privacy**: AvatarLDP and AvatarRotation mechanisms (Wilson et al. 2025)
- **gaussian_blur/uniform_blur**: Blur-based anonymization
- **identity_dp**: Identity Differential Privacy using SimSwap
- **metric_privacy**: MetricSVD implementation (Fan 2019)
- **pixel_dp**: Pixel Differential Privacy (Fan 2018)
- **simswap**: Face swapping for privacy (Chen et al. 2020)

### 3. Evaluation Framework (`src/evaluation/`)
- **rank_k_evaluation**: Rank-k accuracy assessment
- **validation_evaluation**: Validation/EER metrics
- **lfw_validation_evaluation**: LFW dataset-specific validation
- **utility_evaluation**: Demographic attribute preservation (age, race, gender, emotion)

## Results Structure

### Directory Organization
```
Results/
├── Privacy/                    # Privacy-preserving evaluation results
│   ├── Datasets/              # Dataset-specific baseline results
│   ├── lfw_validation/        # LFW validation results
│   ├── rank_k/               # Rank-k evaluation results
│   └── lfw_validation_backup/ # Backup LFW results
└── Utility/                   # Utility preservation results
    ├── Datasets/             # Dataset utility baselines
    └── utility/              # Utility evaluation results
```

### Results File Format
- **CSV files**: Primary storage format for evaluation results
- **Naming convention**: `{dataset}_{privacy_mechanism_suffix}.csv`
- **Privacy mechanism suffixes**: Include parameters (e.g., `theta0.0_eps1.0`)

## Data Access Patterns

### Primary Access Function
**`src/data_analysis/query_accuracy.py`** - Main interface for querying results

#### Key Functions:
- `query_accuracy()`: Universal query interface for all evaluation types
- `get_results_csv_path()`: Path resolution for result files
- `_query_rank_k()`: Rank-k result processing
- `_query_validation()`: Validation/EER result processing  
- `_query_utility()`: Utility metrics processing

#### Usage Pattern:
```python
from src.data_analysis.query_accuracy import query_accuracy

# Query rank-k accuracy
results = query_accuracy(
    evaluation_method="rank_k",
    dataset="CelebA_test", 
    anonymized_dataset="CelebA_test_dtheta_privacy_theta0.0_eps1.0",
    mode="mean",
    denominator=dataset_size
)

# Query validation accuracy
accuracy = query_accuracy(
    evaluation_method="validation",
    dataset="lfw",
    anonymized_dataset="lfw_dtheta_privacy_theta0.0_eps1.0", 
    mode="mean",
    denominator=lfw_pairs
)
```

### Result File Structure

#### Rank-K Results (`rank_k/`)
- **Columns**: `k` (rank position), `distance` (similarity score)
- **Usage**: Calculate cumulative accuracy at different rank thresholds

#### Validation Results (`lfw_validation/`, `validation/`)
- **Columns**: `real_label`, `result`, `distance`
- **Usage**: Calculate verification accuracy and EER rates

#### Utility Results (`utility/`)
- **Regression metrics**: `ssim`, `age`
- **Classification metrics**: `emotion`, `race`, `gender` (accuracy)

## Key Scripts for Data Analysis

### Jupyter Notebooks
- **`compare_results.ipynb`**: Primary results comparison and visualization
- **`plotting_script.ipynb`**: Advanced plotting and analysis
- **`investigate_embeddings.ipynb`**: Embedding space analysis
- **`generate_face_figures.ipynb`**: Face image generation for figures

### Core Scripts
- **`process_dataset.py`**: Apply privacy mechanisms to datasets
- **`evaluate_mechanism.py`**: Run evaluations on anonymized datasets
- **`plot.py`**: Basic plotting functionality

## Dataset Management

### Supported Datasets
- **CelebA**: Primary evaluation dataset (test subset available)
- **LFW**: Labeled Faces in the Wild for validation

### Dataset Structure
- Raw datasets in `Datasets/`
- Anonymized datasets in `Anonymized Datasets/`
- Results reference dataset names with privacy mechanism suffixes

## Configuration and Arguments

### Argument Parser (`src/argument_parser.py`)
- Handles dataset, privacy mechanism, and evaluation method selection
- Supports batch processing and GPU/CPU configuration
- Manages output paths and overwrite options

### Privacy Mechanism Parameters
- **dtheta_privacy**: `theta` (rotation angle), `eps` (privacy budget)
- **identity_dp**: `eps` (privacy budget)
- **metric_privacy**: `eps`, `k` (parameters)
- **pixel_dp**: `eps`, `b` (block size)

## Experiment Workflow

1. **Dataset Preparation**: Place datasets in `Datasets/` directory
2. **Anonymization**: Run `process_dataset.py` with desired privacy mechanism
3. **Evaluation**: Run `evaluate_mechanism.py` with evaluation method
4. **Analysis**: Use `query_accuracy.py` or notebooks to analyze results
5. **Visualization**: Use plotting notebooks for figure generation

## Key Insights for Follow-up Experiments

- Results are stored in standardized CSV format with consistent naming
- The `query_accuracy()` function provides unified access to all result types
- Privacy mechanism parameters are encoded in filenames for easy identification
- Both privacy (recognition resistance) and utility (attribute preservation) are evaluated
- The framework supports custom datasets and evaluation methods
