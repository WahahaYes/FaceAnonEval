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

## Gallery Scaling Analysis Patterns

### Embedding and Identity Discovery

**Critical Pattern**: When working with pre-computed embeddings and identity mappings:

1. **Embedding Key Structure**: 
   - Format: `{dataset_name}/{image_filename}.jpg`
   - Example: `CelebA_test/000001.jpg` (not just `000001.jpg`)
   - **Common Pitfall**: Missing dataset prefix causes "No embeddings found" errors

2. **Identity Lookup Usage**:
   ```python
   from src.dataset.celeba_identity_lookup import CelebAIdentityLookup
   
   # Initialize with dataset name
   identity_lookup = CelebAIdentityLookup("CelebA_test")
   
   # Map embedding keys to identities
   identity = identity_lookup.get_identity(embedding_key)
   # Returns: identity label (e.g., "0001")
   ```

3. **Gallery Construction Pattern**:
   ```python
   # Build identity-to-images mapping
   identity_to_images = {}
   for embedding_key in embeddings.keys():
       identity = identity_lookup.get_identity(embedding_key)
       if identity not in identity_to_images:
           identity_to_images[identity] = []
       identity_to_images[identity].append(embedding_key)
   
   # Sample gallery ensuring different image of same identity
   def sample_gallery_by_identity(query_identity, query_key, gallery_size):
       # Always include query identity with DIFFERENT image
       # Exclude query image itself from selection
       # Randomly sample remaining identities
   ```

### Common Debugging Patterns

**Error**: "No gallery embeddings found for identities"
- **Cause**: Mismatch between embedding keys and identity lookup
- **Fix**: Verify embedding key format includes dataset prefix
- **Debug**: Print sample keys from both embeddings and identity lookup

**Error**: "Rank calculation returns all zeros"
- **Cause**: Incorrect rank-to-accuracy conversion
- **Fix**: Calculate per-query accuracy first, then average
- **Pattern**: `accuracy = sum(rank == 1) / total_queries`

**Error**: "Gallery doesn't contain query identity"
- **Cause**: Random sampling without identity constraints
- **Fix**: Always include query identity with different image
- **Pattern**: Force inclusion of query identity in gallery construction

### File Location Patterns

**Embeddings**: `Embeddings/{dataset_name}/{privacy_mechanism}/`
**Identity Mappings**: `src/dataset/celeba_identity_lookup.py`
**Results**: `Results/Privacy/{evaluation_method}/`

### Evaluation Workflow

1. **Load embeddings** with correct key format
2. **Initialize identity lookup** with matching dataset name
3. **Map identities** to available images
4. **Construct galleries** with identity constraints
5. **Calculate ranks** per query (not aggregate)
6. **Convert to accuracy** after rank calculation
7. **Average across trials** for final results
