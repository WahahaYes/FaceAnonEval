# Failure Analysis Experiment Plan

## Objective
Analyze failure cases where the privacy mechanism failed to prevent re-identification in the d_theta eps 1.0 condition. Specifically, we want to examine samples where the TRUE individual was matched to their anonymized likeness.

## Background
From the project overview, we understand that:
- The d_theta_privacy mechanism implements AvatarLDP and AvatarRotation mechanisms
- Results are stored in CSV format with naming convention `{dataset}_{privacy_mechanism_suffix}.csv`
- The d_theta eps 1.0 condition would have files like `CelebA_test_dtheta_privacy_theta0.0_eps1.0.csv`
- Rank-k evaluation results contain columns for `k` (rank position) and `distance` (similarity score)

## Research Questions

1. **What characteristics do failure cases share?**
   - Demographic attributes (age, race, gender)
   - Image quality metrics
   - Embedding space properties

2. **When does the privacy mechanism fail?**
   - Specific image types or features
   - Thresholds of similarity scores
   - Patterns in the embedding space

3. **How can we improve the mechanism?**
   - Identify systematic weaknesses
   - Suggest parameter adjustments
   - Recommend additional safeguards

## Data Sources Needed

### Primary Results
- **Rank-k results**: `Results/Privacy/rank_k/CelebA_test_dtheta_privacy_theta0.0_eps1.0.csv`
- **Validation results**: `Results/Privacy/lfw_validation/lfw_dtheta_privacy_theta0.0_eps1.0.csv`

### Supporting Data
- **Original embeddings**: `Embeddings/CelebA_test/dtheta_privacy_theta0.0_eps1.0/`
- **Identity mappings**: `src/dataset/celeba_identity_lookup.py`
- **Original images**: `Datasets/CelebA_test/`
- **Anonymized images**: `Anonymized Datasets/CelebA_test_dtheta_privacy_theta0.0_eps1.0/`

### Demographic Data (Available)
- **Baseline demographics**: `Results/Utility/Datasets/CelebA_test.csv` (original image attributes)
- **Anonymized demographics**: `Results/Utility/utility/CelebA_test_dtheta_privacy_theta0.0_eps1.0.csv` (privacy mechanism output attributes)
- **Available attributes**: age, race, gender, emotion (both original and anonymized)
- **Image quality**: SSIM scores between original and anonymized images
- **Key mapping**: Both files use `key` column to map to specific images

## Methodology

### Phase 1: Identify Failure Cases
1. Load rank-k results for d_theta eps 1.0 condition
2. Extract samples where rank = 1 (correct re-identification)
3. Map results back to original image identities
4. Create failure case dataset

### Phase 2: Characterize Failures
1. **Extract demographic attributes for failure cases**:
   - Use `key` mapping to join with demographic CSV files
   - Compare original vs. anonymized attributes for each failure
   - Calculate attribute preservation rates (emotion, age, race, gender accuracy)
   
2. **Analyze embedding space properties**:
   - Extract embedding distances for failure cases
   - Compare with successful anonymization cases
   - Identify clustering patterns in failure cases

3. **Compute image quality metrics**:
   - Extract SSIM scores for failure cases
   - Analyze correlation between image quality and re-identification success
   - Identify quality thresholds that predict failures

### Phase 3: Visual Analysis
1. Generate comparison figures:
   - Original vs. anonymized images for failures
   - Embedding space visualizations
   - Demographic distribution plots
2. Create case studies of representative failures

### Phase 4: Statistical Analysis
1. Statistical significance testing
2. Correlation analysis between attributes and failure rates
3. Regression modeling to predict failure likelihood

## Expected Deliverables

### Code
- `identify_failures.py` - Extract failure cases from rank-k results using key mapping
- `analyze_failures.py` - Characterize and analyze failure patterns with demographic data
- `visualize_failures.py` - Generate visualizations and case studies
- `failure_analysis.ipynb` - Integrated analysis notebook

### Results
- **Failure case dataset**: CSV with metadata including demographic attributes and SSIM scores
- **Demographic analysis**: Attribute preservation rates for failure cases vs. successful cases
- **Statistical analysis report**: Correlations and significance tests
- **Visualization figures**: Demographic distributions, image quality comparisons
- **Recommendations**: Actionable insights for mechanism improvement

## Technical Considerations

### Data Access
- Use `src/data_analysis/query_accuracy.py` for standardized result access
- Follow embedding key format: `{dataset_name}/{image_filename}.jpg`
- Use `CelebAIdentityLookup` for identity mapping
- **Demographic data access**: Join rank-k results with utility CSV files using `key` column
- **Key mapping strategy**: 
  - Rank-k results contain image keys that can be joined with demographic data
  - Both baseline and anonymized demographic files use same key structure
  - Enables analysis of attribute preservation for failure cases

### Privacy & Ethics
- Ensure analysis doesn't compromise privacy of individuals
- Consider ethical implications of failure case analysis
- Follow data protection guidelines

### Computational Requirements
- Large dataset processing may require memory optimization
- Consider batch processing for embedding analysis
- GPU acceleration for image processing tasks

## Timeline
- **Week 1**: Data extraction and failure case identification
- **Week 2**: Characterization analysis and visualization
- **Week 3**: Statistical analysis and reporting
- **Week 4**: Final report and recommendations

## Success Criteria
- Comprehensive identification of failure cases
- Clear characterization of failure patterns
- Actionable recommendations for mechanism improvement
- Reproducible analysis pipeline
