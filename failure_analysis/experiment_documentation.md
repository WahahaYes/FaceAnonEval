# Failure Case Demographic Analysis for Privacy-Preserving Face Recognition

## Experimental Methodology

This analysis investigates demographic biases in privacy-preserving face recognition failures, examining whether certain demographic groups experience higher rates of re-identification vulnerability. The study compares demographic distributions between successful privacy protection and failure cases to identify potential fairness concerns.

### Dataset and Privacy Mechanism

- **Dataset**: CelebA test set (19,962 images with demographic annotations)
- **Privacy Mechanism**: dtheta_privacy with θ=0°, ε=1.0 (strong privacy protection)
- **Failure Definition**: Re-identification success (privacy breach)
- **Demographic Attributes**: Age, Emotion, Race, Gender

### Failure Analysis Protocol

1. **Failure Identification**: Extract all cases where privacy mechanism failed (re-identification occurred)
2. **Demographic Matching**: Match failure cases with original demographic annotations
3. **Multiple Subsample Averaging**: Generate 10 random subsamples from full dataset (2,840 images each) and average distributions to prevent sampling bias
4. **Comparative Analysis**: Compare demographic distributions between failure cases and averaged full dataset
5. **Statistical Testing**: Chi-square tests for categorical variables, t-tests for continuous variables
6. **Clustering Analysis**: K-means clustering to identify demographic patterns in failures

### Evaluation Metrics

**Primary Metrics**:
- **Failure Rate**: Percentage of images where privacy protection failed
- **Demographic Distribution**: Comparison of failure cases vs. full dataset
- **Statistical Significance**: p-values for demographic differences
- **Cluster Analysis**: Identification of demographic vulnerability patterns

**Fairness Goal**: Ensure privacy protection is equally effective across all demographic groups without systematic biases.

## Statistical Analysis

- **Failure Cases**: 2,840 images (14.23% failure rate)
- **Multiple Subsamples**: 10 random subsamples of 2,840 images each from full dataset
- **Averaged Comparisons**: Categorical distributions averaged across all 10 subsamples for statistical robustness
- **Age Visualization**: Single subsample used for equal point count comparison (2,840 vs 2,840)
- **Statistical Tests**: Chi-square for categorical, t-test for continuous variables
- **Significance Level**: p < 0.05 for statistical significance

## Visualization Description

**Figure Layout**: Comprehensive single-panel visualization showing demographic distributions across four key attributes

**Age Distribution (Left Section)**:
- Scatter plot displaying individual age values for failure cases (red) vs. full dataset (blue)
- Left y-axis shows age range from 15 to 80 years
- Points are horizontally jittered for better visibility
- Reveals age concentration patterns in privacy failures

**Emotion Distribution (Second Section)**:
- Side-by-side proportional bar charts comparing emotion categories
- Seven emotions displayed: happy, neutral, sad, angry, surprise, fear, disgust
- Right y-axis shows proportion from 0.0 to 1.0
- Red bars represent failure cases, blue bars represent full dataset
- 45° rotated labels for readability

**Race Distribution (Third Section)**:
- Proportional bar charts for six racial categories
- Categories: White, Black, Asian, Hispanic, Indian, Middle Eastern
- Same color coding and axis as emotion section
- Highlights racial disparities in failure rates

**Gender Distribution (Right Section)**:
- Simple two-category comparison (Woman, Man)
- Maintains consistent visualization style
- Shows gender-based vulnerability differences

**Visual Design Elements**:
- Gray vertical separators between demographic sections
- Bold section headers (AGE, EMOTION, RACE, GENDER) positioned at Y=70
- Times New Roman font throughout for academic consistency
- Clean legend with color-coded rectangles
- Subtle grid lines for age axis readability
- All categorical labels positioned above maximum bar heights with configurable offset

## Example Figure Caption

> **Figure X**: Demographic distribution analysis of privacy-preserving face recognition failures. The visualization compares demographic characteristics between failure cases (red) and full dataset (blue) across four key attributes. Age values are shown as scatter plots on the left axis (15-80 years), while emotion, race, and gender distributions are displayed as proportional bar charts on the right axis (0.0-1.0 proportion). Results reveal significant demographic biases in failure rates, with certain groups experiencing higher vulnerability to re-identification. Statistical analysis shows significant differences in age (p < 0.001), race (p < 0.001), and gender (p < 0.001), but not emotion (p = 0.104), highlighting important fairness considerations in privacy-preserving systems.

## Key Findings for Publication

1. **Significant Age Bias**: Failure cases show significantly different age distribution (p < 0.001)
2. **Gender Disparity**: Strong gender bias in failure rates (p < 0.001)
3. **Racial Differences**: Significant racial variation in failure susceptibility (p < 0.001)
4. **Emotion Neutrality**: No significant emotion-based bias in failures (p = 0.235)
5. **Cluster Patterns**: Four distinct demographic vulnerability clusters identified
6. **Statistical Robustness**: Results based on 10 subsample averaging reduce sampling bias

## Experimental Parameters Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Dataset Size | 19,962 images | Complete CelebA test set |
| Failure Cases | 2,840 images | All privacy breaches identified |
| Subsamples | 10 random samples | Prevents sampling bias through averaging |
| Sample Size | 2,840 images per subsample | Matched to failure cases for comparison |
| Clusters | 4 | Optimal number for demographic pattern identification |
| Significance Level | p < 0.05 | Standard statistical threshold |

## Reproducibility Information

**Total Evaluations**: 2,840 failure cases + (10 × 2,840) comparison samples = 31,240 demographic analyses

**Runtime**: ~5-8 minutes on standard CPU (increased due to multiple subsamples)

**Memory Requirements**: ~4GB RAM for demographic processing with subsamples

**Statistical Significance**: Age, Race, and Gender show significant differences (p < 0.001), Emotion shows no significant bias (p = 0.104)

## Cluster Analysis Results

**Cluster 0**: Young adults (Age 25.0, Race: White, Gender: Female)
**Cluster 1**: Young adults (Age 27.2, Race: White, Gender: Female)  
**Cluster 2**: Older adults (Age 52.9, Race: White, Gender: Female)
**Cluster 3**: Young adults with surprise emotion (Age 26.1, Race: White, Gender: Female)

These clusters reveal distinct demographic vulnerability patterns that warrant further investigation for fairness improvements in privacy-preserving systems. Notably, all clusters predominantly identify female subjects with specific age and emotion characteristics.
