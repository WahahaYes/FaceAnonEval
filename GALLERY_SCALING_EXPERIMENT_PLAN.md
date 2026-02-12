# Rapid Gallery Scaling Analysis Plan

## Research Question
How does re-identification rate scale with gallery size for different privacy mechanisms using existing CelebA test data (1000 identities)?

## Background & Motivation
Current evaluations use the full 1000-identity CelebA test set. We need to understand how re-identification rates change across gallery sizes 2-1000 to provide insights for larger metaverse ecosystems.

## Available Data Analysis
**Existing Assets**:
- **Dataset**: CelebA test set with 1000 identities (already anonymized)
- **Results**: 43+ rank-k evaluation files for CelebA_test with various privacy mechanisms
- **Privacy mechanisms**: dtheta_privacy (different theta/eps combinations), identity_dp, simswap
- **Current evaluation**: Full gallery (1000 identities) vs query set

## Rapid Implementation Strategy (Hours, Not Weeks)

### 1. Core Approach: Re-run Identity Matching with Precomputed Embeddings
**Key Insight**: We need to re-run the distance-based matching with different gallery sizes, but we can reuse the precomputed embeddings to avoid expensive face recognition model inference.

**Current Understanding**: 
- `k` = rank position where true match was found in full gallery (1000 identities)
- To simulate smaller galleries, we need to recompute distances and find matches within subsets
- Precomputed embeddings in `evaluator.real_embeddings` and `evaluator.anon_embeddings` can be reused

### 2. Implementation Plan

#### Step 1: Load Precomputed Embeddings (15 minutes)
- Load cached embeddings for CelebA test set (both real and anonymized)
- Load identity lookup table for mapping images to identities
- Organize embeddings by identity for efficient gallery subsampling

#### Step 2: Implement Gallery Subsampling Matching (1 hour)
```python
def gallery_scaling_evaluation(real_embeddings, anon_embeddings, identity_lookup, gallery_sizes):
    results = {}
    
    for gallery_size in gallery_sizes:
        # Sample gallery_size unique identities
        gallery_identities = sample_identities(identity_lookup, gallery_size)
        
        # For each query, find matches within this gallery
        for query_path, query_embedding in anon_embeddings.items():
            # Filter gallery to only sampled identities
            gallery_embeddings = filter_gallery(real_embeddings, gallery_identities)
            
            # Compute distances and find rank within this gallery
            rank = find_rank_in_gallery(query_embedding, gallery_embeddings, identity_lookup)
            
            # Store result
            results[gallery_size].append(rank)
    
    return results
```

#### Step 3: Batch Process dtheta_privacy Results (1 hour)
- Focus only on dtheta_privacy mechanism (most relevant for follow-up)
- Test key parameter combinations: theta=0°, eps=[-1, 0, 1, 10, 100, 1000]
- Generate scaling curves for gallery sizes 2-1000

#### Step 4: Analysis & Visualization (1 hour)
- Plot accuracy vs gallery size curves
- Fit scaling models (power law, logarithmic) 
- Generate insights for extrapolation to larger metaverse scales

### 3. Target Privacy Mechanisms
**Focused Scope** (dtheta_privacy only):
- **dtheta_privacy**: theta=0°, eps=[-1, 0, 1, 10, 100, 1000] (primary focus)
- **Optional**: theta=[30, 120, 135, 150, 180]°, eps=[-1, 0, 1, 10, 100, 1000] (if time permits)

**Rationale**: dtheta_privacy is the core contribution mechanism; other mechanisms are baselines not needed for this follow-up analysis.

### 4. Expected Outputs

#### Scaling Curves
- Accuracy vs gallery size (2-1000) for dtheta_privacy parameters
- Epsilon comparison plots (theta=0°, different eps values)
- Scaling rate analysis across privacy budgets

#### Modeling Results
- Power law fitting: `accuracy = a * gallery_size^b`
- Logarithmic fitting: `accuracy = a - b * log(gallery_size)`
- Extrapolation insights for million-scale metaverse galleries

#### Key Metrics
- **Scaling exponent**: How quickly accuracy decreases with gallery size
- **Half-life**: Gallery size where accuracy drops to 50%
- **Parameter sensitivity**: How epsilon affects scaling behavior
- **Extrapolation estimates**: Predicted accuracy at 10K, 100K, 1M users

## Next Steps

1. **Review and refine** this experimental plan
2. **Assess computational resources** and timeline feasibility
3. **Begin implementation** of gallery sampling infrastructure
4. **Validate methodology** on a small pilot study
5. **Proceed with full data collection** based on pilot results

---

*This plan provides a comprehensive framework for addressing the gallery size scaling question while leveraging existing infrastructure and data. The methodology is designed to be both scientifically rigorous and practically implementable within resource constraints.*
