# AI-Aided Lipid Nanoparticle Design

## Project question

Can ionizable-lipid molecular structure and formulation composition predict normalized LNP luminescence performance under chemically disjoint validation?

## Dataset

The analysis uses the LNPDB dataset from the public LNPDB repository.

Source:
https://github.com/evancollins1/LNPDB

The raw LNPDB CSV was used as the starting point. Published model splits, pretrained models, and prepared feature matrices were not used.

### Dataset subset

The modeling subset contains:

- Experiment method: `luminescence_normalized`
- Valid target values: 14,302
- Unique ionizable-lipid SMILES: 8,600
- Study groups: 31
- Target: `Experiment_value`

## Feature engineering

Two feature groups were evaluated:

1. Morgan molecular fingerprints
   - Radius: 2
   - Fingerprint size: 1024
   - Molecular structure source: ionizable-lipid SMILES

2. Formulation features
   - IL molar ratio
   - Helper-lipid molar ratio
   - Cholesterol molar ratio
   - PEG-lipid molar ratio
   - IL-to-nucleic-acid mass ratio

The combined representation contains 1,029 features.

## Models

Random Forest regression was used with:

- 300 trees
- `max_features="sqrt"`
- `min_samples_leaf=5`
- fixed random seed: 42

Three feature settings were compared:

- formulation ratios only
- Morgan fingerprints only
- Morgan fingerprints + formulation ratios

## Validation strategy

Two chemically constrained validation strategies were evaluated:

### Exact-SMILES chemical-disjoint validation

Identical ionizable-lipid SMILES were prevented from appearing in both training and test sets.

### Cluster-disjoint validation

Ionizable lipids were grouped using Tanimoto-based Butina clustering of Morgan fingerprints, and clusters were kept separate between training and test folds.

## Main results

| Validation | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| Exact-SMILES | Ratios only | 0.989 | 0.775 | 0.017 |
| Exact-SMILES | Morgan only | 0.903 | 0.697 | 0.183 |
| Exact-SMILES | Morgan + ratios | 0.887 | 0.688 | 0.209 |
| Cluster-disjoint | Ratios only | 0.987 | 0.796 | 0.010 |
| Cluster-disjoint | Morgan only | 0.952 | 0.768 | 0.078 |
| Cluster-disjoint | Morgan + ratios | 0.944 | 0.759 | 0.094 |

Under both chemical validation strategies, molecular structure provided substantially more predictive information than formulation ratios alone. Adding formulation ratios to Morgan fingerprints produced a modest additional improvement.

## Independent analysis

This project does not simply reproduce the published LNPDB modeling pipeline.

The independent analysis includes:

- reconstruction of the modeling subset from the raw dataset
- independent feature construction using Morgan fingerprints
- comparison of structure-only, formulation-only, and combined representations
- exact-SMILES chemical-disjoint validation
- fingerprint-based chemical clustering and cluster-disjoint validation
- permutation importance analysis of formulation variables
- out-of-fold prediction analysis
- residual analysis
- independent model training and model artifact generation

No claim of formal novelty is made without a dedicated literature review.

## Interpretation

Permutation importance under cluster-disjoint validation indicated that formulation ratios contributed additional predictive information to the molecular representation.

Mean permutation importance:

| Feature | Mean importance |
|---|---:|
| HL molar ratio | 0.00821 |
| CHL molar ratio | 0.00687 |
| IL molar ratio | 0.00329 |
| PEG molar ratio | 0.00188 |
| IL-to-nucleic-acid mass ratio | 0.00163 |

These values represent predictive contribution within the evaluated model and should not be interpreted as causal biological effects.

## Limitations

- The target is study-normalized luminescence rather than a universal raw transfection measurement.
- The cluster-disjoint split is highly imbalanced because one chemical cluster contains a large fraction of the observations.
- Exact-SMILES separation does not guarantee complete scaffold-level chemical separation.
- Random Forest performance is modest under chemically constrained validation.
- The analysis does not establish causal relationships between formulation variables and biological activity.
- Additional validation on external datasets would be required before practical deployment.

## Outputs

### Tables

- `results/tables/final_model_comparison.csv`
- `results/tables/permutation_importance_cluster.csv`
- `results/tables/oof_predictions_cluster.csv`

### Figures

- `results/figures/model_performance_r2.png`
- `results/figures/predicted_vs_actual_cluster.png`
- `results/figures/residual_distribution_cluster.png`

### Model

- `results/models/lnp_rf_morgan_ratios.joblib`
- `results/models/lnp_rf_morgan_ratios_metadata.json`

## Reproducibility

The analysis was performed in Python using pandas, NumPy, scikit-learn, RDKit, matplotlib, and joblib.

The project is organized so that raw data, processed outputs, analysis notebooks, source code, figures, tables, and trained models can be tracked separately.
