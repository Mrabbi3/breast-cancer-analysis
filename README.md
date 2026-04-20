# Breast cancer analysis

Course / team project layout for exploratory analysis, hypothesis testing, and modeling on breast cancer datasets.

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Raw CSV files, or data loaded in notebooks (e.g. `sklearn.datasets.load_breast_cancer`) |
| `notebooks/` | Analysis notebooks by team member |
| `figures/` | Exported figures (PNG, etc.) for the report |
| `report/` | Final write-up (LaTeX, Word, or PDF) |
| `slides/` | Presentation materials |

## Notebooks

- `01_eda_memberA.ipynb` — exploratory data analysis
- `02_correlation_hypothesis_memberB.ipynb` — correlation and hypothesis tests
- `03_feature_importance_memberC.ipynb` — feature importance / model interpretation

## Data

If you do not commit raw CSVs, document the source in `data/README.md` and load from `sklearn` or a public URL inside the notebooks.
