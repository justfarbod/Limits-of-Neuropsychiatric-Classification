# Inputs and commands

## Connectivity and metadata

`features_file` is an external NPZ archive with:

| Array | Shape | Content |
|---|---|---|
| `features` | observations × features | Finite connectivity values |
| `row_ids` | observations | Unique strings or integers; no pickled objects |

`metadata_file` is an external CSV with a unique `row_id` field. String identifiers, including leading zeros, are preserved. Rows are joined by identifier and reordered to match the feature archive. Missing matches and duplicates fail validation. Extra metadata rows are unused. Only configured columns are interpreted.

`subset` optionally restricts aligned rows before target construction:

```json
{"subset": {"column": "subset", "values": ["A"]}}
```

Each definition in `targets` has a unique filename-safe key. Positive classes receive label 1 and higher MaxEnt scores favour that class.

```json
{
  "targets": [
    {"key": "groups", "kind": "groups", "column": "group", "negative": "A", "positive": "B"},
    {"key": "binary", "kind": "binary", "column": "binary_covariate", "negative": 0, "positive": 1},
    {"key": "threshold", "kind": "threshold", "column": "continuous_covariate", "threshold": 0.0},
    {"key": "pairs", "kind": "pairwise_groups", "column": "group", "groups": ["A", "B", "C"]},
    {"key": "strata", "kind": "score_strata", "column": "score", "group_column": "group", "negative": "A", "positive": "B"}
  ]
}
```

Binary/group comparisons exclude other categories and missing values. Threshold comparisons exclude missing values and assign label 1 to values greater than or equal to the threshold. Pairwise comparisons use list order to define negative and positive groups; suffixes identify their list positions.

Score strata use separate medians for the two groups, calculated on observed scores in the complete analysis subset before cross-validation. Scores equal to the median belong to the low stratum; strictly larger scores belong to the high stratum. Four cross-group comparisons are constructed. The suffix is `positive_high_negative_high`, with 0 meaning low and 1 meaning high. Medians define the analysis outcomes; they are not trained fold-specific transformations. Missing scores are excluded. For selection, all configured outcomes must cover exactly the same rows so common joint-stratified folds can be used.

## Constructing connectivity

`build-fc` reads an external CSV manifest with `row_id,file`. Each file is a numeric NPY matrix of already preprocessed signals. Relative filenames resolve against the external manifest directory. A configuration contains `time_series_manifest`, `output_dir`, `orientation`, `triangle`, and optionally `constant_policy`.

`orientation` is `time_by_region` or `region_by_time`. Pearson correlation is computed across time for each region pair. `triangle` is `lower` (default) or `upper`, excluding the diagonal. Ordering follows NumPy's `tril_indices(n, -1)` or `triu_indices(n, 1)`, respectively: row order, then column order. Choose the same ordering throughout an analysis. Constant signals produce zero incident correlations by default; `constant_policy: "error"` rejects them. All observations must have the same region count. The command writes an external `features.npz`.

## Configuration and outputs

Configuration files themselves must be outside the repository. Relative configuration paths resolve from the command's working directory; absolute external paths avoid ambiguity. Every input and output path is validated after resolving symbolic links. Keep user-defined fields, outcome names, and file locations in private configuration files only.

| Command | Required configuration | Principal external outputs |
|---|---|---|
| `validate-inputs` | Feature archive, metadata, targets | Counts only on standard output |
| `build-fc` | Signal manifest, output directory | Feature archive |
| `empirical` | Feature archive, metadata, targets, pipeline settings, output directory | Summary, folds, predictions, marginal diagnostics, cache, fit records |
| `select` | Same inputs; complete-row outcomes; candidate grid and fold counts | Candidate metrics, rankings, inner/outer predictions, training audit, winners |
| `fit-source` | One comparison; pipeline settings; optional supplied checkpoint | `source_model.npz`, `autoencoder.pt`, descriptive source-fit record |
| `synthetic` | `source_model`, `autoencoder_checkpoint`, dimensions/method, simulation settings | Replicate/fold/population summaries and round-trip records |
| `mechanical` | Quadrature settings, output directory | `mechanical.csv` |
| `plot` | `results_file`, `output_dir`, optional `plot_kind` | PNG figure |

Source models use a versioned numeric NPZ containing both classes' biases and symmetric interactions, method, selected edges, and a JSON decomposition certificate. Checkpoints contain an architecture tag, scalar settings, tensor weights, and scaling tensors. Checkpoints load with `weights_only=True`. Externally supplied models must follow these schemas; no original project directory is consulted. `fit-source` creates both files in the expected format. A supplied checkpoint's training scope remains the user's responsibility.

Plots accept supplied CSV results. `plot_kind` can be `auc`, `profile`, or `marginals`. AUC tables need `outcome,method,auc` or `alpha,method,auc`; interval columns are optional. Profile plots compare supplied method columns with `maxent`. Marginal tables require `observed,modeled`. No empirical values or dataset labels are built into the figures.

## Mechanical figure

Use the external configuration fields in `configs/mechanical_figure.json`. `landmarks_file` points to another external JSON object with `landmarks` and optional `bands` lists. The default deformation grid is 151 equally spaced values from zero to 1.5; an explicit `alpha_values` list must increase strictly from zero and retain three wells throughout. The output stem is set by `figure_prefix`.

Each landmark requires a unique `label` and an `auc` between chance and the largest AUC attainable on the configured range. Optional fields are `color`, `marker` (a Matplotlib marker string, default `o`), and `show_distribution` (default `false`). Select one or two landmarks with `show_distribution: true` for the potential, density, and occupancy annotations; the other landmarks appear only in the AUC panel. Colors can be `grey`, `rose`, `blue`, `purple`, or Matplotlib color specifications.

Each band requires `label`, `lower`, and `upper`, with optional `color`. Bounds must satisfy `0.5 <= lower < upper <= 1`. These represent ranges across outcomes and are explicitly labeled as such. Do not supply confidence intervals as bands.

Create the metadata outside the repository. For example, the following is a schema illustration; replace the text placeholders with your own labels and numeric values before running:

```json
{
  "landmarks": [
    {"label": "<COMPARISON_LABEL>", "auc": "<TARGET_AUC>",
     "color": "rose", "show_distribution": true}
  ],
  "bands": [
    {"label": "<RANGE_LABEL>", "lower": "<LOWER_AUC>",
     "upper": "<UPPER_AUC>", "color": "rose"}
  ]
}
```

The figure command writes `<prefix>.png`, `<prefix>.pdf`, `<prefix>_curves.csv`, `<prefix>_landmarks.csv`, `<prefix>_densities.csv`, and `<prefix>_validation.json`. The curve table records AUC, its bound, well occupancies, Jeffreys divergence, and dimensionless work. The landmark table includes both requested and achieved AUCs and the solved tilts. The density table contains the reference and selected distributions. Inputs, metadata, and every output must be outside the repository.
