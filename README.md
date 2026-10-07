# Energy, Information, and the Limits of Neuropsychiatric Classification

This repository contains the code for the maximum-entropy, machine-learning, synthetic, and mechanical analyses accompanying the manuscript *Energy, Information, and the Limits of Neuropsychiatric Classification*. The analyses compare discrimination and distributional fit across connectivity representations and model structures.

The empirical data and fitted models cannot be shared through this repository. The code accepts externally supplied inputs, and a generated-data example is provided to demonstrate the workflows. Image preprocessing and literature analyses are outside the scope of this code.

## Installation

The package requires Python 3.11 or newer. Install it from the repository directory:

```sh
python -m pip install .
```

The command-line interface is available as `neuroimaging-maxent`. Run `neuroimaging-maxent --help` to see the available commands. The optional `fast` dependency enables the Numba inference backend; NumPy inference is available without it.

## Data Preparation

The empirical analyses use an observations-by-features matrix of functional-connectivity vectors. Supply this matrix in an external NPZ archive containing `features` and `row_ids`, together with an external CSV metadata file containing `row_id`. The metadata are aligned with the feature matrix by identifier. Comparisons, covariates, score strata, and subset filters are defined in the analysis configuration.

Connectivity vectors can also be constructed from already preprocessed regional time series using `build-fc`. Signal orientation and triangle ordering are specified explicitly. The input formats and configuration fields are described in [docs/inputs.md](docs/inputs.md); input loading and target construction are implemented in [inputs.py](src/neuroimaging_maxent/inputs.py) and [targets.py](src/neuroimaging_maxent/targets.py).

Copy a template from [configs/](configs/) to an external directory, replace its placeholders, and configure the required comparisons. Private configurations, inputs, checkpoints, caches, and generated outputs must remain outside the repository. Unresolved placeholders and paths resolving inside the checkout are rejected.

```sh
neuroimaging-maxent validate-inputs --config '<PRIVATE_CONFIG>'
```

## Maximum-Entropy and Machine-Learning Analyses

The main empirical workflow is implemented in [empirical/workflow.py](src/neuroimaging_maxent/empirical/workflow.py). It uses repeated stratified cross-validation to compare class-conditional maximum-entropy likelihood ratios with RBF support-vector classifiers. The classifiers receive original connectivity, continuous latent activations, binary latent states, or connectivity decoded from those states.

The compact [binary autoencoder](src/neuroimaging_maxent/autoencoders/compact.py) learns a connectivity representation within each training fold. Scaling, autoencoder training, topology learning, parameter estimation, and classifier fitting are confined to the training data. The classifier applied to original connectivity bypasses the autoencoder and provides a common baseline across latent configurations.

The [sparse model](src/neuroimaging_maxent/models/model.py) learns a shared graph for the two classes and uses [junction-tree inference](src/neuroimaging_maxent/models/inference.py) for exact normalization and likelihood evaluation. The [full-pairwise model](src/neuroimaging_maxent/models/enumerated.py) evaluates all binary configurations by enumeration. Configuration templates cover the 120-variable tree, the 22-variable full-pairwise comparator, and the larger-latent and higher-treewidth comparisons.

```sh
neuroimaging-maxent empirical --config '<PRIVATE_CONFIG>'
```

## Postprocessing

The empirical workflow saves comparison-level summaries, fold-level estimates, held-out predictions, reconstruction measures, and model-adequacy diagnostics. Confidence intervals are calculated by class-stratified participant bootstrap, retaining the repeated held-out predictions associated with each resampled participant. Marginal agreement, total latent activity, held-out likelihood, Brier score, and classification log loss are evaluated separately.

[Reporting functions](src/neuroimaging_maxent/reporting/plots.py) produce AUC, comparison-profile, and marginal-agreement figures from supplied result tables. The plotting configuration specifies the external results file, figure type, and output directory.

```sh
neuroimaging-maxent plot --config '<PRIVATE_CONFIG>'
```

## Latent-Dimension and Treewidth Selection

The [selection workflow](src/neuroimaging_maxent/selection/workflow.py) compares sparse models across latent dimensions and treewidths using common joint-stratified folds. Representation learning and model fitting occur within the corresponding training splits. Three [selection rules](src/neuroimaging_maxent/selection/rules.py) prioritize mean AUC, balanced performance across outcomes, or held-out marginal fit.

Outer-fold evaluation of selected configurations and full-input configuration selection are reported separately. Their estimates also remain distinct from the repeated cross-validation estimates of the fixed empirical pipelines.

```sh
neuroimaging-maxent select --config '<PRIVATE_CONFIG>'
```

## Fixed-Source Synthetic Analyses

The [synthetic workflow](src/neuroimaging_maxent/synthetic/workflow.py) generates observations along an interpolation and extrapolation of two fitted classes' natural parameters. It compares population and finite-sample oracle AUC with SVC performance on binary and decoded representations. A decode–encode–refit analysis evaluates the effect of passing generated states through the fixed source autoencoder.

Source models and checkpoints are supplied externally or generated with [source fitting](src/neuroimaging_maxent/synthetic/source.py). The historical [staged autoencoder](src/neuroimaging_maxent/autoencoders/staged.py) used for source fitting is kept separate from the compact fold-specific architecture. Synthetic performance describes the supplied source distributions and does not establish a universal ceiling for empirical classification.

```sh
neuroimaging-maxent fit-source --config '<SOURCE_CONFIG>'
neuroimaging-maxent synthetic --config '<SYNTHETIC_CONFIG>'
```

## Mechanical Analysis

The independent [triple-well calculation](src/neuroimaging_maxent/mechanical/quadrature.py) uses numerical quadrature to evaluate equilibrium densities, classification AUC, moving-well occupancies, and forward and reverse work. It also checks the reciprocal-work identity. The potential is `U_alpha(x) = a*x²*(x²−1)² − alpha*h*x`, where `h` scales the applied field. This analysis requires no empirical inputs.

```sh
neuroimaging-maxent mechanical --config '<MECHANICAL_CONFIG>'
```

The four-panel Figure 3 is produced by the [figure calculation and renderer](src/neuroimaging_maxent/mechanical/figure.py), using the same quadrature routine. Its [typography and palette](src/neuroimaging_maxent/reporting/style.py) follow the main analysis figure. Copy [mechanical_figure.json](configs/mechanical_figure.json) outside the repository, replace both path placeholders, and supply an external landmark JSON as described in [docs/inputs.md](docs/inputs.md#mechanical-figure). Setting `make_figure` to `true` enables figure generation; omitting it retains numerical-only output.

```sh
neuroimaging-maxent mechanical --config '<FIGURE_CONFIG>'
```

The equivalent [versioned launcher](scripts/make_figure3_paper_style_v2.py) accepts the same configuration:

```sh
python scripts/make_figure3_paper_style_v2.py --config '<FIGURE_CONFIG>'
```

Landmark labels, target AUCs, and comparison ranges come from the external JSON. The calculation solves for the tilts that reach each target, uses the unrounded solutions for annotations, and labels shaded bands as outcome ranges. Outputs are a PNG, a vector PDF, curve and landmark tables, selected potential and density curves, and a numerical validation report, all under the configured `figure_prefix` in the external output directory. No manuscript-specific landmarks or generated figures are bundled here.

## Example and Tests

A small example generates toy inputs at runtime and runs empirical evaluation, configuration selection, source fitting, synthetic analysis, and mechanical quadrature. Replace the output placeholder with a directory outside the repository:

```sh
neuroimaging-maxent demo --output-dir '<OUTPUT_DIR>'
```

The [tests](tests/) cover inference against enumeration, row alignment, target definitions, training separation, bootstrap grouping, numerical retries, selection rules, synthetic controls, and the mechanical work identity.

```sh
python -m pip install '.[test]'
pytest -p no:cacheprovider
```

Detailed numerical settings, scoring conventions, and evaluation scopes are documented in [docs/methods.md](docs/methods.md). The example uses small settings to demonstrate execution; reproducing empirical findings requires the corresponding private inputs and analysis configurations.
