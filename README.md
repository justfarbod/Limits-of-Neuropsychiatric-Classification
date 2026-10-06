# Neuroimaging MaxEnt Analysis

This repository contains Python code for comparing pairwise maximum-entropy models and machine-learning classifiers on functional-connectivity representations. The analyses examine how binary compression and restrictions on model dependencies affect distributional fit and classification performance.

The code covers empirical cross-validation, latent-dimension and treewidth selection, simulations from fixed fitted distributions, and an independent triple-well mechanical example. Private observations and fitted models are not included. The workflows accept externally supplied inputs, and a small generated-data demo is available for exploring the implementation.

## Reading the code

Start with [the empirical workflow](src/neuroimaging_maxent/empirical/workflow.py), which connects the main components:

1. [Input loading](src/neuroimaging_maxent/inputs.py) aligns connectivity vectors with metadata by row identifier. [Target construction](src/neuroimaging_maxent/targets.py) defines the configured comparisons and subsets.
2. [Representation learning](src/neuroimaging_maxent/autoencoders/representation.py) fits the standardizer and autoencoder on training rows, then encodes and reconstructs the evaluation rows.
3. [Model fitting and scoring](src/neuroimaging_maxent/models/pair.py) fit separate class distributions and compute their log-likelihood ratio.
4. [Discriminative evaluation](src/neuroimaging_maxent/empirical/discriminative.py) fits the SVC comparisons and computes bootstrap intervals from repeated held-out predictions. [Adequacy diagnostics](src/neuroimaging_maxent/empirical/diagnostics.py) assess marginal and total-activity agreement.

The empirical comparisons use original connectivity, continuous latent activations, binary latent states, and connectivity decoded from those states. Scaling, representation learning, graph selection, and classifier fitting remain within the training folds. The classifier on original connectivity bypasses the autoencoder and provides a common baseline across latent configurations.

## Main implementations

| Component | Code | Purpose |
|---|---|---|
| Compact binary autoencoder | [compact.py](src/neuroimaging_maxent/autoencoders/compact.py) | Fold-specific representation learning with a straight-through binary bottleneck |
| Historical staged autoencoder | [staged.py](src/neuroimaging_maxent/autoencoders/staged.py) | Separate soft-pretraining and binary-fine-tuning architecture for source fitting |
| Full-pairwise MaxEnt | [enumerated.py](src/neuroimaging_maxent/models/enumerated.py) | Exhaustive normalization and moments for small binary systems |
| Sparse graph learning | [graph.py](src/neuroimaging_maxent/models/graph.py) | Dependency screening, a Chow–Liu tree, and certified bounded-width extensions |
| Sparse inference and fitting | [inference.py](src/neuroimaging_maxent/models/inference.py), [model.py](src/neuroimaging_maxent/models/model.py) | Exact junction-tree normalization, marginals, independent sampling, and parameter estimation |
| Configuration selection | [workflow.py](src/neuroimaging_maxent/selection/workflow.py), [rules.py](src/neuroimaging_maxent/selection/rules.py) | Joint-stratified nested evaluation and AUC, balanced one-SE, and fit-priority rules |
| Fixed-source simulations | [source.py](src/neuroimaging_maxent/synthetic/source.py), [workflow.py](src/neuroimaging_maxent/synthetic/workflow.py) | Source fitting, parameter interpolation, oracle AUC, SVC evaluation, and decode–encode–refit comparisons |
| Mechanical example | [quadrature.py](src/neuroimaging_maxent/mechanical/quadrature.py) | Triple-well equilibrium densities, classification AUC, well occupancies, and reciprocal work |
| Reporting | [plots.py](src/neuroimaging_maxent/reporting/plots.py) | Figures from supplied result tables |

The sparse model restricts the dependencies it can represent while retaining exact normalization and likelihood evaluation. The full-pairwise model retains all interactions but requires enumeration of all binary configurations. Templates include the 120-variable tree, the 22-variable full-pairwise comparator, and the larger-latent and higher-treewidth comparisons.

Synthetic oracle performance is defined under the supplied source distributions. It is distinct from held-out empirical performance and does not establish a universal classification ceiling.

## Documentation and configurations

- [Methods](docs/methods.md): architectures, fitting settings, scoring conventions, training boundaries, bootstrap procedure, selection rules, and simulation design.
- [Inputs](docs/inputs.md): feature and metadata schemas, target definitions, subset filters, signal orientation, connectivity ordering, and command outputs.
- [Configuration templates](configs/): empirical, selection, synthetic, and mechanical settings with external-file placeholders.
- [Tests](tests/): inference against enumeration, training separation, row alignment, target definitions, bootstrap grouping, numerical retries, selection, synthetic controls, and the work identity.

Connectivity construction begins with already preprocessed regional time series. Image preprocessing and literature analyses are outside the repository's scope.

## Installation and a small example

Python 3.11 or newer is required. From the repository directory:

```sh
python -m pip install .
neuroimaging-maxent --help
neuroimaging-maxent demo --output-dir '<OUTPUT_DIR>'
```

Replace `<OUTPUT_DIR>` with a directory outside the checkout. The demo generates toy inputs at runtime and exercises empirical evaluation, configuration selection, source fitting, simulation, and mechanical quadrature with small settings. It is an execution example, not a reproduction of empirical findings.

To run an analysis with supplied inputs, copy a configuration template to an external directory, replace its placeholders, and define the relevant outcomes:

```sh
neuroimaging-maxent validate-inputs --config '<PRIVATE_CONFIG>'
neuroimaging-maxent empirical --config '<PRIVATE_CONFIG>'
```

Other commands are `build-fc`, `select`, `fit-source`, `synthetic`, `mechanical`, and `plot`; their inputs and outputs are documented in [Inputs](docs/inputs.md). The full enumeration and configuration search can require substantial computation.

For the automated checks:

```sh
python -m pip install '.[test]'
pytest -p no:cacheprovider
```

The optional `fast` dependency enables the Numba inference backend. NumPy inference is available without it.

## Data handling

Connectivity inputs use an external NPZ archive containing `features` and `row_ids`, with external CSV metadata joined by `row_id`. Arbitrary string identifiers are supported. Optional signal manifests, fitted source models, and autoencoder checkpoints are also supplied explicitly.

Keep private configurations, inputs, checkpoints, caches, and generated outputs outside the repository. The package rejects unresolved placeholders and paths resolving inside the checkout. No data, fitted weights, cached representations, predictions, or results are bundled with the code.
