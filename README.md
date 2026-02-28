## Overview

This repository contains the code accompanying the paper **"On importance sampling and multilinear extensions for approximating Shapley values with applications to explainable artificial intelligence"**.

All experiments described in the paper can be reproduced using the commands described in the [Usage](#usage) section.

## Project structure

| Path | Content |
|----------|----------|
| [```src/shapley_mcis_mes/approx_algorithms```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/src/shapley_mcis_mes/approx_algorithms) | The Shapley value approximation algorithms considered in the paper.  |
| [```src/shapley_mcis_mes/games```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/src/shapley_mcis_mes/games) | The cooperative games used for the experiments. |
| [```src/shapley_mcis_mes/scripts```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/src/shapley_mcis_mes/scripts) | The scripts starting the experiments. |
| [```data```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/data) | The results of the experiments saved as CSV files. |
| [```figures```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/figures) | The figures from paper based on the experiment results in the [```data```](https://github.com/tim-pollmann/shapley-mcis-mes/tree/main/data) directory. |


## Installation

The installation and execution requires **Python ≥ 3.12**. The code was tested on **Ubuntu 24.04.3 LTS** only. Based on your needs, choose one of the following installation types:

### Standard (recommended)

```sh
pip install .
```

### Development

```sh
pip install -e .[development]
pre-commit install
```

## Usage

After installation, run one of the following commands to reproduce the figures from the paper:

| Command | Description |
|----------|----------|
| ```run-ag-mse-comparison``` | Compares the mean squared errors of all algorithms on an airport game with 100 players. |
| ```run-wvg-mse-comparison``` | Compares the mean squared errors of all algorithms on a weighted voting game with 50 players. |
| ```run-diabetes-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```GradientBoostingRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) in the context of the [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html). |
| ```run-housing-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of an [```MLPRegressor```](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) in the context of the [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). |
| ```run-wine-mse-comparison``` | Compares the mean squared errors of all algorithms when approximating the feature importances of a [```RandomForestClassifier```](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in the context of the [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) (probability of class $0$ only).  |
| ```run-bias-comparison``` | Compares the biases of *OS*, *MES*, and *S-MES* on an airport game with 23 players. |
| ```run-variance-comparison``` | Compares the theoretical and empirical variances of *MES*, *MCIS*, and *S-MCIS* on different weighted voting games. |
| ```run-mle-visualization``` | Visualizes the sampling distribution of *OS*. |

> [!IMPORTANT]
> The mean squared error comparisons do ```iters_per_T``` runs per $T$ to average the mean squared errors at any given $T$. When executing *S-MCIS*, it is not guaranteed that the algorithm runs successfully (compare **Proposition 8** in the paper). Thus, for any $T$, we require at least ```iters_per_T / 2``` successful executions for the average mean squared error to be shown in the final figure. On the other hand, we do not account for that behavior when executing the variance comparison. Instead, we rely on large $T$ and small $n$ to assume that the probability of *S-MCIS* failing is close to $0$ (again, compare **Proposition 8** in the paper).

## Citation

The paper is currently **under review**. In the meantime, you can reference the preprint via the following DOI: https://doi.org/10.20944/preprints202601.0530.v1.

<!-- If you use this code, please cite:

```sh
@article{...}
``` -->
