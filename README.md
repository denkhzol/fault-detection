# Tabular Data Classification with PyTorch Tabular

This repository contains a workflow for loading, preprocessing, and training a tabular data classification model, TabTransformer, using PyTorch Tabular. The workflow includes data preprocessing, applying SMOTE for balancing, standardizing data, and training a model with PyTorch Tabular. It will be updated after publication. :)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone git@github.com:denkhzol/fault-detection.git
cd fault-detection
pip install -r requirements.txt
```

## Usage
To run the workflow, execute the main function in the main.py file:

python3 roc_curve_tabtransformer_smote_standartization.py

Results
The ROC curve is plotted and saved for further analysis.

evaluate_and_plot_roc(tabular_model, test_all, y_test2, filename)

## Datasets
Chowdhury, S., Uddin, G., Hemmati, H., & Holmes, R. (2024). Method-level bug prediction: Problems and promises. ACM Transactions on Software Engineering and Methodology, 33(4), 1-31.

Pengő, E. (2021, September). Examining the bug prediction capabilities of primitive obsession metrics. In International Conference on Computational Science and Its Applications (pp. 185-200). Cham: Springer International Publishing.

## Reference model

https://github.com/manujosephv/pytorch_tabular/tree/main

### TabTransformer: Tabular Data Modeling Using Contextual Embeddings
https://arxiv.org/abs/2012.06678
