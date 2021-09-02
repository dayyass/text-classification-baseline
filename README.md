[![tests](https://github.com/dayyass/text-classification-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/dayyass/text-classification-baseline/actions/workflows/tests.yml)
[![linter](https://github.com/dayyass/text-classification-baseline/actions/workflows/linter.yml/badge.svg)](https://github.com/dayyass/text-classification-baseline/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/dayyass/text-classification-baseline/branch/main/graph/badge.svg?token=ABFF3YQBJV)](https://codecov.io/gh/dayyass/text-classification-baseline)

[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/dayyass/text-classification-baseline#requirements)
[![release (latest by date)](https://img.shields.io/github/v/release/dayyass/text-classification-baseline)](https://github.com/dayyass/text-classification-baseline/releases/latest)
[![license](https://img.shields.io/github/license/dayyass/text-classification-baseline?color=blue)](https://github.com/dayyass/text-classification-baseline/blob/main/LICENSE)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/dayyass/text-classification-baseline/blob/main/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pypi version](https://img.shields.io/pypi/v/text-classification-baseline)](https://pypi.org/project/text-classification-baseline)
[![pypi downloads](https://img.shields.io/pypi/dm/text-classification-baseline)](https://pypi.org/project/text-classification-baseline)

### Text Classification Baseline
Pipeline for fast building text classification baselines with **TF-IDF + LogReg**.

### Usage
Instead of writing custom code for specific text classification task, you just need:
1. install pipeline:
```shell script
pip install text-classification-baseline
```
2. run pipeline:
- either in **terminal**:
```shell script
text-clf-train --path_to_config config.yaml
```
- or in **python**:
```python3
import text_clf

text_clf.train(path_to_config="config.yaml")
```

**NOTE**: more about config file [here](https://github.com/dayyass/text-classification-baseline/tree/main#config).

No data preparation is needed, only a **csv** file with two raw columns (with arbitrary names):
- `text`
- `target`

The **target** can be presented in any format, including text - not necessarily integers from *0* to *n_classes-1*.

#### Config
The user interface consists of two files:
- [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml) - general configuration with sklearn **TF-IDF** and **LogReg** parameters
- [**hyperparams.py**](https://github.com/dayyass/text-classification-baseline/blob/main/hyperparams.py) - sklearn **GridSearchCV** parameters

Change **config.yaml** and **hyperparams.py** to create the desired configuration and train text classification model with the following command:
- **terminal**:
```shell script
text-clf-train --path_to_config config.yaml
```
- **python**:
```python3
import text_clf

text_clf.train(path_to_config="config.yaml")
```

Default **config.yaml**:
```yaml
seed: 42
path_to_save_folder: models

# data
data:
  train_data_path: data/train.csv
  valid_data_path: data/valid.csv
  sep: ','
  text_column: text
  target_column: target_name_short

# tf-idf
tf-idf:
  lowercase: true
  ngram_range: (1, 1)
  max_df: 1.0
  min_df: 1

# logreg
logreg:
  penalty: l2
  C: 1.0
  class_weight: balanced
  solver: saga
  n_jobs: -1

# grid-search
grid-search:
  do_grid_search: false
  grid_search_params_path: hyperparams.py
```

**NOTE**: grid search is disabled by default, to use it set `do_grid_search: true`.

**NOTE**: `tf-idf` and `logreg` are sklearn [**TfidfVectorizer**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidf#sklearn.feature_extraction.text.TfidfVectorizer) and [**LogisticRegression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) parameters correspondingly, so you can parameterize instances of these classes however you want. The same logic applies to `grid-search` which is sklearn [**GridSearchCV**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) parametrized with [**hyperparams.py**](https://github.com/dayyass/text-classification-baseline/blob/main/hyperparams.py).

#### Output
After training the model, the pipeline will return the following files:
- `model.joblib` - sklearn pipeline with TF-IDF and LogReg steps
- `target_names.json` - mapping from encoded target labels from *0* to *n_classes-1* to it names
- `config.yaml` - config that was used to train the model
- `hyperparams.py` - grid-search parameters (if grid-search was used)
- `logging.txt` - logging file

### Requirements
Python >= 3.6

### Citation
If you use **text-classification-baseline** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2021textclf,
    author       = {El-Ayyass, Dani},
    title        = {Pipeline for training text classification baselines},
    howpublished = {\url{https://github.com/dayyass/text-classification-baseline}},
    year         = {2021}
}
```
