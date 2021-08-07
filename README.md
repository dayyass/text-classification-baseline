### Text Classification Baseline
Pipeline for building text classification **TF-IDF + LogReg** baselines using **sklearn**.

### Usage
Instead of writing custom code for specific text classification task, you just need:
1. install pipeline:
```shell script
pip install text-classification-baseline
```
2. run pipeline:

    - either in **terminal**:
    ```shell script
    text-clf --config config.yaml
    ```
    
    - or in **python**:
    ```python3
    import text_clf
    
    text_clf.train(path_to_config="config.yaml")
    ```

No data preparation is needed, only a **csv** file with two raw columns (with arbitrary names):
- `text`
- `target`

**NOTE**: the **target** can be presented in any format, including text - not necessarily integers from *0* to *n_classes-1*.

#### Config
The user interface consists of only one file [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml).

Change **config.yaml** to create the desired configuration and train text classification model.

Default **config.yaml**:
```yaml
seed: 42
verbose: true
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
  min_df: 0.0

# logreg
logreg:
  penalty: l2
  C: 1.0
  class_weight: balanced
  solver: saga
  multi_class: auto
  n_jobs: -1
```

#### Output
After training the model, the pipeline will return the following files:
- `model.joblib` - sklearn pipeline with TF-IDF and LogReg steps
- `target_names.json` - mapping from encoded target labels from *0* to *n_classes-1* to it names
- `config.yaml` - config that was used to train the model
- `logging.txt` - logging file

### Requirements
Python >= 3.7

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
