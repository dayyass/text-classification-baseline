### Text Classification Baseline
Pipeline for building text classification **TF-IDF + LogReg** baselines using **sklearn**.

### Usage
Instead of writing custom code for specific text classification task, you just need:
1) install pipeline:
```shell script
pip install text-classification-baseline
```
2a) either run pipeline in **terminal**:
```shell script
text-clf --config config.yaml
```
2b) or run pipeline in **python**:
```python3
import text_clf
text_clf.train(path_to_config="config.yaml")
```

No data preparation is needed, only a **csv** file with two raw columns (with arbitrary names):
- text
- target

**NOTE**: the target can be presented in any format, including text - not necessarily integers from *0* to *n_classes-1*.

#### Config
The user interface consists of only one file [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml).

Change **config.yaml** to create the desired configuration and train text classification model.

Default **config.yaml**:
```{r engine='bash', comment=''}
cat config.yaml
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
