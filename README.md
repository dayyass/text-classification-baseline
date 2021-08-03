### TF-IDF + LogReg baseline
TODO

### Installation
`
pip install text-classification-baseline
`

### Usage
The user interface consists of only one file [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml).

Change [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml) to create the desired configuration and train text classification model with the following command:
```python3
python baseline.py --config config.yaml
```
**NOTE**: if **--config** argument is not specified, then [**config.yaml**](https://github.com/dayyass/text-classification-baseline/blob/main/config.yaml) is used.

### Models
List of implemented models:
- [x] TF-IDF + LogReg
- [ ] TF-IDF + NaiveBayes

### Requirements
TODO

### Citation
If you use **text-classification-baseline** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2021textclassification,
    author       = {El-Ayyass, Dani},
    title        = {Pipeline for training text classification baselines},
    howpublished = {\url{https://github.com/dayyass/text-classification-baseline}},
    year         = {2021}
}
```
