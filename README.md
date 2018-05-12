# AES system for ESL essays in Python

This AES system seeks to improve the validity of AES used for ELL essays by employing features based on the acquisition order of English negation along with 40 other more commonly used features.

## Getting Started

This system requires a corpus of scored essays.
* [CLC FCE Dataset](https://ilexir.co.uk/datasets/index.html) - This is an open source dataset with scores. The code is not optimized for this dataset but it could be easily reworked to take this corpus

**feature_extractor.py** will take a corpus of essays and extract the following features:

<img width="618" alt="aes_features" src="https://user-images.githubusercontent.com/32346063/39960460-c8764bdc-55e0-11e8-843a-0738568f48ad.png">

**train_model.py** will create a scoring model based on the features extracted against the scores of the essays. This model will be saved as **trained_essay_scoring_model.pkl**. Evaluations of the model will then take place using an 80%/20% training/testing split. Visualizations of the evaluations are optionally available at the bottom of the script.

### Prerequisites

**feature_extractor.py** requires [NLTK](http://www.nltk.org/install.html), [textstat](https://pypi.org/project/textstat/), and [Language-Check](https://pypi.org/project/language-check/)

**train_model.py** needs numpy, pandas, and the following imports from sklearn and matplotlib:

```
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
```

## Contributing

As I am new to this world, any and all contributions are welcome. Please help...

## Authors

* **Travis Moore** - *Link to thesis where this AES system was employed forthcoming*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
