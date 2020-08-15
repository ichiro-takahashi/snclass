SN Type Classifier and Redshift Regression 
=========================================================

A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models             <- Trained and serialized models, model predictions, and model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc..
    │   └── figures        <- Generated graphics and figures.
    │
    ├── requirements.txt   <- The requirements for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt.`
    │
    ├── src                <- Source code for this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    │   ├── data           <- Scripts to make datasets.
    │   │   ├── make_dataset.py  <- Script to make input dataset to learn.
    │   │   └── cosmology.py     <- Script to compute distmod.
    │   │
    │   ├── hsc             <- Scripts to load input data or model.
    │   │   ├── dataset.py  <- Script to format input values.
    │   │   ├── loader.py   <- Script to load dataset.
    │   │   └── model.py    <- Script to define DNN architecture.
    │   ├── hsc_redshift.py <- Script to train a redshift regression model, to predict redshift values,
    │   │                      and to search hyper parameters of the model.
    │   ├── hsc_search.py   <- Script to search hyper parameters of SN classifier.
    │   ├── hsc_sn_type.py  <- Script to train sn classifier and predict for sn types.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

----

# Requirements
- python
- tensorflow-gpu == 1.13
- dm-sonnet == 1.23
- click
- numpy
- scipy
- scikit-learn == 0.21
- pandas
- pytables
- joblib
- tqdm
- matplotlib
- seaborn
- mlflow
- optuna == 0.14.0
- AdaBound-Tensorflow (github repository)

We recommend [anaconda](https://www.anaconda.com/distribution/) to
set up an environment.  
You can install the requirements with the following commands. 

```bash
conda create -n crest python click numpy scipy scikit-learn=0.21 pandas \
    pytables joblib tqdm matplotlib seaborn tensorflow-gpu=1.13
conda activate crest
pip install dm-sonnet==1.23 mlflow optuna==0.14.0

# AdaBound-Tensorflow is registered as a submodule of our program.
git submodule update -i

or

cd references
git clone https://github.com/taki0112/AdaBound-Tensorflow.git
```

# Data Preparation
You need to make input data (HDF5 format) to learn a classifier/regression model.

The input data (HDF5 format) are made from the two tables (csv files);
one is flux data table and the other is meta data table. The structures of the tables are as follows:  

**Flux data table**

|object_id|mjd|passband|index|flux|flux_err|
|:-------:|:---:|:------:|:----:|:----:|:------:|
| name1 | 59751 |  u  |   0 | 38.649 | 0.4791  |
| name1 | 59758 |  u  |   1 | 46.387 | 1.0287  |
| :  |  :  |  :  | :  |  :  |  :  |
| name100 | 59749 |  Y  |   6 | 0.3974 | 0.6411  |
| :  |  :  |  :  | :  |  :  |  :  |

**Meta data table**

|object_id| sn_type| redshift|
|:-------:|:------:|:-------:|
|name1| Ia | 1.23|
|name2| Ib | 0.89|
| :  |  :  |  :  |
|name100| Ib | 1.11|
| :  |  :  |  :  |

You need to save the tables as csv (comma separated values) format.

A compressed file of sample dataset (Simdataset_HSC_sample.tar.gz) is available. It contains a flux data table and meta data table of 100,000 simulated supernovae for the classification of the HSC survey data.

The SN types are `Ia`, `Ib`, `Ic`, `Ibc`, `II`, `IIL`, `IIN`, or `IIP`.  

## Command
You convert the csv files into hdf5 files by the following command.

```
# Making training dataset of HSC
python data/convert_dataset.py hsc \
    --data=../data/raw/data.csv --metadata=../data/raw/metadata.csv \
    --output-path=../data/processed/sim_sn_data.h5 --data-type=train

# Making test dataset of HSC
python data/convert_dataset.py hsc \
    --data=../data/raw/data_test.csv --metadata=../data/raw/metadata_test.csv \
    --output-path=../data/processed/hsc_data.h5 --data-type=test
```

The names of the csv files and the hdf5 files are chosen arbitrarily.

### PLAsTiCC
The input data dimensions of all samples must be same.  
You need to extract the data properly.

The data format of the csv files are same as the original data.
(e.g. the values of `passband` are digits.)  
The true labels of the test data are needed to evaluate the test accuracy.
(You can download the dataset from [here](https://zenodo.org/record/2539456).)

```
# Making training dataset of PLAsTiCC
python data/convert_dataset.py plasticc \
    --data=../data/raw/training_set_extracted.csv --metadata=../data/raw/training_set_metadata.csv \
    --output-path=../data/processed/training_cosmos.h5 --data-type=train

# Making test dataset of PLAsTiCC
python data/convert_dataset.py plasticc \
    --data=../data/raw/test_set_extracted.csv --metadata=../data/raw/plasticc_test_metadata.csv \
    --output-path=../data/processed/test_cosmos.h5 --data-type=test
```

# SN type classifier

## Training (HSC)
This classifier supports two classification tasks.  
One is binary classification and the other is 3-class classification.

The binary classification task classifies the samples as:
- class 0: `Ia`,
- class 1: others (`Ib`, `Ic`, `Ibc`, `II`, `IIL`, `IIN`, `IIP`).
  
The 3-class classification task classifies the samples as:
- class 0: `Ia`,
- class 1: `Ib` `Ic` `Ibc`,
- class 2: `II` `IIL` `IIN` `IIP`.

**Usage**
```
python hsc_sn_type.py fit-hsc \
    --sim-sn-path=../data/processed/sim_sn_data.h5 \
    --hsc-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/result1 --seed=0 \
    --n-highways=3 --hidden-size=500 --drop-rate=1e-2 \
    --patience=200 --batch-size=1000 --norm \
    --activation=sigmoid \
    --input1=absolute-magnitude --input2=scaled-flux \
    --optimizer=adam --adabound-final-lr=1e-1 --lr=1e-3 \
    --eval-frequency=20 --mixup=mixup --fold=-1 --cv=5 --threads=4 \
    --binary --remove-y --use-batch-norm --use-dropout
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|sim-sn-path|File path of the training dataset|string|
|hsc-path|File path of the test dataset|string|
|model-dir|Output directory to save the learned models and prediction results|string|
|seed|Random seed|int|
|n-highways|The number of Highway layers|int|
|drop-rate|The ratio of dropout layer|float|
|patience|The model decides to finish to learn every `patience` epochs.|int|
|norm|The flag to normalize the input values|-|
|activation|The types of activation layer|relu, sigmoid, tanh, or identical|
|input1|The type of the main input values|magnitude or absolute-magnitude|
|input2|The type of the sub input values|none or scaled-flux|
|optimizer|The type of optimizer|momentum, adam, amsbound, or adabound|
|adabound-final-lr|The final learning rate. This is available when `optimizer` is adabound or amsbound|float|
|lr|Learning rate|float|
|eval-frequency|Evaluate test dataset every `eval-frequency` epochs.|int|
|mixup|Use mixup or not|mixup or none|
|fold|The index of the folds of cross validation. If 'fold' is -1, all folds are target|int|
|threads|The number of threads|int|
|cv|The number of folds of cross validation|int|
|binary / multi|`binary` for binary classification, `multi` for 3-class classification|-|
|remove-y|The flag to remove Y band or not|-|
|use-batch-norm|The flag to use batch normalization layer or not|-|
|use-dropout|The flag to use dropout layer or not|-|

## Training (PLAsTiCC)
**Usage**
```
python hsc_sn_type.py fit-plasticc \
    --sim-sn-path=../data/processed/sim_sn_data.h5 \
    --training-cosmos-path=../data/processed/training_cosmos.h5 \
    --test-cosmos-path=../data/processed/test_cosmos.h5  \
    --model-dir=../models/result1 --seed=0 \
    --n-highways=3 --hidden-size=500 --drop-rate=1e-2 \
    --patience=200 --batch-size=1000 --norm \
    --activation=sigmoid \
    --input1=absolute-magnitude --input2=scaled-flux \
    --optimizer=adam --adabound-final-lr=1e-1 --lr=1e-3 \
    --eval-frequency=20 --mixup=mixup --fold=-1 --cv=5 --threads=4 \
    --binary --use-batch-norm --use-dropout
```

The options are almost same with the case in training HSC dataset.  
`hsc-path` and `remove-y` are unavailable.  
`training-cosmos-path` and `test-cosmos-path` are newly added.

option name| description| value type / choices|
|:-------|:------|:-------|
|training-cosmos-path|File path of the training dataset derived from PLAsTiCC|string|
|test-cosmos-path|File path of the test dataset derived from PLAsTiCC|string|

## Prediction
The classifier predicts the class of input data with the trained model.

**Usage**
```
python hsc_sn_type.py predict \
    --data-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/result1/0 \
    --data-type=HSC --output-name=prediction.csv
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|data-path|File path of the dataset to predict|str|
|model-dir|Directory that the trained model is in|str|
|data-type|`SimSN` for training dataset, `HSC` for test dataset|`SimSN`, `HSC`, or `PLAsTiCC`|
|output-name|File name to output the predicted results (the file is created in `model-dir`)|str|

The first line of the output file show the class IDs.  
In binary classification case (this means you used the option `--binary` to train the model),
the relations between the class IDs and sn types are as follows,
- class 0: `Ia`,
- class 1: the other sn types.

In multi classification case (the option is `--multi`), 
the relationship is as follows,
- class 0: `Ia`,
- class 1: `Ib` `Ic` `Ibc`,
- class 2: `II` `IIL` `IIN` `IIP`.

The predicted values are logits.
A larger value means a higher probability to belong to the class.    
If you apply the softmax function, you can interpret the output values as probabilities.

## Hyper-parameter search of classifier model
Note that it takes several days to optimize the hyper-parameters.

```
python hsc_search.py search-hsc \
    --sim-sn-path=../data/processed/sim_sn_data.h5 \
    --hsc-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/result1 --seed=0 \
    --patience=200 --batch-size=1000 --norm \
    --input1=absolute-magnitude --input2=scaled-flux \
    --optimizer=adam --adabound-final-lr=1e-1 --lr=1e-3 \
    --eval-frequency=20 --mixup=mixup --threads=4 \
    --binary --remove-y --n-trials=100
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|n-trials|The number of search|int|

# Redshift regression
This is a regression task to predict the redshift from the flux  
This regression task is more difficult than the classification task.
Therefore, it is recommended to use a larger model.

**Usage**
```
python hsc_redshift.py learn \
    --sim-sn-path=../data/processed/sim_sn_data.h5 \
    --hsc-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/result1 --seed=0 \
    --n-highways=3 --hidden-size=500 --drop-rate=1e-2 \
    --patience=200 --batch-size=1000 --norm \
    --activation=sigmoid \
    --input1=magnitude --input2=scaled-flux \
    --optimizer=adam --adabound-final-lr=1e-1 --lr=1e-3 \
    --eval-frequency=20 --fold=-1 --cv=5 --threads=4 \
    --remove-y --use-batch-norm --target-redshift
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|sim-sn-path|File path of the training dataset|string|
|hsc-path|File path of the test dataset|string|
|model-dir|Output directory to save the learned models and prediction results|string|
|seed|Random seed|int|
|n-highways|The number of Highway layers|int|
|drop-rate|The ratio of dropout layer|float|
|patience|The model decides to finish to learn every `patience` epochs.|int|
|norm|The flag to normalize the input values|-|
|activation|The types of activation layer|relu, sigmoid, tanh, or identical|
|input1|The type of the main input values|magnitude or flux|
|input2|The type of the sub input values|none or scaled-flux|
|optimizer|The type of optimizer|momentum, adam, amsbound, or adabound|
|adabound-final-lr|The final learning rate, it is available when `optimizer` is adabound or amsbound|float|
|lr|Learning rate|float|
|eval-frequency|Evaluate test dataset every `eval-frequency` epochs.|int|
|fold|The index of the folds of cross validation, if `fold` is -1 then all folds are target|int|
|threads|The number of threads|int|
|cv|The number of folds of cross validation|int|
|remove-y|The flag to remove Y band or not|-|
|use-batch-norm|The flag to use batch normalization layer or not|-|
|target-distmod / target-redshift|The distmod is the target value if the flag `target-distmod` is set. the redshift value is the target if `target-redshift` is set.|- 

## Prediction
This script predicts the redshift of input data with the trained model.

**Usage**
```
python hsc_redshift.py predict \
    --data-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/result1/0 \
    --data-type=HSC --output-name=prediction.csv
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|data-path|File path of the dataset to predict|str|
|model-dir|Directory that the trained model is in|str|
|data-type|`SimSN` for training dataset, `HSC` for test dataset|`SimSN` or `HSC`|
|output-name|File name to output the predicted results (the file is created in `model-dir`)|str|

## Hyper-parameter search of regression model

```
python hsc_redshift.py search \
    --sim-sn-path=../data/processed/sim_sn_data.h5 \
    --hsc-path=../data/processed/hsc_data.h5 \
    --model-dir=../models/search2 --seed=0 \
    --patience=200 --batch-size=1000 --norm \
    --input1=absolute-magnitude --input2=scaled-flux \
    --optimizer=adam --adabound-final-lr=1e-1 --lr=1e-3 \
    --eval-frequency=20 --threads=4 \
    --remove-y --target-redshift --n-trials=100
```

|option name| description| value type / choices|
|:-------|:------|:-------|
|n-trials|the number of search|int|

# Acknowledgments

This project is supported by JST CREST Grant NumberJPMHCR1414, JST AIP Acceleration Research Grant NumberJP20317829, MEXT KAKENHI Grant Numbers 18H04345, 17H06363, and JSPS KAKENHI GrantNumbers 18K03696, 16H02183, 19H00694, 20H00179.