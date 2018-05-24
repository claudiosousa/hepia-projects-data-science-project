"""
Detemine the best model by cross-validation: http://scikit-learn.org/stable/model_selection.html

Model to test::
  - KNeighborsClassifier:   http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  - DecisionTreeClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  - MLPClassifier:          http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Certain model suffer from scale, so it is important to normalise: http://scikit-learn.org/stable/modules/preprocessing.html

Author: Claudio Sousa, David Gonzalez
"""

from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn import datasets
from sklearn import preprocessing
import numpy as np

# Constants
CV_SEGMENT = 5
CV_REPEAT = 10
CV_PARAM_NUM = 10
CV_MLP_MAX_LAYER = 2

# Data to work with
data = datasets.load_wine()
#data = datasets.load_breast_cancer()
X = data.data
Y = data.target

# Generate models to train
models = [KNeighborsClassifier]
model_values = [
    list(range(1, 11))
]
model_variations = [[] for _ in range(len(models))]
for v in model_values[0]:
    model_variations[0].append(models[0](n_neighbors=v))

# Cross-validate each model variation
# Array dimension: 1: model, 2: model variation, 3: subdataset index
scores = [
    [
        [
            [] for _ in range(CV_SEGMENT * CV_REPEAT)
        ] for _ in range(CV_PARAM_NUM)
    ] for _ in range(len(models))
]

rkf = RepeatedKFold(n_splits=CV_SEGMENT, n_repeats=CV_REPEAT)

for m_i, m in enumerate(model_variations):
    for mvar_i, model in enumerate(m):
        for train_i, indexes in enumerate(rkf.split(X)):
            train_index, test_index = indexes

            model.fit(X[train_index], Y[train_index])
            scores[m_i][mvar_i][train_i] = model.score(X[test_index], Y[test_index])

pprint(scores)

# Compute score average
# Array dimension: 1: model, 2: model variation
score_avgs = [
    [
        [] for _ in range(CV_PARAM_NUM)
    ] for _ in range(len(models))
]

for m_i, m in enumerate(scores):
    for mvar_i, mvar in enumerate(m):
       score_avgs[m_i][mvar_i] = float(np.mean(mvar))

pprint(score_avgs)

# Compute best value
for m_i, m in enumerate(score_avgs):
    print("Best K:", model_values[m_i][np.array(m).argmax()])
