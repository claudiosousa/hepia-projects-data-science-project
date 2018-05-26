"""
Launch a cross-validation on three models to see which is best on wine data.

Model to test:
  - KNeighborsClassifier:   http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  - DecisionTreeClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  - MLPClassifier:          http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Author: Claudio Sousa, David Gonzalez
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from cross_validation import *

CV_MLP_MAX_LAYER = 2

# Data to work with
data = datasets.load_wine()

# Data structure to hold models and generated info
models = [
    {
        "model_constructor": KNeighborsClassifier,
        "model_variations": [
            {
                "model": KNeighborsClassifier(n_neighbors=i)
            } for i in range(1, 11)
        ]
    },
    {
        "model_constructor": DecisionTreeClassifier,
        "model_variations": [
            {
                "model": DecisionTreeClassifier(min_samples_leaf=i)
            } for i in range(1, 11)
        ]
    }
]

best_model = cross_validate(data, models)
plot_validation(models, best_model)
