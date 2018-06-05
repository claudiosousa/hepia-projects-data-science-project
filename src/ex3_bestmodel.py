"""
Launch a cross-validation on three models to see which is best on custom data from:
    https://archive.ics.uci.edu/ml/datasets.html

Chosen data: https://archive.ics.uci.edu/ml/datasets/Leaf
Tested models: SVC, RandomForestClassifier, DecisionTreeClassifier.

Author: Claudio Sousa, David Gonzalez
"""

from sklearn import datasets
from cross_validation import cross_validate, plot_validation, normalise_data
from models import instanciate_svc_model, instanciate_randomforest_model, instanciate_decisiontree_model
import numpy as np
from sklearn.datasets.base import Bunch
import pandas as pd

csv = pd.read_csv("../data/leaf.csv")
data = Bunch(
    data=np.array([list(d[1:]) for d in csv.values]),
    target=np.array([d[0] for d in csv.values])
)
data.data = normalise_data(data.data)

models = [
    instanciate_svc_model(),
    instanciate_randomforest_model(1, 11),
    instanciate_decisiontree_model(1, 11)
]

best_model = cross_validate(data, models, 5, 10)
plot_validation(models, best_model)
