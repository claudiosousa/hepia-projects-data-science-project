"""
Launch a cross-validation on three models to see which is best on wine data.
Model to test: KNeighborsClassifier, DecisionTreeClassifier and MLPClassifier.

Author: Claudio Sousa, David Gonzalez
"""

from sklearn import datasets
from cross_validation import cross_validate, plot_validation
from models import instanciate_kneighbors_model, instanciate_decisiontree_model, instanciate_mlp_model

data = datasets.load_wine()

models = [
    instanciate_kneighbors_model(1, 11),
    instanciate_decisiontree_model(1, 11),
    instanciate_mlp_model(10)
]

best_model = cross_validate(data, models, 5, 10)
plot_validation(models, best_model)
