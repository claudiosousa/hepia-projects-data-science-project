"""
Launch a cross-validation on three models to see which is best on wine data.
Model to test: KNeighborsClassifier, DecisionTreeClassifier and MLPClassifier.

Author: Claudio Sousa, David Gonzalez
"""

from sklearn import datasets
from cross_validation import cross_validate, plot_validation, output_csv, normalise_data
from models import instanciate_kneighbors_model, instanciate_decisiontree_model, instanciate_mlp_model

data = datasets.load_wine()
data.data = normalise_data(data.data)

models = [
    instanciate_kneighbors_model(1, 11),
    instanciate_decisiontree_model(1, 11),
    instanciate_mlp_model()
]

best_model = cross_validate(data, models, 5, 10)
output_csv(models, best_model, "wine")
plot_validation(models, best_model)
