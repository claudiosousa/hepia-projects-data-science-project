"""
Utility functions for instancing models.

Author: Claudio Sousa, David Gonzalez
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def instanciate_kneighbors_model(n_neighbors_min, n_neighbors_max, n_neighbors_step=1):
    """
    Instanciate KNeighborsClassifier by varying the 'n_neighbors' parameter.
    http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Keyword arguments:
    n_neighbors_min -- parameter variation start
    n_neighbors_max -- parameter variation end
    n_neighbors_step -- parameter variation step
    """
    return {
        "model_constructor": KNeighborsClassifier,
        "model_variations": [
            {
                "model": KNeighborsClassifier(n_neighbors=i)
            } for i in range(n_neighbors_min, n_neighbors_max, n_neighbors_step)
        ]
    }

def instanciate_decisiontree_model(min_samples_leaf_min, min_samples_leaf_max, min_samples_leaf_step=1):
    """
    Instanciate DecisionTreeClassifier by varying the 'min_samples_leaf' parameter.
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Keyword arguments:
    min_samples_leaf_min -- parameter variation start
    min_samples_leaf_max -- parameter variation end
    min_samples_leaf_step -- parameter variation step
    """
    return {
        "model_constructor": DecisionTreeClassifier,
        "model_variations": [
            {
                "model": DecisionTreeClassifier(min_samples_leaf=i)
            } for i in range(min_samples_leaf_min, min_samples_leaf_max, min_samples_leaf_step)
        ]
    }

def instanciate_mlp_model(n):
    """
    Instanciate MLPClassifier n times.
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    Keyword arguments:
    n -- number of MLP to instanciate
    """
    return {
        "model_constructor": MLPClassifier,
        "model_variations": [
            {
                "model": MLPClassifier(solver='lbfgs', activation='logistic',
                                       max_iter=1000, hidden_layer_sizes=2,
                                       learning_rate_init=0.1, early_stopping=True)
            } for i in range(n)
        ]
    }
