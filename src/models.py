"""
Utility functions for instancing models.

Author: Claudio Sousa, David Gonzalez
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct

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
                "model": KNeighborsClassifier(n_neighbors=i),
                "label": i
            } for i in range(n_neighbors_min, n_neighbors_max, n_neighbors_step)
        ],
        "x_label": "K-neighbors"
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
                "model": DecisionTreeClassifier(min_samples_leaf=i),
                "label": i
            } for i in range(min_samples_leaf_min, min_samples_leaf_max, min_samples_leaf_step)
        ],
        "x_label": "Min samples leaf"
    }

def instanciate_mlp_model():
    """
    Instanciate MLPClassifier with all different solvers and activator functions.
    http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    """
    return {
        "model_constructor": MLPClassifier,
        "model_variations": [
            {
                "model": MLPClassifier(solver=solver, activation=activator,
                                       max_iter=1000, hidden_layer_sizes=2,
                                       learning_rate_init=0.1, early_stopping=True),
                "label": f'{solver[:5]},{activator[:4]}'
            } for solver in ['lbfgs', 'sgd', 'adam']
            for activator in ['identity', 'logistic', 'tanh', 'relu']
        ],
        "x_label": "Solver & activator",
        "rotate_x_labels": 90
    }

def instanciate_svc_model():
    """
    Instanciate SVC with different kernel.
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    return {
        "model_constructor": SVC,
        "model_variations": [
            {
                "model": SVC(kernel=kernel),
                "label": kernel
            } for kernel in ['linear', 'poly', 'rbf', 'sigmoid']
        ],
        "x_label": "Kernel",
        "rotate_x_labels": 90
    }

def instanciate_randomforest_model(min_samples_leaf_min, min_samples_leaf_max, min_samples_leaf_step=1):
    """
    Instanciate RandomForestClassifier by varying the 'min_samples_leaf' parameter.
    http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Keyword arguments:
    min_samples_leaf_min -- parameter variation start
    min_samples_leaf_max -- parameter variation end
    min_samples_leaf_step -- parameter variation step
    """
    return {
        "model_constructor": RandomForestClassifier,
        "model_variations": [
            {
                "model": RandomForestClassifier(min_samples_leaf=i),
                "label": i
            } for i in range(min_samples_leaf_min, min_samples_leaf_max, min_samples_leaf_step)
        ],
        "x_label": "Min samples leaf"
    }

def instanciate_gaussianprocess_model():
    """
    Instanciate GaussianProcessClassifier with different kernels.
    http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
    """
    return {
        "model_constructor": GaussianProcessClassifier,
        "model_variations": [
            {
                "model": GaussianProcessClassifier(kernel=kernel(), random_state=0, n_jobs=-1),
                "label": kernel.__name__
            } for kernel in [ConstantKernel, WhiteKernel, RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct]
        ],
        "x_label": "Kernel",
        "rotate_x_labels": 90
    }
