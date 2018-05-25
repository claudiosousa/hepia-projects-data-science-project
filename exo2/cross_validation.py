"""
Detemine the best model by cross-validation: http://scikit-learn.org/stable/model_selection.html

Model to test:
  - KNeighborsClassifier:   http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
  - DecisionTreeClassifier: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
  - MLPClassifier:          http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Certain model suffer from scale, so it is important to normalise: http://scikit-learn.org/stable/modules/preprocessing.html

Author: Claudio Sousa, David Gonzalez
"""

from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
CV_MLP_MAX_LAYER = 2

# Data to work with
#data = datasets.load_wine()
data = datasets.load_breast_cancer()

# Data structure to hold models and generated info
models = [
    {
        "model_constructor": KNeighborsClassifier,
        "model_variations": [
            {
                "model": KNeighborsClassifier(n_neighbors=i),
                "tests": [],
                "avg": 0,
                "val": i
            } for i in range(1, 11)] ,
        "best": None
    },
    {
        "model_constructor": DecisionTreeClassifier,
        "model_variations": [
            {
                "model": DecisionTreeClassifier(min_samples_leaf=i),
                "tests": [],
                "avg": 0,
                "val": i
            } for i in range(1, 11)] ,
        "best": None
    }
]
best_model = None

# Fill the tests result for each model variations,
# compute their average and get the best model variation by model type
rkf = RepeatedKFold(n_splits=CV_SEGMENT, n_repeats=CV_REPEAT)
for m in models:
    for mvar in m["model_variations"]:
        for train_index, test_index in rkf.split(data.data):
            mvar["model"].fit(data.data[train_index], data.target[train_index])
            mvar["tests"].append(mvar["model"].score(data.data[test_index], data.target[test_index]))

        mvar["avg"] = float(np.mean(mvar["tests"]))

    m["best"] = m["model_variations"][np.array([mvar["avg"] for mvar in m["model_variations"]]).argmax()]

# Get the best model from all best model variations
best_model = models[np.array([m["best"]["avg"] for m in models]).argmax()]

# Plot all
ax = plt.figure().add_subplot(111, projection='3d')
plt.title("Models performance by variations, their average, their best average and best")
for m_i, m in enumerate(models):
    ax.scatter(
        [[mvar_i] * len(mvar["tests"]) for mvar_i, mvar in enumerate(m["model_variations"])],
        [[m_i] * len(mvar["tests"]) for mvar in m["model_variations"]],
        [mvar["tests"] for mvar in m["model_variations"]],
        s=40
    )
for m_i, m in enumerate(models): # Remake loop to allow plt.legend() to get the right color for each model
    ax.scatter(
        [mvar_i for mvar_i, mvar in enumerate(m["model_variations"])],
        [m_i for mvar in m["model_variations"]],
        [mvar["avg"] for mvar in m["model_variations"]],
        c=["black" if m["best"] != mvar else "green" if m != best_model else "red" for mvar in m["model_variations"]],
        s=200
    )
plt.legend([m["model_constructor"].__name__ for m in models], loc='lower right')
plt.grid(True)
ax.set_xlabel("Model variations")
ax.set_ylabel("Model")
ax.set_zlabel("Performance")
plt.show()
