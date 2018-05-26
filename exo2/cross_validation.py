"""
Functions that help determine the best model by cross-validation: http://scikit-learn.org/stable/model_selection.html

Certain model suffer from scale, so it is important to normalise: http://scikit-learn.org/stable/modules/preprocessing.html

Author: Claudio Sousa, David Gonzalez
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
import numpy as np

# Constants
CV_SEGMENT = 5
CV_REPEAT = 10

def cross_validate(data, models):
    """
    Determine the best model by cross-validation.
    Fill the tests result for each model variations,
    compute their average and get the best model variation by model type

    Keyword arguments:
    data -- data on which to test the models
    models -- instanciated model variations to test
    """
    rkf = RepeatedKFold(n_splits=CV_SEGMENT, n_repeats=CV_REPEAT)
    for m in models:
        for mvar in m["model_variations"]:
            mvar["tests"] = []
            for train_index, test_index in rkf.split(data.data):
                mvar["model"].fit(data.data[train_index], data.target[train_index])
                mvar["tests"].append(mvar["model"].score(data.data[test_index], data.target[test_index]))

            mvar["avg"] = float(np.mean(mvar["tests"]))

        m["best"] = m["model_variations"][np.array([mvar["avg"] for mvar in m["model_variations"]]).argmax()]

    # Get the best model from all best model variations
    return models[np.array([m["best"]["avg"] for m in models]).argmax()]

def plot_validation(models, best_model):
    """
    Plot the cross-validation data generated by cross_validate()

    Keyword arguments:
    models -- models data filled by cross_validate()
    best_model -- best model determined by cross_validate()
    """
    fig = plt.figure()
    disposition = 100 + (len(models) * 10) + 1
    for m_i, m in enumerate(models):
        ax = fig.add_subplot(disposition + m_i)
        ax.scatter(
            [[mvar_i] * len(mvar["tests"]) for mvar_i, mvar in enumerate(m["model_variations"])],
            #[[m_i] * len(mvar["tests"]) for mvar in m["model_variations"]],
            [mvar["tests"] for mvar in m["model_variations"]],
            c="black",
            s=40
        )
        ax.scatter(
            [mvar_i for mvar_i, mvar in enumerate(m["model_variations"])],
            #[m_i for mvar in m["model_variations"]],
            [mvar["avg"] for mvar in m["model_variations"]],
            c=["tab:gray" if m["best"] != mvar else "red" for mvar in m["model_variations"]],
            s=200
        )

        ax.grid(True)
        ax.axis(ymin=0, ymax=1)
        ax.set_title(m["model_constructor"].__name__ + (" (best)" if m == best_model else ""))
        ax.set_xlabel("Model variations")
        ax.set_ylabel("Performance")
    plt.show()
