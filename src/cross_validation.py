"""
Functions that help determine the best model by cross-validation: http://scikit-learn.org/stable/model_selection.html

Certain model suffer from scale, so it is important to normalise: http://scikit-learn.org/stable/modules/preprocessing.html

Author: Claudio Sousa, David Gonzalez
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv

def normalise_data(data):
    """
    Transforms features by scaling each feature to the interval [0,1].

    Keyword arguments:
    data -- data with features to normalize
    """
    min_max_scaler = MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def cross_validate(data, models, n_split, n_repeat):
    """
    Determine the best model by cross-validation.
    Fill the tests result for each model variations,
    compute their average and get the best model variation by model type

    Keyword arguments:
    data -- data on which to test the models
    models -- instanciated model variations to test
    n_split -- number of split for training and test set
    n_repeat -- number of times the training will be repeated with different combination
    """
    rkf = RepeatedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=0)
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
        for mvar_i, mvar in enumerate(m["model_variations"]):
            ax.scatter([mvar_i] * len(mvar["tests"]), mvar["tests"], s=40)
            ax.scatter(mvar_i, mvar["avg"], c="black" if m["best"] != mvar else "red", s=400, marker="_")

        ax.grid(True)
        ax.axis(ymin=0, ymax=1)
        ax.set_title(m["model_constructor"].__name__ + (" (best)" if m == best_model else ""))
        ax.set_xlabel(m["x_label"])
        ax.set_ylabel("Performance")

        plt.sca(ax)
        plt.xticks(range(len(m["model_variations"])), map(lambda v: v['label'], m["model_variations"]))
        if "rotate_x_labels" in m:
            plt.xticks(rotation=m["rotate_x_labels"])

    plt.show()

def output_csv(models, best_model, name):
    """
    Output into a CSV the data generated by cross_validate()

    Keyword arguments:
    models -- models data filled by cross_validate()
    best_model -- best model determined by cross_validate()
    name -- name of the dataset
    """
    with open(name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for m in models:
            writer.writerow([mvar["avg"] for mvar in m["model_variations"]])
