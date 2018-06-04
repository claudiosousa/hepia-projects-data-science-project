"""


Author:      Claudio Sousa, David Gonzalez
Data source: http://sci2s.ugr.es/keel/dataset.php?cod=189
"""
from pprint import pprint
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd


def get_data():
    filename = '../data/titanic.dat'
    data = pd.read_csv(filename)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data_normalized = pd.DataFrame(np_scaled)
    data_normalized.columns = data.columns
    return data_normalized

data = get_data()
variables = ['Class', 'Age', 'Sex']

markers = ['o', 's', '^', 'P']
colormap = LinearSegmentedColormap.from_list('survived', [(1,0,0), (0,1,0)], N=20)


def plot_inertia(data):
    limit = 13
    k_inertia = []
    for k in range(1, limit):
        kmeans = KMeans(n_clusters=k, max_iter=300, random_state=0)
        labels = kmeans.fit_predict(data)
        k_inertia.append([k, kmeans.inertia_])

    k_inertia = list(zip(*k_inertia))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(range(0, limit))
    ax.set_title("Geometric median")
    ax.set_xlabel('k')
    ax.set_ylabel('$\sum_{i=0}^{n}(C_i - s_i)^2$')
    plt.grid()
    plt.plot(k_inertia[0],k_inertia[1])


def group_data_by_variables(data):
    data = data.groupby(variables)
    count = data.count().values[:, 0]
    mean = data.mean().values[:, 0]
    points = np.array([list(p) for p,_ in data.groups.items()])
    return points, count, mean

def plot_3d(points, count, mean):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],c=mean, cmap='RdYlGn', edgecolor='k', s=80, alpha=1)
    cbar = fig.colorbar(sc, orientation='horizontal')
    cbar.set_label('Survived', rotation=0)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Class')
    ax.set_ylabel('Age')
    ax.set_zlabel('Sex')
    ax.set_title('Survived')

    for p, c, m in zip(points, count, mean):
       ax.text(p[0], p[1]+0.05, p[2], f'{c}({m:.0%})', zorder=10, color='k')

points, count, mean = group_data_by_variables(data)
grouped_features = np.array(list(zip(points[:, 0], points[:, 1], points[:, 2], mean * 3)))
plot_inertia(grouped_features)

k = 4
kmeans = KMeans(n_clusters=k, max_iter=300, random_state=0)
labels = kmeans.fit_predict(grouped_features)
centroids = kmeans.cluster_centers_


plot_3d(points, count, mean)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, c in enumerate(centroids):
    indices = [li for (li, c) in enumerate(labels) if c == i]
    ax.scatter(c[0], c[1], c[2], marker=markers[i], edgecolor='k', s=200, c=colors[i], alpha=1)
    ax.text(c[0], c[1]+0.05, c[2], sum(count[indices]), zorder=10, color=colors[i])
    ax.scatter(points[indices, 0],points[indices, 1], points[indices, 2], marker=markers[i], edgecolor='k', s=80, c=colors[i], alpha=1)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Class')
ax.set_ylabel('Age')
ax.set_zlabel('Sex')
ax.set_title('Clusters')


plt.show()

