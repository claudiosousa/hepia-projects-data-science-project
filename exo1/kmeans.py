"""

Data source: http://sci2s.ugr.es/keel/dataset.php?cod=189
"""
from pprint import pprint
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import pandas as pd


filename = 'titanic.dat'
data = pd.read_csv(filename)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data)
data_normalized = pd.DataFrame(np_scaled)
data_normalized.columns = data.columns
data = data_normalized

survived_colors = [[1, 0, 0, .2] if s == 0 else [0, 1, 0, .2] for s in data.Survived]
markers = [m for m in Line2D.filled_markers if m != '8']

best_inertia = float("inf")
for k in range(4, 5):
    kmeans = KMeans(n_clusters=k, max_iter=100000, random_state=0)
    labels = kmeans.fit_predict(data)
    if kmeans.inertia_ > best_inertia:
        break
    best_inertia = kmeans.inertia_
    print(best_inertia)
    print(k)
    centroids = kmeans.cluster_centers_

fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')

ax.scatter(data.Class, data.Age, data.Sex, c=survived_colors, edgecolor='k', s=80)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Class')
ax.set_ylabel('Age')
ax.set_zlabel('Sex')
ax.set_title('Survived')


ax = fig.add_subplot(212, projection='3d')
for i, c in enumerate(centroids):
    ax.scatter(c[0], c[1], c[2], marker=markers[i], edgecolor='k', s=200)
    ax.scatter(data.Class[labels == i], data.Age[labels == i], data.Sex[labels == i], marker=markers[i], edgecolor='k', s=80)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Class')
ax.set_ylabel('Age')
ax.set_zlabel('Sex')
ax.set_title('Clusters')


fig = plt.figure()
# 2d charts
for i, (var1, var2) in enumerate(combinations(['Class', 'Age', 'Sex'], 2)):
    ax = fig.add_subplot(311 + i)
    ax.scatter(data[var1], data[var2], c=survived_colors, edgecolor='k', s=80)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)

plt.show()
"""
c_mean_distances = []
for i, centroid in enumerate(centroids):
    mean_distance = centroid_mean_distance(data, i, centroid, labels)
    c_mean_distances.append(mean_distance)

pprint(c_mean_distances)


pred_data = np.array([[0, 0], [4, 4]])
pred = kmeans.predict(pred_data)

plt.scatter(test_data[:, 0], test_data[:, 1], c=lab)
plt.scatter(pred_data[:, 0], pred_data[:, 1], s=200, c=pred, marker='o', alpha=0.6)
plt.show()
print(lab)
print(pred)
"""
