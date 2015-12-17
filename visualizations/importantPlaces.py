import time

import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
from sklearn import neighbors
from sklearn.metrics.pairwise import pairwise_distances


def distMatrix(X):
    distM = np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            if j>i:
                d = great_circle(X[i,:],X[j,:]).meters
                distM[i,j] = d
    np.save('distmat.npy',distM)
    print(distM)

uids1=['u16','u19','u44','u24']

n_clusters = 5
X = []
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()

for u in uids1:
    cur.execute('SELECT latitude,longitude FROM {0} WHERE longitude<=-72.2'.format(u+'gpsdata'))
    X += cur.fetchall()
print(type(X))

X = np.array(X)
print(X.shape)

# Setting DBSCAN parameters and clustering student data
db = DBSCAN(eps=0.002, min_samples=8,metric='haversine', algorithm='auto').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('dbscan.png')