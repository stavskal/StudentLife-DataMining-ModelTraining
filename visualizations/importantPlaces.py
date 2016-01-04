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

uids1=['u00','u02','u52']
X=[]
# Connecting to DB to fetch data
con = psycopg2.connect(database='dataset', user='tabrianos')                                                                        
cur = con.cursor()                              

for u in uids1:
    cur.execute('SELECT latitude,longitude FROM {0} WHERE latitude>=43.60 AND latitude<=43.75 AND longitude>-72.35 AND longitude<-72.20'.format(u+'gpsdata'))
    X += cur.fetchall()
# Converting to numpy for compatibility with sklearn
X = np.array(X)

# Setting DBSCAN parameters and clustering student data
db=DBSCAN(eps=0.004, min_samples=5,metric='haversine', algorithm='auto').fit(X)
y = DBSCAN(eps=0.004, min_samples=5,metric='haversine', algorithm='auto').fit_predict(X)



core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
print(db.labels_)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

centers = np.zeros((n_clusters_,2))
bbox = np.zeros((n_clusters_,4))
#Discovering Bounding Box coordinates (kinda sloppy but works)
for i in range(0,n_clusters_):
    coords=[]
    for j in range(0,len(y)):
        if y[j] == i:
            coords.append(X[j,:])
    # coords holds all gps coordinate pairs that correspond to one label per loop
    coords = np.array(coords)
    #          min lat, min long        max lat, max long
    bbox[i,:] = [min(coords[:,0]),min(coords[:,1]),max(coords[:,0]),max(coords[:,1])]
    centers[i,:] = [ (bbox[i,0]+bbox[i,2])/2 , (bbox[i,1]+bbox[i,3])/2]
print(centers)
# save for further use
np.save('bbox.npy',bbox)
np.save('clustercenters.npy',centers)


# Plot results
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    print(class_member_mask)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=8)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=4)
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('dbscan1.png')