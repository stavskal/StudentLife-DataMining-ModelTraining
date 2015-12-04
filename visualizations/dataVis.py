import matplotlib.pyplot as pyp
import numpy as np
import seaborn as sns
import psycopg2,random
import datetime as dt
from unbalanced_dataset import UnderSampler
from sklearn.cluster import KMeans

#from processingFunctions import *

sns.set_style('darkgrid')
sns.set(color_codes=True)

x=['Feeling Great','Feeling Good','A little stressed','Definitely Stressed','Stressed Out']
y=[62,286,292,212,156]

ax = sns.barplot(x=x,y=y)
pyp.ylim(0,592)
pyp.title('Manually balanced distribution of stress responses')
fig = ax.get_figure()
fig.savefig('LabelDistAfter.png')
exit()

con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
cur.execute('SELECT latitude,longitude FROM u33gpsdata')
records = cur.fetchall()



print(len(records))
longitude = [item[1] for item in records ]
latitude = [item[0] for item in records ]

pyp.scatter(latitude,longitude)
pyp.savefig('gpsplot.png')



"""
#heat = np.array([[62.06,63.34, 60.70,60.8 ,54],[60.44,61.14, 55.38, 63.99, 60.05],[67.1,60.6, 60.54,57.52,58.14]])
#print(heat.shape)
uids=['u16','u19','u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','Mean']

#user specific    
x = np.array([67.33,75.79,65.6,79.3,83.5,93.27,43.8,74.64,51.4,66,58.57,56.68,55.97,70.5,60.8,54.4])
x = np.append(x,x.mean())


#LOSO
y = np.array([58.87,78.02,47.12,61.53,78.88,86.36,57.62,68.65,45.94,76.6,58.53,59.43,50,34,55.84,34.4])
y = np.append(y,y.mean())
print(x.shape,y.shape,x.mean(),y.mean())
listaa=[x.min(),y.min(),x.max(),x.max()]

mini = min(listaa)
maxi = max(listaa)
print(listaa)
pyp.scatter(x,y, marker='o')
pyp.xlim(mini-4,maxi+6.7)
pyp.ylim(mini-4,maxi+6.7)
pyp.gca().set_aspect('equal', adjustable='box')
pyp.xlabel('User-Specific Model Accuracy')
pyp.ylabel('Group Model Accuracy (LOSO)')
pyp.title('Comparison of Personalized and Generic Models')

for label, xi, yi in zip(uids,x,y):
	if label!='Mean':
		fc='yellow'
	else:
		fc='red'
	pyp.annotate(
		label,
		xy=(xi,yi), xytext=(0,15),
		textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.3', fc = fc, alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

pyp.savefig('losoUser1.png')

ax1 = sns.heatmap(heat, vmin=50, vmax=65)

pyp.title('Cross-Validation Accuracy Heatmap')

xA = np.array([i for i in range(0,5)])
yA = np.array([i for i in range(0,3)])
ytic = ['100','60','25']
xtic = ['Two days','One day', '3/4 day','Half day', 'Quarter of day']
pyp.xticks(xA, xtic)
pyp.yticks(yA, ytic)
sb = ax1.get_figure()
sb.savefig('heatmap2.png')
print('done baby')

pyp.figure()
y1=np.array([]   [62.98, 58.8, 61.25, 60.07, 60.09, 61.11 ,55.2])
x1 = [0,1,2,3,4,5,6]
xtic = ['100','70','60','50','40','30','20']
pyp.xticks(x1,xtic)
pyp.title('Accuracy performance with User-Specific Time Window')
pyp.xlabel('No. of Most Common Apps kept')
pyp.ylabel('Averaged accuracy over Users (%)')

ax = sns.tsplot(data=y1, estimator=np.median)
fig = ax.get_figure()
fig.savefig('meanStressWinTRY.png')


#               max             		average            			min
y1 = np.array([[70.3,74,69.2,69.2,84.6],[61.4,52.6,54.6,50.1,58.2],[50,31.8,41.66,50,25]])

y = np.array([[90.3,74,69.2,69.2,84.6],[61.4,52.6,54.6,50.1,58.2],[50,31.8,41.66,50,25]])

x=[0,1,2,3,4]
xtic=['120','100','70','50','35']
pyp.xticks(x, xtic)
pyp.title('Accuracy performance with User-Specific Time Window')
pyp.xlabel('No. of Most Common Apps kept')
pyp.ylabel('Accuracy over Users (%)')
ax = sns.tsplot(data=y)
fig = ax.get_figure()
fig.savefig('coolplot.png')




uids1=['u00','u24','u36','u19']
uids2=['u00','u24','u36','u19','u52','u16','u59']


con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
labels =[]
for u in uids2:
	records = loadStressLabels(cur,u)
	labels += [i[1] for i in records]

print(labels)

x=[0,1]
xtic=['zero', 'one']
pyp.hist(labels)
pyp.savefig('tttt123.png')
ax = sns.distplot(labels, kde=False)
fig = ax.get_figure()	
fig.savefig('ttttt.png')

"""