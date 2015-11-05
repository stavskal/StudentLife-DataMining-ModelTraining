import matplotlib.pyplot as pyp
import numpy as np
import seaborn as sns
import psycopg2,random
import datetime as dt
from processingFunctions import *

sns.set_style('darkgrid')
sns.set(color_codes=True)
#heat = np.array([[62.06,63.34, 60.70,60.8 ,54],[60.44,61.14, 55.38, 63.99, 60.05],[67.1,60.6, 60.54,57.52,58.14]])
#print(heat.shape)

"""
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


"""
#               max             		average            			min
y = np.array([[70.3,74,69.2,69.2,84.6],[61.4,52.6,54.6,50.1,58.2],[50,31.8,41.66,50,25]])
x=[0,1,2,3,4]
xtic=['120','100','70','50','35']
pyp.xticks(x, xtic)
pyp.title('Accuracy performance with User-Specific Time Window')
pyp.xlabel('No. of Most Common Apps kept')
pyp.ylabel('Accuracy over Users (%)')
ax = sns.tsplot(data=y)
fig = ax.get_figure()
fig.savefig('coolplot.png')