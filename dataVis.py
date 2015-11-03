import matplotlib.pyplot as pyp
import numpy as np
import seaborn as sns
import psycopg2,random
import datetime as dt
from processingFunctions import *


heat = np.array([[58.4,63.34, 60.70,60.8 ,54],[64.8,61.14, 55.38, 63.99, 60.05],[66.5,60.6, 60.54,57.52,58.14]])
print(heat)
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



