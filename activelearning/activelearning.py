import json,csv,sys,os,psycopg2,random
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import warnings








def main():
	print('-----------------------------')
	print('| Active Learning Activated |')
	print('-----------------------------')
	
	X=np.load('numdata/withgps/epochFeats.npy')
	Y=np.load('numdata/withgps/epochLabels.npy')
	




if __name__ == '__main__':
	main()