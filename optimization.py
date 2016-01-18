import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from operator import itemgetter
import numpy as np
from adasyn import ADASYN

def deleteClass(X, y, num, c):
    """Delete 'num' samples from class=c in StudentLife dataset stress reports
    """

    twoIndex = np.array([i for i in range(len(y)) if y[i] == c])
    np.random.shuffle(twoIndex)

    delIndex = twoIndex[0:num]

    X = np.delete(X, delIndex, 0)
    y = np.delete(y, delIndex, 0)

    return(X, y)

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

X = np.load('data/X51.npy')
Y = np.load('data/y51.npy')

	# fixes errors with Nan data
X = preprocessing.Imputer().fit_transform(X)
print(X.shape,Y.shape)

   # Recursive oversampling and undersampling
adsn = ADASYN(imb_threshold=0.5,ratio=0.7)
X,Y = adsn.fit_transform(X,Y)
X,Y = deleteClass(X,Y,100,2)
print(int(np.sqrt(X.shape[1])))

# Create the RFE object and compute a cross-validated score.
rf = RandomForestClassifier(n_jobs=-1)
gbm =xgb.XGBClassifier(n_estimators=300)
# The "accuracy" scoring is proportional to the number of correct
# classifications

param_dist = {"n_estimators": [10,50,100,150,300],
                "criterion": ['gini', 'entropy'],
                "bootstrap": [True,False],
                "max_features": [10,20,30,40,45,48],
                "class_weight": ['auto']}
param_dist_xgb = {"max_depth": [5,10,15,25,30],
            "learning_rate": [0.001,0.01,0.2,0.5,0.7],
            "subsample": [0.3,0.5,0.9,1],
            "gamma": [0.001,0.01,0.2,0.7,2],
            "colsample_bytree": [0.3,0.5,1],
            "objective": ["multi:softmax"]}


n_iter=500
random_search = RandomizedSearchCV(gbm,param_distributions=param_dist_xgb)
random_search.fit(X,Y)
report(random_search.grid_scores_)


random_search = RandomizedSearchCV(rf,param_distributions=param_dist_xgb)
random_search.fit(X,Y)
report(random_search.grid_scores_)

