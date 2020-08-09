import pickle as pkl
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from algo_training import algo_results

#Choosing the best algorithm

f_val = pd.read_csv('data/f_val.csv')
l_val= pd.read_csv('data/l_val.csv', header=None)
f_test = pd.read_csv('data/f_test.csv')
l_test = pd.read_csv('data/l_test.csv', header=None)

lrg = LogisticRegression()
sv = SVC()
mp = MLPClassifier()
rf = RandomForestClassifier()
dtc = DecisionTreeClassifier()
boost = GradientBoostingClassifier()
hgb = HistGradientBoostingClassifier()

#read in all of the pickles
algo_names = ['lrg', 'sv', 'mp', 'rf', 'dtc', 'boost', 'hgb']
algorithms = {'lrg':lrg, 'sv':sv, 'mp':mp, 'rf':rf, 'dtc':dtc, 'boost':boost, 'hgb':hgb}
algo_pkl = {}
for name in algo_names: #get pkl objects
	pklfile = open('pkl/{}pickle.pkl'.format(name), 'rb') #opens pickle and stores it in an array  
	algo_pkl[name] = pkl.load(pklfile)
	pklfile.close()

#Testing best scores on validation set
for name in zip(algo_pkl, algorithms):
    print("Score for {}".format(name)) #assigning test score
    parameters = algo_pkl[name].best_params_
    temp_cv = GridSearchCV(estimator=algorithms[name], param_grid=parameters, cv=10, scoring='accuracy', refit=True)
    temp_cv.fit(f_val, l_val.values.ravel())
    algo_results(temp_cv)