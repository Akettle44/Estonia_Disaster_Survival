import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

#Setting up feauture and label headers
feature_headers = ['Sex','Age','Category']
label_headers = 'Survived'

#Reading in split data
f_train = pd.read_csv('data/f_training.csv')
l_train = pd.read_csv('data/l_training.csv', header=None)
f_val = pd.read_csv('data/f_val.csv')
l_val= pd.read_csv('data/l_val.csv', header=None)
f_test = pd.read_csv('data/f_test.csv')
l_test = pd.read_csv('data/l_test.csv', header=None)

if(1):
	print("Hello")

#Printing function for scores and metrics
def algo_results(results):
	mts = results.cv_results_['mean_test_score'] #avg score
	tol = results.cv_results_['std_test_score']  #toleranc
	for mean, std, params in zip(mts, tol, results.cv_results_['params']):
		print("%0.3f (+/-%0.2f for %r" % (mean, std*2, params))

### Algorithm portion ###

#Fitting Logistic regression alogrithm
lrg = LogisticRegression()
c_list = [0.01, 0.1, 1, 10, 100]
parameters = { 'C': c_list }
cv = GridSearchCV(lrg, parameters, cv=10) #using crossfold validation of 10 
cv.fit(f_train, l_train.values.ravel())
