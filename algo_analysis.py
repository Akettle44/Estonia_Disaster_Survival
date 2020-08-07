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

#Printing function for scores and metrics
def algo_results(results):
	mts = results.cv_results_['mean_test_score'] #avg score
	tol = results.cv_results_['std_test_score']  #toleranc
	optimal = results.best_params_
	print('Best parameters: {}'.format(optimal))
	for mean, std, params in zip(mts, tol, results.cv_results_['params']):
		print("%0.3f (+/-%0.3f) for %r" % (mean, std*2, params))

### Algorithm portion ###

#Fitting Logistic regression alogrithm
lrg = LogisticRegression()
c_list = [0.01, 0.1, 1, 10, 100] #C = .01 and C performed best (.872)-> High regularization 
parameters = { 'C': c_list }
lrg_cv = GridSearchCV(estimator=lrg, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
lrg_cv.fit(f_train, l_train.values.ravel())
algo_results(lrg_cv) 

#Fitting support vector machine alogrithm
sv = SVC() 
c_list = [0.01, 0.1, 1, 10, 100] 
parameters = { 'C': c_list }
sv_cv = GridSearchCV(estimator=sv, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
sv_cv.fit(f_train, l_train.values.ravel())
algo_results(sv_cv)

#Fitting Multilayer Perceptron
mp = MLPClassifier() 
hidden_sz = [5, 10, 20, 50, 100, 200]
activ_func = ['logistic', 'tanh', 'relu']
learning_rte = ['constant', 'invscaling', 'adaptive']
parameters = {'activation':activ_func, 'learning_rate':learning_rte, 'hidden_layer_sizes':hidden_sz}
mp_cv = GridSearchCV(estimator=mp, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
mp_cv.fit(f_train, l_train.values.ravel())
algo_results(mp_cv)