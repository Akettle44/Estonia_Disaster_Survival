import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

#Setting up feauture and label headers
feature_headers = ['Sex','Age','Category']
label_headers = 'Survived'

#Reading in split data
f_train = pd.read_csv('data/f_training.csv')
l_train = pd.read_csv('data/l_training.csv', header=None)

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
lrg_output=open('results/lrgoutput.txt', 'w')
sys.stdout = lrg_output
algo_results(lrg_cv) 
lrg_file=open('lrgpickle.pkl', 'wb')
pkl.dump(lrg_cv, lrg_file)
lrg_output.close()
lrg_file.close()

#Fitting support vector machine alogrithm
sv = SVC() 
c_list = [0.01, 0.1, 1, 10, 100] 
parameters = { 'C': c_list }
sv_cv = GridSearchCV(estimator=sv, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
sv_cv.fit(f_train, l_train.values.ravel())
svm_output=open('results/svmoutput.txt', 'w')
sys.stdout = svm_output
algo_results(sv_cv)
sv_file=open('svpickle.pkl', 'wb')
pkl.dump(sv_cv, sv_file)
svm_output.close()
sv_file.close()

#Fitting Multilayer Perceptron
mp = MLPClassifier() 
hidden_sz = [5, 10, 20, 50, 100, 200]
activ_func = ['logistic', 'tanh', 'relu']
learning_rte = ['constant', 'invscaling', 'adaptive']
parameters = {'activation':activ_func, 'learning_rate':learning_rte, 'hidden_layer_sizes':hidden_sz}
mp_cv = GridSearchCV(estimator=mp, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
mp_cv.fit(f_train, l_train.values.ravel())
mp_output=open('results/mpoutput.txt', 'w')
sys.stdout = mp_output
algo_results(mp_cv)
mp_file=open('mppickle.pkl', 'wb')
pkl.dump(mp_cv, mp_file)
mp_file.close()
mp_output.close()

#Random Forest
rf = RandomForestClassifier()
n_estimators = [1, 2, 4, 6, 8, 10]
max_depth = [2, 4, 6, 10, 16, 22, 28, 36]
parameters={'n_estimators':n_estimators, 'max_depth':max_depth}
rf_cv = GridSearchCV(estimator=rf, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
rf_cv.fit(f_train, l_train.values.ravel())
rf_output=open('results/rfoutput.txt', 'w')
sys.stdout = rf_output #setting output file equal to the output of print
algo_results(rf_cv)
rf_file=open('rfpickle.pkl', 'wb')
pkl.dump(rf_cv, rf_file)
rf_file.close()
rf_output.close() #closing results file

#Decision Tree classifier
dtc = DecisionTreeClassifier()
n_estimators = [1, 2, 4, 6, 8, 10]
max_depth = [6, 10, 16, 22, 28, 36]
parameters={'max_depth':max_depth}
dtc_cv = GridSearchCV(estimator=dtc, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
dtc_cv.fit(f_train, l_train.values.ravel())
dtc_output=open('results/dtcoutput.txt', 'w')
sys.stdout = dtc_output #setting output file equal to the output of print
algo_results(dtc_cv)
dtc_file=open('dtcpickle.pkl', 'wb')
pkl.dump(dtc_cv, dtc_file)
dtc_file.close()
dtc_output.close()

#Gradient Boosted Trees
boost = GradientBoostingClassifier()
n_estimators = [1, 2, 4, 6, 8, 10]
max_depth = [6, 10, 16, 22, 28, 36]
learning_rate = [0.01, 0.1, 1, 10, 100]
parameters={'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate}
boost_cv = GridSearchCV(estimator=boost, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
boost_cv.fit(f_train, l_train.values.ravel())
boost_output=open('results/boostoutput.txt', 'w')
sys.stdout = boost_output #setting output file equal to the output of print
algo_results(boost_cv)
boost_file=open('boostpickle.pkl', 'wb')
pkl.dump(boost_cv, boost_file)
boost_file.close()
boost_output.close()

#Histogram Gradient Boosting classifier
hgb = HistGradientBoostingClassifier()
max_depth = [6, 10, 16, 22, 28, 36]
learning_rate = [0.001, 0.01, 0.1, 1, 10, 100]
parameters={'max_depth':max_depth, 'learning_rate':learning_rate}
hgb_cv = GridSearchCV(estimator=hgb, param_grid=parameters, cv=10, scoring='accuracy', refit=True)#using crossfold validation of 10 
hgb_cv.fit(f_train, l_train.values.ravel())
hgb_output=open('results/hgboutput.txt', 'w')
sys.stdout = hgb_output #setting output file equal to the output of print
algo_results(hgb_cv)
hgb_file=open('pkl/hgbpickle.pkl', 'wb')
pkl.dump(hgb_cv, hgb_file)
hgb_file.close()
hgb_output.close()