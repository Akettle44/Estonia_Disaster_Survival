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
from sklearn.utils import class_weight
from collections import Counter

#Setting up feauture and label headers
feature_headers = ['Sex','Age','Category']
label_headers = 'Survived'

#Reading in split data
f_train = pd.read_csv('data/f_training.csv')
l_train = pd.read_csv('data/l_training.csv', header=None)
#class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=(np.unique(l_train)), y=l_train.values.ravel()) #computes the class weight
#class_weights = {1:0.580, 0:3.6094}
class_weights = {1:0.139, 0:0.861}
class_wght = [class_weights]

#Algorithm names and functions
lrg = LogisticRegression()
sv = SVC()
mp = MLPClassifier()
rf = RandomForestClassifier()
dtc = DecisionTreeClassifier()
boost = GradientBoostingClassifier()
hgb = HistGradientBoostingClassifier()
algorithms = {'lrg':lrg, 'sv':sv, 'mp':mp, 'rf':rf, 'dtc':dtc, 'boost':boost, 'hgb':hgb}

#Printing function for scores and metrics
def algo_results(results):
	mts = results.cv_results_['mean_test_score'] #avg score
	tol = results.cv_results_['std_test_score']  #toleranc
	optimal = results.best_params_
	print('Best parameters: {}'.format(optimal))
	for mean, std, params in zip(mts, tol, results.cv_results_['params']):
		print("%0.3f (+/-%0.3f) for %r" % (mean, std*2, params))

### Algorithm portion ###
def train_algo(algo, parameters):
	cv = GridSearchCV(estimator=algorithms[algo], param_grid=parameters, cv=5, scoring='f1', refit=True)
	cv.fit(f_train, l_train.values.ravel())
	results=open('results/{}output.txt'.format(algo), 'w')
	sys.stdout = results
	algo_results(cv) 
	pkl_file=open('pkl/{}pickle.pkl'.format(algo), 'wb')
	pkl.dump(cv.best_estimator_, pkl_file)
	results.close()
	pkl_file.close()

def plot_pr_curve(precision, recall, models):
	for name in models:
		plt.plot(precision[name], recall[name], label=name)
	plt.title("Precision Recall Curve")
	plt.xlabel("Recall")		
	plt.ylabel("Precision")
	plt.legend(loc="best")
	plt.show()

def train_models():
	#Fitting Logistic regression alogrithm
	c_list = [0.01, 0.1, 1, 10, 100] #C = .01 and C performed best (.872)-> High regularization 
	parameters = { 'C':c_list, 'class_weight':class_wght}
	train_algo('lrg', parameters)

	#Fitting support vector machine alogrithm
	c_list = [0.01, 0.1, 1, 10, 100] 
	parameters = { 'C':c_list, 'class_weight':class_wght}
	train_algo('sv', parameters)

	#Fitting Multilayer Perceptron
	hidden_sz = [5, 10, 20, 50, 100, 200]
	activ_func = ['logistic', 'tanh', 'relu']
	learning_rte = ['constant', 'invscaling', 'adaptive']
	parameters = {'activation':activ_func, 'learning_rate':learning_rte, 'hidden_layer_sizes':hidden_sz}
	train_algo('mp', parameters)

	#Random Forest
	n_estimators = [1, 2, 4, 6, 8, 10]
	max_depth = [2, 4, 6, 10, 16, 22, 28, 36]
	parameters={'n_estimators':n_estimators, 'max_depth':max_depth, 'class_weight':class_wght}
	train_algo('rf', parameters)

	#Decision Tree classifier
	n_estimators = [1, 2, 4, 6, 8, 10]
	max_depth = [6, 10, 16, 22, 28, 36]
	parameters={'max_depth':max_depth, 'class_weight':class_wght}
	train_algo('dtc', parameters)

	#Gradient Boosted Trees
	n_estimators = [1, 2, 4, 6, 8, 10]
	max_depth = [6, 10, 16, 22, 28, 36]
	learning_rate = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
	parameters={'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate}
	train_algo('boost', parameters)

	#Histogram Gradient Boosting classifier
	max_depth = [6, 10, 16, 22, 28, 36]
	learning_rate = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	parameters={'max_depth':max_depth, 'learning_rate':learning_rate}
	train_algo('hgb', parameters)

train_models()