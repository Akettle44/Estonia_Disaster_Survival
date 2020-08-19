import pickle as pkl
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from algo_training import plot_pr_curve 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, average_precision_score
from collections import Counter

#Choosing the best algorithm
f_val = pd.read_csv('data/f_val.csv')
l_val= pd.read_csv('data/l_val.csv', header=None)
f_test = pd.read_csv('data/f_test.csv')
l_test = pd.read_csv('data/l_test.csv', header=None)

#read in all of the pickles
algo_names = ['lrg', 'sv', 'mp', 'rf', 'dtc', 'boost', 'hgb']
complex_names = ['mp', 'rf', 'dtc', 'boost', 'hgb']
simple_names = ['lrg', 'sv']

algo_pkl = {}
for name in algo_names: #get pkl objects
	pklfile = open('pkl/{}pickle.pkl'.format(name), 'rb') #opens pickle and stores it in an array  
	algo_pkl[name] = pkl.load(pklfile)
	pklfile.close()

def complex_algo_analysis():
    prec = {}
    reca = {}
    for name in complex_names:
        prob = algo_pkl[name].predict_proba(f_val)[:,1]
        precision, recall, _ = precision_recall_curve(l_val, prob)
        prec[name] = precision
        reca[name] = recall
    return prec, reca 

def simple_algo_analysis():
    prec = {}
    reca = {}
    for name in simple_names:
        prob = algo_pkl[name].decision_function(f_val)
        precision, recall, _ = precision_recall_curve(l_val, prob)
        prec[name] = precision
        reca[name] = recall
    return prec, reca 

#Calculate precision and recall then plot graph
prec_simple, recall_simple = simple_algo_analysis()
prec_complex, recall_complex = complex_algo_analysis()
prec_simple.update(prec_complex)
recall_simple.update(recall_complex)
plot_pr_curve(prec_simple, recall_simple, algo_names)

#final test
for name in algo_names:
    print("Score for {}".format(name)) #assigning test score
    prediction = algo_pkl[name].predict(f_test)
    print(accuracy_score(l_test, prediction))
    print(precision_score(l_test, prediction, average='binary'))
    print(recall_score(l_test, prediction))
    print(f1_score(l_test, prediction))