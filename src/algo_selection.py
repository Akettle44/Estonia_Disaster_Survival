import pickle as pkl
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter

#Choosing the best algorithm
f_val = pd.read_csv('data/f_val.csv')
l_val= pd.read_csv('data/l_val.csv', header=None)
f_test = pd.read_csv('data/f_test.csv')
l_test = pd.read_csv('data/l_test.csv', header=None)

#read in all of the pickles
algo_names = ['lrg', 'sv', 'mp', 'rf', 'dtc', 'boost', 'hgb']
algo_pkl = {}
for name in algo_names: #get pkl objects
	pklfile = open('pkl/{}pickle.pkl'.format(name), 'rb') #opens pickle and stores it in an array  
	algo_pkl[name] = pkl.load(pklfile)
	pklfile.close()

#Testing best scores on validation set
for name in algo_names:
    print("Score for {}".format(name)) #assigning test score
    prediction = algo_pkl[name].predict(f_val)
    print(accuracy_score(l_val, prediction))
    print(precision_score(l_val, prediction, average='binary'))
    print(recall_score(l_val, prediction))