import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

df = pd.read_csv('estonia-passenger-list.csv') #read in csv
df.drop(['Firstname', 'Lastname', 'Country'], axis=1, inplace=True) #Remove categorical data
df['Sex'].replace(['M', 'F'], [0,1], inplace=True) #making sex column numerical
df['Category'].replace(['P', 'C'], [0,1], inplace=True) #P=passengers (0), C=Crew (1)
