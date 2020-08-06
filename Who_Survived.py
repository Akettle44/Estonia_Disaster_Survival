import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

#Exploring Data
'''
df['Age'].mean() #Average age of 44
df['Survived'].mean() #13% survival rate, yoinks!
df.describe()
a = sns.catplot(x='Sex', y='Survived', kind="point", data=df)
plt.show()
'''

#Cleaning data
df = pd.read_csv('estonia-passenger-list.csv') #read in csv
df.drop(['Firstname', 'Lastname', 'PassengerId', 'Country'], axis=1, inplace=True) #Remove identifying data and country because there didn't seem to be any statistical relevance
df['Sex'].replace(['M', 'F'], [0,1], inplace=True)      #enumerate sex column
df['Category'].replace(['P', 'C'], [0,1], inplace=True) #enumerate passenger column P=passengers (0), C=Crew (1)

#Splitting Data
features = df.drop('Survived', axis=1) 
labels = df['Survived'] #What we are looking to predict
train, test = train_test_split(df, test_size=.6, random_state=42) #training on 60% of the dataset
val, test = train_test_split(df, test_size=.5, random_state=42)   #Validation and testing take up 20% each

#Using algortihms
