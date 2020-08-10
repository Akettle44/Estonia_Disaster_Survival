import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Exploring Data
'''
df['Age'].mean() #Average age of 44
df['Survived'] #13% survival rate, yoinks!
df.describe()
a = sns.catplot(x='Sex', y='Survived', kind="point", data=df)
plt.show()
df.groupby('Category').mean()
'''

#Cleaning data
df = pd.read_csv("data/estonia-passenger-list.csv") #read in csv
df.drop(['Firstname', 'Lastname', 'PassengerId', 'Country'], axis=1, inplace=True) #Remove identifying data and country because there didn't seem to be any statistical relevance
df['Sex'].replace(['M', 'F'], [0,1], inplace=True)      #enumerate sex column
df['Category'].replace(['P', 'C'], [0,1], inplace=True) #enumerate passenger column P=passengers (0), C=Crew (1)

#Splitting Data into training, validation, and testing
features = df.drop('Survived', axis=1) 
labels = df['Survived'] #What we are looking to predict
f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=.4, random_state=42) #training on 60% of the dataset
f_val, f_test, l_val, l_test = train_test_split(features, labels, test_size=.5, random_state=42)   #Validation and testing take up 20% each

#Saving split data to a file so comparison can be recreated
f_train.to_csv('data/f_training.csv', index=False)
l_train.to_csv('data/l_training.csv', index=False, header=False)
f_val.to_csv('data/f_val.csv', index=False)
l_val.to_csv('data/l_val.csv', index=False, header=False)
f_test.to_csv('data/f_test.csv', index=False)
l_test.to_csv('data/l_test.csv', index=False, header=False)

