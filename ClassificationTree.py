import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df=pd.read_csv("heart_cleveland_upload.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df['thal'].unique())
print(df['ca'].unique())

#Missing data 
print(len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')]))
#0 missing values so no need for preprocessing

#split data into test and training

X=df.drop('condition',axis=1).copy()
y=df['condition'].copy()
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.dtypes)

#one hot encoding
X_encoded=pd.get_dummies(X,columns=['cp','restecg','slope','thal'])
print(X_encoded.head())

y_not_zero_index=y>0
y[y_not_zero_index]=1
print(y.unique())

#building preliminary tree
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
print(tree.score(X_train, y_train))