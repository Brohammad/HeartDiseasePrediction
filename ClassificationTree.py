import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# Load dataset
df = pd.read_csv("heart_cleveland_upload.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df['thal'].unique())
print(df['ca'].unique())

# Missing data 
print(len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')]))
# 0 missing values so no need for preprocessing

# Split data into features (X) and labels (y)
X = df.drop('condition', axis=1).copy()
y = df['condition'].copy()
print(y.head())
print(X.dtypes)

# One hot encoding
X_encoded = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'])
print(X_encoded.head())

# Convert 'condition' values > 0 to 1
y = y.copy()  # avoid SettingWithCopyWarning
y[y > 0] = 1
print(y.unique())

# Split data again after encoding
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Print accuracy
print(tree.score(X_test, y_test))   # test accuracy
print(tree.score(X_train, y_train)) # train accuracy

# Plot the decision tree
plt.figure(figsize=(25, 20)) 
plot_tree(tree, filled=True, rounded=True, class_names=["No HD", "Has HD"], feature_names=X_encoded.columns)
plt.show()

#confusion matrix

ConfusionMatrixDisplay.from_estimator(tree, X_test, y_test, display_labels=["No HD", "Has HD"], cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#pruning the tree
path=tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas= path.ccp_alphas
ccp_alphas=ccp_alphas[:-1]
clf_tree=[]
for ccp_alpha in ccp_alphas:
    clf=DecisionTreeClassifier(ccp_alpha=ccp_alpha,random_state=0)
    clf.fit(X_train,y_train)
    clf_tree.append(clf)

#now we try to find the accuracy for training and testing scores
train_scores=[clf.score(X_train,y_train) for clf in clf_tree]
test_scores=[clf.score(X_test,y_test) for clf in clf_tree]
fig,ax=plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores,label="train")
ax.plot(ccp_alphas,test_scores,label="test")
ax.legend()
plt.show()
