# -*- coding: utf-8 -*-
"""Decision Tree for Phishing Attack

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jnrC-u-9LEubHd6TV3pFTk5qUtgx-dy1

## Importing libraries
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt

"""## Predict Phishing Web Page Using Decision Tree

Phishing is a method of trying to gather personal information using deceptive e-mails and websites.

In this notebook, we will read the data and look at what are the features that can give us information on what are the attributes of a phishing website, the apply decision tree model to predict the result

## Loading dataset
"""

ml = pd.read_csv('https://raw.githubusercontent.com/khuynh22/Phishing-Detection/main/Phishing_Legitimate_full.csv')

"""##Viewing the data"""

ml.info()

ml.head()

"""## Taking unique values

Computes the number of unique elements in each column of the DataFrame 'ml', sorts the resulting Series in ascending order of unique values, resets the index, and creates a new DataFrame 'unique' that contains the sorted and indexed Series of unique values.
"""

unique = ml.nunique().sort_values(ascending=True).reset_index()
unique

"""##Taking data for modeling and prediction

Separate data into data for building model, and data for prediction
"""

data = ml.sample(frac = 0.8, random_state=42)
data_unseen = ml.drop(data.index)
data.reset_index(inplace = True, drop = True)

data_unseen.reset_index(inplace = True, drop = True)
print('Data for Modeling: ' + str(data.shape))

#similar with above for unseen data for predictions
print('Data for Prediction: ' + str(data_unseen.shape))

"""Put data into training and testing data"""

X_train = data.iloc[:,1:-1]
y_train = data.iloc[:,-1]
X_test = data_unseen.iloc[:, 1:-1]
y_test = data_unseen.iloc[:,-1]

"""## Classification Tree

Building classification tree using Gini index
"""

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

"""Plot the decision tree model"""

plt.figure(figsize = (50, 50))
features = X_train.columns
classes = ['Not Phishing', 'Phishing']
tree.plot_tree(clf, feature_names = features, class_names = classes, filled = True)
plt.show()

y_pred = clf.predict(X_test)

classes = ['Not Phishing', 'Phishing']
def plot_confusionMatrix(y_train_pred, y_train, dom):
  print(f'{dom} Confusion matrix')
  cf = confusion_matrix(y_train_pred, y_train)
  sns.heatmap(cf, annot = True, yticklabels= classes, xticklabels= classes, cmap= 'Blues', fmt= 'g')
  plt.tight_layout()
  plt.show()

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
print(f'Train score: {accuracy_score(y_train_pred, y_train)}')
print(f'Test score: {accuracy_score(y_test_pred, y_test)}')
plot_confusionMatrix(y_train_pred, y_train, 'Train')
plot_confusionMatrix(y_test_pred, y_test, 'Test')

"""#Prune Tree
Removing impurities data from the dataset to find valuable branches for tree pruning. Tree pruning reduces the model's size, enhances its functionality, and prevents overfitting.
"""

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
  clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf.fit(X_train, y_train)
  clfs.append(clf)

"""Cutting branches process"""

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize = (10, 10))
plt.scatter(ccp_alphas, node_counts)
plt.scatter(ccp_alphas, depth)
plt.plot(ccp_alphas, node_counts, label = 'No of nodes', drawstyle = 'steps-post')
plt.plot(ccp_alphas, depth, label = 'Depth', drawstyle = 'steps-post')
plt.legend()
plt.show()

clf_ = tree.DecisionTreeClassifier(random_state = 0, ccp_alpha = 0.01)
clf_.fit(X_train , y_train)
plt.figure(figsize = (20,20))
features = X_train.columns
tree.plot_tree(clf_, feature_names = features, class_names = classes, filled = True)
plt.show()

"""Graph between accuracy and alpha"""

train_acc = []
test_acc = []
for c in clfs:
  y_train_pred = c.predict(X_train)
  y_test_pred = c.predict(X_test)
  train_acc.append(accuracy_score(y_train, y_train_pred))
  test_acc.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize = (10, 10))
plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas, train_acc, label = 'Train Accuracy', drawstyle = 'steps-post')
plt.plot(ccp_alphas, test_acc, label = 'Test Accuracy', drawstyle = 'steps-post')
plt.legend()
plt.title('Accuracy vs Alpha')
plt.show()

"""# Confusion Matrix"""

#choose alpha = 0.010
clf_ = tree.DecisionTreeClassifier(random_state=0, ccp_alpha = 0.010)
clf_.fit(X_train, y_train)

y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)
print(f'Train score{accuracy_score(y_train_pred, y_train)}')
print(f'Test score{accuracy_score(y_test_pred, y_test)}')
plot_confusionMatrix(y_train_pred, y_train, 'Train')
plot_confusionMatrix(y_test_pred, y_test, 'Test')

sns.lineplot(x = 'num_of_features', y = 'precision', data = df, label = 'Precision Score')
sns.lineplot(x = 'num_of_features', y = 'recall', data = df, label = 'Recall Score')
sns.lineplot(x = 'num_of_features', y = 'f1_score', data = df, label = 'F1 Score')
sns.lineplot(x = 'num_of_features', y = 'accuracy', data = df, label = 'Accuracy Score')

"""# Classification Model Training Report"""

from sklearn.metrics import classification_report
print(f'Train classification report \n {classification_report(y_train, y_train_pred, target_names = classes)}')
print(f'\n Test classification report \n {classification_report(y_test_pred, y_test, target_names = classes)}')