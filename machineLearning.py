# -*- coding: utf-8 -*-
"""
@author: liutao
"""

# Load libraries
import pyfiglet
import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn import preprocessing
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import os     

banner = pyfiglet.figlet_format("ML Classifier")
bannerEnd = pyfiglet.figlet_format("GTI/LOG 795 HIVER 23")
print(banner)
print("")
fileName = input("Please enter the filename (ex: test.csv) : ")

# load dataset
dataset = pandas.read_csv(fileName)
print (dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('label').size())

# split dataset
array = dataset.values

print("")
print("----- Printing array : START. -----")
print("")
print(array)
print("")
print("----- Printing array : DONE. -----")
print("")

X = array[:,0:1]

formattedArray = X.flatten()

print("---- Label encoder -----")
le = preprocessing.LabelEncoder()
le.fit(formattedArray)
print(le.classes_)
X2 = le.transform(formattedArray)
X2 = X2.reshape(-1,1)
print(X2)
print("---------")

print("Printing 'X' : ")
print(X[0])
print("")

Y = array[:,1]
print("Printing 'Y' : ")
print(Y)
print("")

print("Declaring ML attributes.")
trainSize = 0.8
validation_size = 0.1
seed = 42
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2, Y,train_size=trainSize, test_size=validation_size, random_state=seed, shuffle = True)
X_train_set, X_test, Y_train_set, Y_test = model_selection.train_test_split(X_train, Y_train,train_size=trainSize, test_size=validation_size, random_state=seed, shuffle = True)

# Test options and evaluation metric
print("Declaring 'score' attribute.")
scoring = 'accuracy'

print("Creating models array.")
# valuating algorithm model
models = []
models.append(('LR', LogisticRegression()))
models.append(('LRCV', LogisticRegressionCV()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RNC', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('ETC', ExtraTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('MLP', MLPClassifier()))
#Ajouter d'autres modèles.

# evaluate each model in turn
print("Models evaluation : START.")
print("")
results = []
names = []
for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle = True)
        print("[ PRINT CV_RESULTS ]")
        print(kfold)
        print(X_train)
        print(Y_train)
        print("[ ---------------- ]")

        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s Accuracy: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
print("")
print("Models evaluation : DONE.")

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
print("RFC result 30% test set")
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train_set, Y_train_set)
filename = 'finalized_RFC_model.sav'
joblib.dump(rfc, filename)
# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print (result)

predictions_rfc = rfc.predict(X_test)
print("RFC accuracy test: \n")
print(accuracy_score(Y_test, predictions_rfc))
print(confusion_matrix(Y_test, predictions_rfc))
print(classification_report(Y_test, predictions_rfc))

# Make predictions on test dataset
print("RFC result final 30% validation")
newrfc = RandomForestClassifier(n_estimators=10)
newrfc.fit(X_train_set, Y_train_set)
newpredictions_rfc = newrfc.predict(X_validation)
print("----- RFC accuracy validation: -----\n")
print(accuracy_score(Y_validation, newpredictions_rfc))
print(confusion_matrix(Y_validation, newpredictions_rfc))
print(classification_report(Y_validation, newpredictions_rfc))

# Feature Importance
print ("Validating Feature importance")
# fit an Extra Trees model to the data
test_model = RandomForestClassifier()
test_model.fit(X_train, Y_train)
# display the relative importance of each attribute
print(test_model.feature_importances_)
# plot
plt.bar(range(len(test_model.feature_importances_)), test_model.feature_importances_)
plt.show()
#
df = dataset.reset_index(drop = False)
print("Finishing last steps ... OK.")
feat_importances = pandas.Series(test_model.feature_importances_, index=dataset.columns[0:1])
feat_importances.nlargest(20).plot(kind='barh')
print("Script is done.")
print(bannerEnd)
print("Helwan Mandé, Marc-Antoine Nadeau, Youssef Benmous.")