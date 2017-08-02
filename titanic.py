# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math
import xgboost as xgb

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

list_model = [LogisticRegression, SVC, LinearSVC, RandomForestClassifier, KNeighborsClassifier, GaussianNB, Perceptron,
              SGDClassifier, DecisionTreeClassifier]

def one_lett_cab(value):
    if (isinstance(value,str)):
        return value[:1]
    else:
        return 'Z'

def age_groups(value):
    if math.isnan(value):
        return 2
    if value < 13:
        return 0
    if value >= 13 and value < 19:
        return 1
    if value >= 19 and value < 30:
        return 2
    if value >=30 and value < 45:
        return 3
    if value >= 45 and value < 60:
        return 4
    if value >= 60:
        return 5

def fare_groups(value):
    if math.isnan(value):
        return 1
    if value == 0:
        return 0
    if value > 0 and value < 50:
        return 1
    if value >= 50 and value < 100:
        return 2
    if value >= 100:
        return 3

def embarked_rename(value):
    if value == "C":
        return 0
    if value == "S":
        return 1
    if value == "Q":
        return 2
    if math.isnan(value):
        return 1

def cabin_rename(value):
    if value == "Z":
        return 2
    if value == "A":
        return 0
    if value == "B":
        return 1
    if value == "C":
        return 2
    if value == "D":
        return 3
    if value == "E":
        return 4
    if value == "F":
        return 5
    if value == "G":
        return 6
    if value == "T":
        return 7
    # if value is None:
    #     return 2
    # try:
    #     if math.isnan(value):
    #         return 2
    # except:
    #     print("lazha")

def sex_rename(value):
    if not str(value):
        return 1
    if value == "female":
        return 0
    if value == "male":
        return 1

def data_proc_sex(ser):
    ser = ser.apply(sex_rename)
    return ser

def data_proc_cabin(ser):
    tmp = ser.apply(one_lett_cab)
    tmp = tmp.apply(cabin_rename)
    return tmp

def data_proc_age(ser):
    ser = ser.apply(age_groups)
    return ser

def data_proc_fare(ser):
    ser = ser.apply(fare_groups)
    return ser

def data_proc_embarked(ser):
    ser = ser.apply(embarked_rename)
    return ser

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

ser = data_proc_age(df_train["Age"])
df_train = df_train.drop("Age", axis=1)
df_train.insert(2, "Age", ser)
ser = data_proc_fare(df_train["Fare"])
df_train = df_train.drop("Fare", axis=1)
df_train.insert(2, "Fare", ser)
ser = data_proc_cabin(df_train["Cabin"])
df_train = df_train.drop("Cabin", axis=1)
df_train.insert(2, "Cabin", ser)
ser = data_proc_embarked(df_train["Embarked"])
df_train = df_train.drop("Embarked", axis=1)
df_train.insert(2, "Embarked", ser)
ser = data_proc_sex(df_train["Sex"])
df_train = df_train.drop("Sex", axis=1)
df_train.insert(2, "Sex", ser)

ser = data_proc_age(df_test["Age"])
df_test = df_test.drop("Age", axis=1)
df_test.insert(2, "Age", ser)
ser = data_proc_fare(df_test["Fare"])
df_test = df_test.drop("Fare", axis=1)
df_test.insert(2, "Fare", ser)
ser = data_proc_cabin(df_test["Cabin"])
df_test = df_test.drop("Cabin", axis=1)
df_test.insert(2, "Cabin", ser)
ser = data_proc_embarked(df_test["Embarked"])
df_test = df_test.drop("Embarked", axis=1)
df_test.insert(2, "Embarked", ser)
ser = data_proc_sex(df_test["Sex"])
df_test = df_test.drop("Sex", axis=1)
df_test.insert(2, "Sex", ser)

df_train.to_csv("train_upd.csv", ";")
series_train = df_train["Survived"]
df_train = df_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]]
df_train.to_csv("train_upd.csv", ";")
df_test = df_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]]

# groups = df_train["Cabin"].groupby(df_train["Cabin"])
# for name, group in groups:
#      #print(type())
#      print(name)
# for i in list_model:
#     clf = i()
#     clf.fit(df_train, series_train)
#     print(clf.score(df_train, series_train))

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
model = xgb.XGBClassifier()
bst = model.fit(df_train, series_train)
# make prediction
series_test = bst.predict(df_test)


# clf = RandomForestClassifier()
# parameters = {'n_estimators': [4, 6, 9],
#               'max_features': ['log2', 'sqrt', 'auto'],
#               'criterion': ['entropy', 'gini'],
#               'max_depth': [2, 3, 5, 10],
#               'min_samples_split': [2, 3, 5],
#               'min_samples_leaf': [1, 5, 8]
#              }
# # Type of scoring used to compare parameter combinations
# acc_scorer = make_scorer(accuracy_score)
#
# # Run the grid search
# grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
# grid_obj = grid_obj.fit(df_train, series_train)
# clf = grid_obj.best_estimator_
# clf.fit(df_train, series_train)
# series_test = clf.predict(df_test)
df_test.insert(2, "Survived", series_test)
df_test.to_csv("test_upd.csv", ";")
#print(clf.score(df_train, series_train))

# print(df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Age', 'Survived']].groupby(['Age']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Fare', 'Survived']].groupby(['Fare']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Cabin', 'Survived']].groupby(['Cabin']).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False))