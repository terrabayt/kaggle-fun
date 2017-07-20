# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def one_lett_cab(value):
    if (isinstance(value,str)):
        return value[:1]
    else:
        return value

def age_groups(value):
    if math.isnan(value):
        return value
    if value < 13:
        return "kid"
    if value >= 13 and value < 19:
        return "teenager"
    if value >= 19 and value < 30:
        return "young adult"
    if value >=30 and value < 45:
        return "adult"
    if value >= 45 and value < 60:
        return "old man"
    if value >= 60:
        return "old old man"

def fare_groups(value):
    if math.isnan(value):
        return value
    if value == 0:
        return "no cost"
    if value > 0 and value < 50:
        return "low cost"
    if value >= 50 and value < 100:
        return "normal cost"
    if value >= 100:
        return "high cost"

def data_proc_cabin(ser):
    ser = ser.apply(one_lett_cab)
    return ser

def data_proc_age(ser):
    ser = ser.apply(age_groups)
    return ser

def data_proc_fare(ser):
    ser = ser.apply(fare_groups)
    return ser

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.assign(Cabin=data_proc_cabin(df_train["Cabin"]))
ser = data_proc_age(df_train["Age"])
df_train = df_train.drop("Age", axis=1)
df_train.insert(2, "Age", ser)
ser = data_proc_fare(df_train["Fare"])
df_train = df_train.drop("Fare", axis=1)
df_train.insert(2, "Fare", ser)


df_train = df_train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]]
df_train.to_csv("train_upd.csv", ";")

print(df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Age', 'Survived']].groupby(['Age']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Fare', 'Survived']].groupby(['Fare']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Cabin', 'Survived']].groupby(['Cabin']).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False))