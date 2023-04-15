import pandas as pd

df = pd.read_csv('titanic_raw.csv')

df.head()

df.describe()

df.dtypes

df.nunique

df.corr(df)['survived']


df.corr()['survivied'].sort_values(ascending=False)
df = df.loc[:,['pclass', 'survived','age', 'sex','embarked']]
df.isna().sum()
df.age.fillna(df.age.median(), inplace=True)
df = df.loc[df.embarked.notna(), :]



calculate_bonus = lambda sales:sales*0.02 if sales > 50000 else sales*0.01
print(calculate_bonus(sales=600))
print(calculate_bonus(600))

#Part 1


import pandas as pd


df = pd.read_csv('titanic_raw.csv')

print('Dataset Shape:', df.shape)

print('\nFirst 5 Rows:')
print(df.head())

print('\nData Types:')
print(df.dtypes)

print('\nMissing Values:')
print(df.isna().sum())

print('\nCategorical Variables:')
for col in ['survived', 'pclass', 'sex', 'embarked']:
    print('\n', col)
    print(df[col].value_counts())

print('\nNumerical Variables:')
print(df.describe())


print('\nCorrelation Matrix:')
corr = df.corr()
print(corr)


#Part 2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("titanic_raw.csv")

X = df.drop("survived", axis='columns')
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

