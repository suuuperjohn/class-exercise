import pandas as pd

df = pd.read_csv('titanic_raw.csv')

df.head()

df.describe()

ddf.dtypes

df.nunique

df.corr()['survived']


df.corr()['survivied'].sort_values(ascending=False)
df = df.loc[:,['pclass', 'survived','age', 'sex','embarked']]
df.isna().sum()
df.age.fillna(df.age.median(), inplace=True)
df = df.loc[df.embarked.notna(), :]

