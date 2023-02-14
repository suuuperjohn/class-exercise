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



calculate_bonus = lambda sales:sales*0.02 if sales > 50000 else sales*0.01
print(calculate_bonus(sales=600))
print(calculate_bonus(600))
