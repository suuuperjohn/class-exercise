import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


br_df = pd.read_csv('Totals.csv')
br_df = br_df.dropna(axis=1)
rev_df = pd.read_csv('9_year_revenue_ver2_by_year_and_office.csv')
rev_df.head()
br_df.head()
br_df
location_to_state = {'Baton Rouge': 'LA',
                     'Dallas/Fort Worth': 'TX',
                     'Gulfport': 'MS',
                     'Houston': 'TX',
                     'Jackson': 'MS',
                     'New Orleans' : 'LA',
                     'Tupelo': 'MS',
                     'Raleigh' : 'NC',
                     'Tampa' : 'FL',
                     'Birmingham' : 'AL',
                     'Mobile' : 'AL'}
rev_df['State'] = rev_df['office_location'].map(location_to_state)
df = pd.merge(br_df, rev_df, on=['year', 'State'], how='inner')
df.head()
df = df.drop(['State','year','office_location'], axis = 1)
plt.scatter(df['revenue'],df['Total_Bankruptcies'])
plt.show()
# we can now start the modeling 
# partitioning the data
X = df.drop(columns= ['revenue'])
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)
# testing new data
new_bankruptcies = np.array([[100], [200], [300]])  # example input data
predicted_revenue = lr.predict(new_bankruptcies)
print(predicted_revenue)
# Calculate R-squared, MSE, and MAE
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)

# Print the evaluation metrics
print(f"R-squared: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")