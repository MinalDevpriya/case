# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data
df = pd.read_csv('laptop_details.csv')

# EDA
# Checking the shape of the dataset
print(df.shape)

# Checking for missing values
print(df.isnull().sum())

# Checking for duplicate values
print(df.duplicated().sum())

# Dropping duplicate values
df.drop_duplicates(inplace=True)

# Checking the shape of the dataset after dropping duplicates
print(df.shape)

# Checking the data types
print(df.dtypes)

# Descriptive statistics
print(df.describe())

# Distribution plots for numerical features
sns.displot(df['Rating'])
plt.show()

sns.displot(df['MRP'])
plt.show()

# Count plot for categorical features
sns.countplot(data=df, x='Brand')
plt.show()

sns.countplot(data=df, x='OS')
plt.show()

# Feature Engineering
# Extracting the processor generation from the Processor column
df['Processor_Generation'] = df['Processor'].str.extract('(\d+)')
df['Processor_Generation'] = pd.to_numeric(df['Processor_Generation'])
df.drop('Processor', axis=1, inplace=True)

# Extracting the RAM size and type from the RAM column
df['RAM_Size'] = df['RAM'].str.extract('(\d+)').astype(int)
df['RAM_Type'] = df['RAM'].str.extract('([A-Z]+)').astype(str)
df.drop('RAM', axis=1, inplace=True)

# Extracting the storage type and size from the Storage column
df['Storage_Type'] = df['Storage'].str.extract('([A-Za-z]+)').astype(str)
df['Storage_Size'] = df['Storage'].str.extract('(\d+)').astype(int)
df.drop('Storage', axis=1, inplace=True)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Brand', 'OS', 'RAM_Type', 'Storage_Type'], drop_first=True)

# Splitting the data into train and test sets
from sklearn.model_selection import train_test_split

X = df.drop(['MRP', 'Product', 'Feature'], axis=1)
y = df['MRP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train R^2 score: {:.3f}".format(r2_score(y_train, y_train_pred)))
print("Test R^2 score: {:.3f}".format(r2_score(y_test, y_test_pred)))
print("Train RMSE score: {:.3f}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print("Test RMSE score: {:.3f}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))

# Building the interface
import streamlit as st

st.set_page_config(page_title="Tesla Laptop Price Predictor", page_icon=":computer:", layout="wide")

st.title("Tesla Laptop Price Predictor")

# RAM Size
ram_size = st.number_input("RAM Size (in GB)", min_value=1, max_value=32, value=8, step=1)

