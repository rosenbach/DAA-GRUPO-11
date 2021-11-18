#%% KNN
import pandas as pd
# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data.csv', encoding='utf-8')

#%% apply methods for data exploration on training_data
# print the first 5 rows of the training_data
print(training_data.head())
print(training_data.describe())
print(training_data.info())