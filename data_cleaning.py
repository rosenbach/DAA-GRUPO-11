import pandas as pd

#%%
# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data_clean_wRecordDate.csv', encoding='ISO-8859-1')

#%% print the length of the training_data
print(len(training_data))

#%% print the amount of rows that contain a value in the AVERAGE_RAIN feature
print(len(training_data[training_data['AVERAGE_RAIN'] > 0]))

#%% 

#in the record_date feature, convert the date to a datetime object
training_data['record_date'] = pd.to_datetime(training_data['record_date'])

training_data['SECOND'] = training_data['record_date'].dt.second
training_data['MINUTE'] = training_data['record_date'].dt.minute

#convert the record_date to a new feature called hour
training_data['HOUR'] = training_data['record_date'].dt.hour

#convert the record_date to a new feature called day
training_data['DAY'] = training_data['record_date'].dt.day

#convert the record_date to a new feature called month
training_data['MONTH'] = training_data['record_date'].dt.month

#convert the record_date to a new feature called year
training_data['YEAR'] = training_data['record_date'].dt.year

# drop the record_date column
training_data = training_data.drop('record_date', axis = 1)
# drop the city_name column
training_data = training_data.drop('city_name', axis = 1)

# drop the AVERAGE_PRECIPITATION feature from the training_data
training_data = training_data.drop('AVERAGE_PRECIPITATION', axis = 1)


# %% create a new IDENTIFIER feature, which is a string combined of the YEAR, MONTH, DAY, HOUR, MINUTE and SECOND features, each as a string combined
# using the str() function
training_data['IDENTIFIER'] = training_data['YEAR'].astype(str) + '-' + training_data['MONTH'].astype(str) + '-' + training_data['DAY'].astype(str) + '-' + training_data['HOUR'].astype(str) + '-' + training_data['MINUTE'].astype(str) + '-' + training_data['SECOND'].astype(str)

#%% export the training_data to a csv file called training_data_cleaned_1.csv
training_data.to_csv('training_data_clean_IDENTIFIER.csv', index = False)
# %%
