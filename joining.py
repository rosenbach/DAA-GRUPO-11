#%% import all the csv files in the folder parallel_data

import pandas as pd
d1 = pd.read_csv('./parallel_data/test_data_AVERAGE_CLOUDINESS.csv')
d2 = pd.read_csv('./parallel_data/test_data_AVERAGE_RAIN.csv')
d3 = pd.read_csv('./parallel_data/test_data_IS_STORMY.csv')




# %% now we create a IDENTIFIER feature in the fulldate_data dataframe, which is created by concatenating the YEAR, MONTH, DAY, HOUR, MINUTE and SECOND features, each as a string
def addIdentifier(df):
    df['IDENTIFIER'] = df['YEAR'].astype(str) + '-' +     df['MONTH'].astype(str) + '-' + df['DAY'].astype(str)     + '-' + df['HOUR'].astype(str)
    return df

#%%
d1 = addIdentifier(d1)
d2 = addIdentifier(d2)
d3 = addIdentifier(d3)

#%% print colnames of d1
print(d1.columns)

#%%
#in d1, drop AVERAGE_RAIN and IS_STORMY
#d1 = d1.drop(['AVERAGE_RAIN', 'IS_STORMY'], axis=1)

#in d2, drop everything but AVERAGE_RAIN and IDENTIFIER
d2 = d2[['IDENTIFIER', 'AVERAGE_RAIN']]

#in d3, drop everything but IS_STORMY and IDENTIFIER
d3 = d3[['IDENTIFIER', 'IS_STORMY']]


#%% join the three dataframes by the IDENTIFIER feature

df = pd.merge(d1, d2, on='IDENTIFIER')
df = pd.merge(df, d3, on='IDENTIFIER')

#%% export df to csv
df.to_csv('./parallel_data/test_data_FULL.csv', index=False)


#%% load the full dataframe
df = pd.read_csv('./parallel_data/training_data_FULL.csv')

#load the old training_data
data = pd.read_csv('training_data.csv' ,encoding='ISO-8859-1')

#map the AVERAGE_SPEED_DIFF to 1-4
data['AVERAGE_SPEED_DIFF'] = data['AVERAGE_SPEED_DIFF'].map({"None":0,"Low":1, "Medium":2, "High":3, "Very_High":4})


#%% print unique elements of AVERAGE_SPEED_DIFF
print(data['AVERAGE_SPEED_DIFF'].unique())

#%% record date to unique columns

#in the record_date feature, convert the date to a datetime object
data['record_date'] = pd.to_datetime(data['record_date'])

#convert the record_date to a new feature called hour
data['HOUR'] = data['record_date'].dt.hour

#convert the record_date to a new feature called day
data['DAY'] = data['record_date'].dt.day

#convert the record_date to a new feature called month
data['MONTH'] = data['record_date'].dt.month

#convert the record_date to a new feature called year
data['YEAR'] = data['record_date'].dt.year

data['IDENTIFIER'] = data['YEAR'].astype(str) + '-' + data['MONTH'].astype(str) + '-' + data['DAY'].astype(str) + '-' + data['HOUR'].astype(str)

#%%from the data, only keep IDENTIFIER and AVERAGE_SPEED_DIFF
data = data[['IDENTIFIER', 'AVERAGE_SPEED_DIFF']]

#%% print info of data
print(data.info())

#%%print info of df
print(df.info())

# %% join df and data by IDENTIFIER
df = pd.merge(df, data, on='IDENTIFIER')

#drop IDENTIFIER from df
df = df.drop('IDENTIFIER', axis=1)


#export as csv
df.to_csv('./parallel_data/training_data_FULL_w.csv', index=False)



# %%
