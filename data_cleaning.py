
#%%
import pandas as pd

# load the data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable data
#data = pd.read_csv('training_data.csv', encoding='ISO-8859-1')
data = pd.read_csv('test_data.csv', encoding='ISO-8859-1')

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

#create a new IDENTIFIER feature, which is a string combined of the YEAR, MONTH, DAY, HOUR, MINUTE and SECOND features, each as a string combined
# using the str() function
data['IDENTIFIER'] = data['YEAR'].astype(str) + '-' + data['MONTH'].astype(str) + '-' + data['DAY'].astype(str) + '-' + data['HOUR'].astype(str)


#%% drop unwanted columns
#  drop the record_date column
data = data.drop('record_date', axis = 1)
# drop the city_name column
data = data.drop('city_name', axis = 1)
# drop the AVERAGE_PRECIPITATION feature from the data
data = data.drop('AVERAGE_PRECIPITATION', axis = 1)

#%% lookup table for AVERAGE_CLOUDINESS
data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].astype('string') 
data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*claro.*$)', 'ceu claro')
data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*nublado.*$)', 'ceu pouco nublado')
data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*limpo.*$)', 'ceu limpo')
data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*quebrados.*$)', 'nuvens quebradas')

#%% lookup table for AVERAGE_RAIN
#print the unique values of the AVERAGE_RAIN feature
print(data['AVERAGE_RAIN'].unique())
AVERAGE_RAIN_to_IS_STORMY_dict = {
    'chuvisco fraco':0, #leichter Nieselregen
    'chuvisco e chuva fraca':0, #Nieselregen und leichter Regen
    'chuva fraca': 0, #leichter Regen
    'chuva leve': 0, #leichter Regen
    'chuva moderada': 0, #mäßiger Regen
    'chuva': 0, #Regen
    'trovoada com chuva leve': 1, #gewitter mit leichtem Regen
    'trovoada com chuva': 1, #gewitter mit Regen
    'aguaceiros fracos':0, #leichte Schauer
    'aguaceiros': 0, #Schauer
    'chuva de intensidade pesada':0, #starker Regen
    'chuva de intensidade pesado':0, #starker Regen
    'chuva forte':0, #starker Regen
    }

#create a new IS_STORMY feature from the AVERAGE_RAIN feature, using the AVERAGE_RAIN_to_IS_STORMY_dict lookup table
data['IS_STORMY'] = data['AVERAGE_RAIN'].map(AVERAGE_RAIN_to_IS_STORMY_dict)

# create a dictionary to map the AVERAGE_RAIN
# values to the new values
AVERAGE_RAIN_to_AVERAGE_RAIN_dict = {
    'chuvisco fraco':0.1, #leichter Nieselregen
    'chuvisco e chuva fraca':3.5, #Nieselregen und leichter Regen
    'chuva fraca': 2.5, #leichter Regen
    'chuva leve': 2.5, #leichter Regen
    'chuva moderada': 7, #mäßiger Regen
    'chuva': 9, #Regen
    'trovoada com chuva leve': 2.5, #gewitter mit leichtem Regen
    'trovoada com chuva': 9, #gewitter mit Regen
    'aguaceiros fracos':3, #leichte Schauer
    'aguaceiros': 6.6, #Schauer
    'chuva de intensidade pesada':50, #starker Regen
    'chuva de intensidade pesado':50, #starker Regen
    'chuva forte':30, #starker Regen
    }

#map the AVERAGE_RAIN feature to the new AVERAGE_RAIN feature, using the AVERAGE_RAIN_to_AVERAGE_RAIN_dict lookup table
data['AVERAGE_RAIN'] = data['AVERAGE_RAIN'].map(AVERAGE_RAIN_to_AVERAGE_RAIN_dict)

#%%map the values of the AVERAGE_SPEED_DIFF from 0 to 4
#data['AVERAGE_SPEED_DIFF'] = data['AVERAGE_SPEED_DIFF'].map({"None": 0, "Low": 1, "Medium": 2, "High": 3, "Very_High": 4})

#map the values of LUMINOSITY from 1 to 3
data['LUMINOSITY'] = data['LUMINOSITY'].map({"DARK": 0, "LOW_LIGHT": 1, "LIGHT": 2})

cloud_dict = {'ceu claro' : 1, 'ceu limpo': 2, 'nuvens dispersas': 3, 'nuvens quebradas': 4, 'algumas nuvens': 5, 'ceu pouco nublado': 6}
data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].map(cloud_dict)

#%% export the data to a csv file called data_cleaned_1.csv
data.to_csv('test_data_clean_lookup.csv', index = False)

#%% print the unique elements of AVERAGE_RAIN
data.info()
print(data['AVERAGE_RAIN'].unique())
# %%
