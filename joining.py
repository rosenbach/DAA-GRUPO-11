#%% import training_data_clean_fullDate_with_predictions_and_identifier.csv with pandas
import pandas as pd
training_data = pd.read_csv('training_data_clean_fullDate_with_predictions_and_identifier.csv')

#%% now we need to get the old data and create an identifier in order to be able to join the data
#import training_data_clean_fullDate.csv
fulldate_data = pd.read_csv('training_data_clean_fullDate.csv')


# %% now we create a IDENTIFIER feature in the fulldate_data dataframe, which is created by concatenating the YEAR, MONTH, DAY, HOUR, MINUTE and SECOND features, each as a string
fulldate_data['IDENTIFIER'] = fulldate_data['YEAR'].astype(str) + '-' + fulldate_data['MONTH'].astype(str) + '-' + fulldate_data['DAY'].astype(str) + '-' + fulldate_data['HOUR'].astype(str) + '-' + fulldate_data['MINUTE'].astype(str) + '-' + fulldate_data['SECOND'].astype(str)

#%% in the fulldata_data, drop everything but the identifier and the AVERAGE_RAIN feature
fulldate_data = fulldate_data[['IDENTIFIER', 'AVERAGE_RAIN']]

#now join the two dataframes by the identifier
training_data = training_data.merge(fulldate_data, on='IDENTIFIER', how='left')

#%% export the training_data to a csv file
training_data.to_csv('training_data_clean_identifier_cloudinesspredictions.csv', index=False)