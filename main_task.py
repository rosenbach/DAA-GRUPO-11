#%% import pandas
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

training_data = pd.read_csv('./parallel_data/training_data_FULL_w.csv')

#train a 
# %% print the colnames of training_data
print(training_data.columns)
# %% drop the AVERAGE_SPEED_DIFF_y column
training_data = training_data.drop(['AVERAGE_SPEED_DIFF_y'], axis=1)
#rename the AVERAGE_SPEED_DIFF_x column to AVERAGE_SPEED_DIFF
training_data = training_data.rename(columns={'AVERAGE_SPEED_DIFF_x': 'AVERAGE_SPEED_DIFF'})
#print the number of rows
print(training_data.shape[0])

#%%drop the 'Unnamed: 0' column
training_data = training_data.drop(['Unnamed: 0'], axis=1)

# %% 

param_grid_svm = [
  {'C': [1,3,5,7,9,11]}
]

param_grid_knn = [
  {'n_neighbors': [1,3,5,7,9,11]}
]
param_grid_dt = [
  {'max_depth': [1,3,5,7,9,11]}
]



#%%
def create_models(data, feature):
    """
        This function takes in a dataframe and a feature and returns a model that can be used to predict the feature.
    """

    #%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
    #Define the X and y
    X = data.drop(feature, axis = 1)
    y = data[feature]

    #KNN
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    knn_cv.fit(X, y)
    print(knn_cv.best_params_)
    print(knn_cv.best_score_)

    #Decision tree
    tree_cv = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
    tree_cv.fit(X, y)
    print(tree_cv.best_params_)
    print(tree_cv.best_score_)

    #SVM with gridsearchcv
    #svm_cv = GridSearchCV(SVC(), param_grid_svm, cv=5)
    #svm_cv.fit(X, y)
    #print(svm_cv.best_params_)
    #print(svm_cv.best_score_)

    #create an array with all the models
    models = [knn_cv, tree_cv]

    #return models
    return models

#%% get the name of the first column
col_name = training_data.columns[0]

#remove col_name from training_data
training_data = training_data.drop(col_name, axis = 1)


training_data.insert(0, 'IS_STORMY1', training_data['IS_STORMY'])

#%%swap the first and the second column
training_data = training_data[training_data.columns[::-1]]

#%%delete the IS_STORMY feature
training_data = training_data.drop(['IS_STORMY'], axis = 1)

#rename the IS_STORMY1 feature to IS_STORMY
training_data = training_data.rename(columns={'IS_STORMY1': 'IS_STORMY'})



#%% export the training_data to FINAL
training_data.to_csv("training_data_FINAL.csv", encoding='ISO-8859-1')


#%% create a decision tree with a gridsearchcv to find the best parameters for the model
models = create_models(training_data, "AVERAGE_SPEED_DIFF")

#%% get the best performing model
best_model = max(models, key=lambda x: x.best_score_)

#%% now load the test_data 
test_data = pd.read_csv('./parallel_data/test_data_FULL.csv')
#sort by the 'Unnamed: 0' column
test_data = test_data.sort_values(by=['Unnamed: 0'])
#remove the IDENTIFIER Column
test_data = test_data.drop(['IDENTIFIER'], axis=1)

predictions = best_model.predict(test_data.drop(['Unnamed: 0'], axis=1))

#create a new "AVERAGE_SPEED_DIFF" column in the test_data 
test_data['AVERAGE_SPEED_DIFF'] = predictions

#rename the "Unnamed: 0" column to RowId
test_data = test_data.rename(columns={'Unnamed: 0': 'RowId'})

#add 1 to each element of the RowId
test_data['RowId'] = test_data['RowId'] + 1

#only keep RowId and AVERAGE_SPEED_DIFF
test_data = test_data[['RowId', 'AVERAGE_SPEED_DIFF']]

#map AVERAGE_SPEED_DIFF to "None", "Low", "Medium", "High", "Very_High"
test_data['AVERAGE_SPEED_DIFF'] = test_data['AVERAGE_SPEED_DIFF'].map({0: "None", 1: "Low", 2: "Medium", 3: "High", 4: "Very_High"})

#rename AVERAGE_SPEED_DIFF to Speed_Diff
test_data = test_data.rename(columns={'AVERAGE_SPEED_DIFF': 'Speed_Diff'})

#%%export the test_data to a csv file
test_data.to_csv('./parallel_data/results_full.csv', index=False)

#%%print info about the test_data
print(test_data.info())

#


#%%
def estimate_feature(data, feature):
    """
    This function takes in a dataframe and a feature and fills up the missing values with the estimated values
    """
    print("hello, we are looking at the feature: " + feature)

    #create a copy of missing_cols
    missing_cols_copy = missing_cols.copy()
    missing_cols_copy.remove(feature)

    #from the data, drop missing_cols
    data_without_cols = data.drop(missing_cols_copy, axis = 1)

    # take all the rows that contain missing values in the feature and put them in a new dataframe
    missing_data_without_cols = data_without_cols[data_without_cols[feature].isnull()]

    #take all the rows that do not contain missing values in the feature and put them into the data
    data_without_cols = data_without_cols[data_without_cols[feature].notnull()]

    #multiply each element of data_without_cols by 100, if feature equals AVERAGE_RAIN
    if feature == "AVERAGE_RAIN":
        data_without_cols[feature] = data_without_cols[feature] * 100


    #if models is empty, create the models
    models = create_models(data_without_cols, feature)

    #find the best model in the models array
    best_model = max(models, key=lambda x: x.best_score_)

    #use best_model to predict the feature in missing_data and put the result in a new dataframe
    missing_data_without_cols[feature] = best_model.predict(missing_data_without_cols.drop(feature, axis = 1))

    #concatenate the two dataframes
    data_with_estimated_values = pd.concat([data_without_cols, missing_data_without_cols])

    #create a string for the filename by concatenating "training_data_", the feature and ".csv"
    filename = "test_data_" + feature + ".csv"

    #export the data to a csv file
    data_with_estimated_values.to_csv(filename, encoding='ISO-8859-1')

#%% print the unique values of the AVERAGE_RAIN column
print(training_data['AVERAGE_RAIN'].unique())

#%% print missing_cols[1]
print(missing_cols[1])
#%%
estimate_feature(training_data,missing_cols[1])

#%%for each element in missing_cols, call the estimate_feature function
for feature in missing_cols:
    estimate_feature(training_data, feature)

# %%

