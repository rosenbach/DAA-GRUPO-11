#%%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

param_grid_svm = [
  {'C': [1,3,5,7,9,11]}
]

param_grid_knn = [
  {'n_neighbors': [1,3,5,7,9,11]}
]
param_grid_dt = [
  {'max_depth': [1,3,5,7,9,11]}
]

param_grid_lr = [
        {'fit_intercept': [True, False],
         'normalize': [True, False],
         'copy_X': [True, False],
         'n_jobs': [-1],
         'max_iter': [100, 500, 1000, 1500],
         }
]


# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('test_data_clean_lookup.csv', encoding='ISO-8859-1')

#find all the columns in training_data that contain missing values
missing_cols = training_data.columns[training_data.isnull().any()].tolist()

# remove the identifier column from the training_data
training_data = training_data.drop('IDENTIFIER', axis = 1)


#%%print the data types of training_data columns
print(training_data.dtypes)

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
