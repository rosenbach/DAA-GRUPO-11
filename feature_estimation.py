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


#%% load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data_clean_lookup.csv', encoding='ISO-8859-1')

#find all the columns in training_data that contain missing values
missing_cols = training_data.columns[training_data.isnull().any()].tolist()

# remove the identifier column from the training_data
training_data = training_data.drop('IDENTIFIER', axis = 1)

#%%print the data types of training_data columns
print(training_data.dtypes)


#%%
def create_model(data, feature):
    """
    This function takes in a dataframe and a feature and returns a model that can be used to predict the feature.
    """
    #from missing_cols, remove the feature column
    missing_cols.remove(feature)

    #from the data, drop missing_cols
    data = data.drop(missing_cols, axis = 1)

    # take all the rows that contain missing values in the feature and put them in a new dataframe
    missing_data = data[data[feature].isnull()]

    #take all the rows that do not contain missing values in the feature and put them into the data
    data = data[data[feature].notnull()]

    #%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
    #Define the X and y
    X = data.drop(feature, axis = 1)
    y = data[feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    #KNN
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    knn_cv.fit(X_train, y_train)
    print(knn_cv.best_params_)
    print(knn_cv.best_score_)

    #Decision tree
    tree_cv = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
    tree_cv.fit(X_train, y_train)
    print(tree_cv.best_params_)
    print(tree_cv.best_score_)

    #Linear regression with gridsearchcv
    #linreg_cv = GridSearchCV(LinearRegression(), param_grid_lr, cv=5)
    #linreg_cv.fit(X_train, y_train)
    #print(linreg_cv.best_params_)
    #print(linreg_cv.best_score_)

    #SVM with gridsearchcv
    svm_cv = GridSearchCV(SVC(), param_grid_svm, cv=5)
    svm_cv.fit(X_train, y_train)
    print(svm_cv.best_params_)
    print(svm_cv.best_score_)


#%%
create_model(training_data,missing_cols[0])

#%% use tree_model to predict the AVERAGE_CLOUDINESS feature of training_data_missing_cloudiness and put the result in a new dataframe
training_data_missing_cloudiness['AVERAGE_CLOUDINESS'] = tree_model.predict(training_data_missing_cloudiness)
# %% concatenate training_data_missing_cloudiness and training_data
training_data = pd.concat([training_data, training_data_missing_cloudiness])

# %% print the number of elements of training_data
print(len(training_data))
# %% export the training_data to a csv file
training_data.to_csv('training_data_clean_fullDate_with_predictions.csv', index = False)

# %% now we create a IDENTIFIER feature in the training_data dataframe, which is created by concatenating the YEAR, MONTH, DAY, HOUR, MINUTE and SECOND features, each as a string
training_data['IDENTIFIER'] = training_data['YEAR'].astype(str) + '-' + training_data['MONTH'].astype(str) + '-' + training_data['DAY'].astype(str) + '-' + training_data['HOUR'].astype(str) + '-' + training_data['MINUTE'].astype(str) + '-' + training_data['SECOND'].astype(str)

# %% export the training_data to a csv file
training_data.to_csv('training_data_clean_fullDate_with_predictions_and_identifier.csv', index = False)

# %% 
