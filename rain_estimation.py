#%%
import pandas as pd

#read training_data_clean_identifier_cloudinesspredictions.csv as training_data
training_data = pd.read_csv('training_data_clean_identifier_cloudinesspredictions.csv')

#%% drop the IDENTIFIER column
training_data.drop(['IDENTIFIER'], axis=1, inplace=True)

# %% ok, now we have the new dataframe which has all the predictions of cloudiness, but not the average_rain predictions
# this means that we have to create a dataframe that contains all the rows of the training_data that are empty in the AVERAGE_RAIN feature

# take all the rows that contain missing values in the AVERAGE_RAIN feature and put them in a new dataframe
training_data_missing_rain = training_data[training_data.AVERAGE_RAIN.isnull()]

# drop the AVERAGE_RAIN feature in training_data_missing_rain
training_data_missing_rain.drop(['AVERAGE_RAIN'], axis=1, inplace=True)

# %% drop all the rows that contain missing values in the AVERAGE_RAIN feature in the training_data
training_data = training_data.dropna(subset=['AVERAGE_RAIN'])

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# # Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
#Define the X and y
X = training_data.drop('AVERAGE_RAIN', axis = 1)
y = training_data['AVERAGE_RAIN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# %% now we create a KNN model, a decision tree model as well as a SVM model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# rate the knn model
from sklearn.metrics import accuracy_score
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(knn_accuracy)

#%% perform grid search to find the best parameters for the KNN model
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn_grid = GridSearchCV(knn, param_grid, cv=5)
knn_grid.fit(X_train, y_train)
knn_grid.best_params_

#%% print accuracy of the best model
knn_best = KNeighborsClassifier(n_neighbors=10)
knn_best.fit(X_train, y_train)
knn_best_predictions = knn_best.predict(X_test)
knn_best_accuracy = accuracy_score(y_test, knn_best_predictions)
print(knn_best_accuracy)

#%% train a decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'max_depth': [1,3,5,7,9,11]}
]

tree_cv = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
tree_cv.fit(X_train, y_train)

print(tree_cv.best_params_)
print(tree_cv.best_score_)

# %% run a gridsearch on a SVM model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

svm_cv = GridSearchCV(SVC(), param_grid, cv=5)
svm_cv.fit(X_train, y_train)

print(svm_cv.best_params_)
print(svm_cv.best_score_)

#%%use the best parameters to train a SVM model
svm = SVC(C=10, gamma=0.001, kernel='rbf')

#%% use the svm model in order to create a AVERAGE_RAIN prediction for the training_data_missing_rain dataframe

training_data_missing_rain['AVERAGE_RAIN'] = svm.predict(training_data_missing_rain)

# %% concatenate the training_data_missing_rain dataframe with the training_data dataframe
training_data_missing_rain = training_data_missing_rain.append(training_data)

#%% print the length of the training_data_missing_rain dataframe
print(len(training_data_missing_rain))

# %% export the training_data_missing_rain dataframe to a csv file
training_data_missing_rain.to_csv('training_data_rain_cloudiness_predicted.csv', index=False)
# %%
