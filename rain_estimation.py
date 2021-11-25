#%%
import pandas as pd

#%%
# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data_cleaned_cloudiness_estimation.csv', encoding='ISO-8859-1')

#%% for 

#%% (task-specific cleaning)
# drop all the rows where AVERAGE_CLOUDINESS is null
training_data = training_data.dropna(subset = ['AVERAGE_RAIN'])

# drop the AVERAGE_RAIN feature from the training_data
training_data = training_data.drop('AVERAGE_RAIN', axis = 1)

#%%use k-neighbors classification in KNeighborsClassifier to predict the AVERAGE_CLOUDINESS feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
#Define the X and y
X = training_data.drop('AVERAGE_CLOUDINESS', axis = 1)
y = training_data['AVERAGE_CLOUDINESS']


#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#%%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% apply grid search to find the best parameters for the KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid = [
  {'n_neighbors': [1,3,5,7,9,11]}
]

knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

print(knn_cv.best_params_)
print(knn_cv.best_score_)
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

#%% train a svm classifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'C': [1,3,5,7,9,11]}
]

svm_cv = GridSearchCV(SVC(), param_grid, cv=5)
svm_cv.fit(X_train, y_train)

print(svm_cv.best_params_)
print(svm_cv.best_score_)
# %% use a decision tree clasifier with max_depth = 7
tree_model = DecisionTreeClassifier(max_depth = 7)
tree_model.fit(X_train, y_train)

#use the model in order to predict the AVERAGE_CLOUDINESS feature in the training_data
y_pred = tree_model.predict(X_test)

tree_model.pre

#%%
#get all the rows of training_data that contain a value for AVERAGE_CLOUDINESS
training_data_with_cloudiness = training_data.dropna(subset = ['AVERAGE_CLOUDINESS'], how = 'any')

#get all the rows of training_data that do not contain a value for AVERAGE_CLOUDINESS
training_data_no_cloudiness = training_data.dropna(subset = ['AVERAGE_CLOUDINESS'])

#%% drop the AVERAGE_CLOUDINESS feature from the training_data_no_cloudiness
training_data_no_cloudiness = training_data_no_cloudiness.drop('AVERAGE_CLOUDINESS', axis = 1)

#%%print all the features of training_data_no_cloudiness
print(training_data_no_cloudiness.columns)

#print all the features that the tree_model expects

#%%now use the tree_model to predict the AVERAGE_CLOUDINESS feature in training_data_no_cloudiness
y_pred = tree_model.predict(training_data_no_cloudiness)

#and add the predicted values to the training_data_no_cloudiness dataframe
training_data_no_cloudiness['AVERAGE_CLOUDINESS'] = y_pred

#now combine the training_data_with_cloudiness and training_data_no_cloudiness dataframes
training_data = pd.concat([training_data_with_cloudiness, training_data_no_cloudiness])


#%% make a new feature called IDENTIFIER which is formed by adding the HOUR, DAY, MONTH and YEAR features
training_data['IDENTIFIER'] = training_data['HOUR'] + training_data['DAY'] + training_data['MONTH'] + training_data['YEAR']

#check if any element in the IDENTIFIER feature appears more than once
training_data['IDENTIFIER'].duplicated().any()

#%%now export the new training_data to a csv file called training_data_cleaned_cloudiness_estimation.csv
training_data.to_csv('training_data_cleaned_cloudiness_estimation.csv', index = False)


# %%
