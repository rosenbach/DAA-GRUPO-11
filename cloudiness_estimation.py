#%%
import pandas as pd

#%%
# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data_clean_fullDate.csv', encoding='ISO-8859-1')

# drop the AVERAGE_RAIN feature from the training_data
training_data = training_data.drop('AVERAGE_RAIN', axis = 1)

#%% check how many rows contain missing values in the AVERAGE_CLOUDINESS feature
training_data.AVERAGE_CLOUDINESS.isnull().sum()

# take all the rows that contain missing values in the AVERAGE_CLOUDINESS feature and put them in a new dataframe
training_data_missing_cloudiness = training_data[training_data.AVERAGE_CLOUDINESS.isnull()]

#drop the AVERAGE_CLOUDINESS feature from the training_data_missing_cloudiness dataframe
training_data_missing_cloudiness = training_data_missing_cloudiness.drop(['AVERAGE_CLOUDINESS'], axis=1)

#%% (task-specific cleaning)
# drop all the rows where AVERAGE_CLOUDINESS is null
training_data = training_data.dropna(subset = ['AVERAGE_CLOUDINESS'])



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

#%% plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd')
plt.show()

#%%plot the classification_report

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd')
plt.show()

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
# %% use the best parameters to train a new decision tree classifier
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(max_depth = 7)
tree_model.fit(X_train, y_train)

#%% print training_data_missing_cloudiness
print(training_data_missing_cloudiness.info())
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
