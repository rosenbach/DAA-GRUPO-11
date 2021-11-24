#%%
import pandas as pd
# load the training_data.csv using the pandas.read_csv() function with UTF-8 encoding
# and assign the result to the variable training_data
training_data = pd.read_csv('training_data_clean.csv', encoding='ISO-8859-1')

#%% print all the features of the training_data with the type object

#in the record_date feature, convert the date to a datetime object
training_data['record_date'] = pd.to_datetime(training_data['record_date'])

#convert the record_date to a new feature called hour
training_data['hour'] = training_data['record_date'].dt.hour

#print the object types of the training_data
print(training_data.dtypes)


#%% drop the record_date column
training_data = training_data.drop('record_date', axis = 1)
# drop the city_name column
training_data = training_data.drop('city_name', axis = 1)

# drop the AVERAGE_RAIN feature from the training_data
training_data = training_data.drop('AVERAGE_RAIN', axis = 1)

# drop all the rows where AVERAGE_CLOUDINESS is null
training_data = training_data.dropna(subset = ['AVERAGE_CLOUDINESS'])

#%%use k-neighbors classification in KNeighborsClassifier to predict the AVERAGE_CLOUDINESS feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


#%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
#Define the X and y
X = training_data.drop('AVERAGE_CLOUDINESS', axis = 1)

#%%
y = training_data['AVERAGE_CLOUDINESS']


#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#%% print all the nan values of X_train
print(y_train.isnull().sum())

#%%
knn = KNeighborsClassifier(n_neighbors=5)


#%%
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

# %%
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

# %%
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

# %%
print(svm_cv.best_params_)
print(svm_cv.best_score_)