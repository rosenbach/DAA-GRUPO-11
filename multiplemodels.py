#%% import pandas
import pandas as pd

# read training_data_rain_cloudiness_predicted
training_data = pd.read_csv('training_data_rain_cloudiness_predicted.csv', encoding='utf-8')

# print all the column names
print(training_data.columns)

#%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
#Define the X and y
X = training_data.drop('AVERAGE_SPEED_DIFF', axis = 1)
y = training_data['AVERAGE_SPEED_DIFF']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#%% train a KNN, decision tree and SVM model on the training data for the columns AVERAGE_SPEED_DIFF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#%%run a gridsearch to find the best parameters for the KNN model
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)
print(knn_cv.best_score_)

