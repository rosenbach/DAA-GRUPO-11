# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 07:30:40 2021

@author: PC
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from scipy import stats

def prepare_data(data):
    #Rename text with unknown character, and combine nuvens quebradas +  nuvens quebrados ( both are cloud broken but spelled differently )
    data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].astype('string') 
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*claro.*$)', 'ceu claro')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*nublado.*$)', 'ceu pouco nublado')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*limpo.*$)', 'ceu limpo')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*quebrados.*$)', 'nuvens quebradas')
    
    #Ordinal Encoding on column LUMINOSITY
    lumi_dict = {'DARK' : 1, 'LOW_LIGHT': 2, 'LIGHT': 3}
    data['LUMINOSITY'] = data.LUMINOSITY.map(lumi_dict)
    
    cloud_dict = {'ceu claro' : 1, 'ceu limpo': 2, 'nuvens dispersas': 3, 'nuvens quebradas': 4, 'algumas nuvens': 5, 'ceu pouco nublado': 6}
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.map(cloud_dict)
    
    
    speed_dict = {'None' : 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very_High': 5}
    data['AVERAGE_SPEED_DIFF'] = data.AVERAGE_SPEED_DIFF.map(speed_dict)
    
    
    data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].fillna(method='bfill') #Fill na using next value
    
    #Split Date in 4 new columns (Day, Month, Year, Hour)
    data['record_date'] = pd.to_datetime( data['record_date'])
    data['Day'] = data['record_date'].dt.day
    data['month'] = data['record_date'].dt.month
    data['Year'] = data['record_date'].dt.year
    data['Hour'] = data['record_date'].dt.hour
    #z=np.abs(stats.zscore(data))
    #threshold = 3
    #training_data = data[(z < 3).all(axis=1)]
    data['AVERAGE_TIME_DIFF'] = data['AVERAGE_TIME_DIFF'].round(0).astype(int)
    data['AVERAGE_FREE_FLOW_SPEED'] = data['AVERAGE_FREE_FLOW_SPEED'].round(0).astype(int)
    
    data['period'] = (data['Hour']% 24 + 4) // 4
    data['period'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Noon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)
    
    one_hot_period = pd.get_dummies(data['period'])
    data = pd.concat([data, one_hot_period], axis=1)
    data['bin_time'] = data['AVERAGE_TIME_DIFF'].apply(f)
    data['bin_time']= data['bin_time'].astype(str).astype(int)
    
    dates = pd.to_datetime({"year": data.Year, "month": data.month, "day": data.Day})
    data["Day of week"] = dates.dt.dayofweek
    data["Is Weekend"] = dates.dt.dayofweek.apply(lambda x: 1 if x > 4 else 0)
    data = data.drop(['period'], axis = 1)
    

    return data


def prepare_test_data(data):
    #Rename text with unknown character, and combine nuvens quebradas +  nuvens quebrados ( both are cloud broken but spelled differently )
    data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].astype('string') 
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*claro.*$)', 'ceu claro')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*nublado.*$)', 'ceu pouco nublado')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*limpo.*$)', 'ceu limpo')
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.str.replace(r'(^.*quebrados.*$)', 'nuvens quebradas')
    
    #Ordinal Encoding on column LUMINOSITY
    lumi_dict = {'DARK' : 1, 'LOW_LIGHT': 2, 'LIGHT': 3}
    data['LUMINOSITY'] = data.LUMINOSITY.map(lumi_dict)
    
    cloud_dict = {'ceu claro' : 1, 'ceu limpo': 2, 'nuvens dispersas': 3, 'nuvens quebradas': 4, 'algumas nuvens': 5, 'ceu pouco nublado': 6}
    data['AVERAGE_CLOUDINESS'] = data.AVERAGE_CLOUDINESS.map(cloud_dict)

    
    data['AVERAGE_CLOUDINESS'] = data['AVERAGE_CLOUDINESS'].fillna(method='bfill') #Fill na using next value
    
    #Split Date in 4 new columns (Day, Month, Year, Hour)
    data['record_date'] = pd.to_datetime( data['record_date'])
    data['Day'] = data['record_date'].dt.day
    data['month'] = data['record_date'].dt.month
    data['Year'] = data['record_date'].dt.year
    data['Hour'] = data['record_date'].dt.hour
    
    data['AVERAGE_TIME_DIFF'] = data['AVERAGE_TIME_DIFF'].round(0).astype(int)
    data['AVERAGE_FREE_FLOW_SPEED'] = data['AVERAGE_FREE_FLOW_SPEED'].round(0).astype(int)
    
    data['period'] = (data['Hour']% 24 + 4) // 4
    data['period'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Noon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)
    
    one_hot_period = pd.get_dummies(data['period'])
    data = pd.concat([data, one_hot_period], axis=1)
    data['bin_time'] = data['AVERAGE_TIME_DIFF'].apply(f)
    data['bin_time']= data['bin_time'].astype(str).astype(int)
    
    dates = pd.to_datetime({"year": data.Year, "month": data.month, "day": data.Day})
    data["Day of week"] = dates.dt.dayofweek
    data["Is Weekend"] = dates.dt.dayofweek.apply(lambda x: 1 if x > 4 else 0)
    data = data.drop(['period'], axis = 1)
    
    return data

def f(x):
    if (x == 0):
        return '0'
    if (x > 0) and (x <= 5):
        return '1'
    elif (x > 5) and (x <= 10):
        return '2'
    elif (x > 10 ) and (x <= 15):
        return'3'
    elif (x > 15) and (x <= 20) :
        return '4'
    elif (x > 20) and (x <= 25):
        return'5'
    elif (x > 25 ) and (x <= 30):
        return'6'
    elif (x > 30) and (x <= 35) :
        return'7'
    elif (x > 35 ) and (x <= 40):
        return'8'
    elif (x > 40) and (x <= 45) :
        return'9'
    elif (x > 45 ) and (x <= 50):
        return'10'
    elif (x > 50) and (x <= 55) :
         return '11'
    elif (x > 55 ) and (x <= 60):
        return'12'
    elif (x > 60 ) and (x <= 65) :
        return '13'
    elif (x > 65) and (x <= 70) :
         return '14'
    elif (x > 70 ) and (x <= 75):
        return'15'
    elif (x > 75) and (x <= 80) :
        return'16'
    elif (x > 80) and (x <= 85) :
        return'17'
    elif (x > 85) and (x <= 90) :
        return'18'
    elif (x > 90) and (x <= 95) :
        return'19'
    elif (x > 95) and (x <= 100) :
        return'20'
    elif (x > 100) and (x <= 105) :
        return'21'
    elif (x > 105) and (x <= 110) :
        return'22'
    elif (x > 110) and (x <= 115) :
        return '23'
    elif (x > 115) and (x <= 120) :
        return'24'
    elif (x > 120) and (x <= 126) :
        return'25'
    elif (x > 126) :
        return'26'

param_dict = {
    "criterion":['gini', 'entropy'],
    "max_depth":range(1,10),
    "min_samples_split":range(1,10),
    "min_samples_leaf":range(1,5)
}

#param_grid_log = dict()
#param_grid_log['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
#param_grid_log['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
#param_grid_log['C'] = [ 1e-1, 1, 10, 100]

param_grid_svm = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001], 'kernel':['linear','rbf']}


def create_models(data, feature):
    """
        This function takes in a dataframe and a feature and returns a model that can be used to predict the feature.
    """

    #%% Prepare and organize sets of case studies dataset into training data and test, using the sklearn.model_selection.train_test_split(, test_size = 0.3) function
    #Define the X and y
    X = data.drop(feature, axis = 1)
    y = data[feature]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=2021)
    print("The Shape of X : %s, for x train : %s, for x test: %s" % (X.shape,X_train.shape, X_test.shape))
    print("The Shape of y: %s, The Shape of y train : %s, The Shape of test y: %s" %(y.shape, y_train.shape, y_test.shape))
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    #Decision tree
    clf_tree = DecisionTreeClassifier(random_state=2021)
    grid_tree = GridSearchCV(clf_tree, param_grid=param_dict,cv=20,verbose=1,n_jobs=-1)
    grid_tree.fit(X,y)
    print(grid_tree.best_params_)
    print(grid_tree.best_score_)
    #plot_confusion_matrix(grid_tree, X_test, y_test)
    #plt.show()
    #KNN
    #knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv)
    #knn_cv.fit(X, y)
    #print(knn_cv.best_params_)
    #print(knn_cv.best_score_)

    #Logistic Regression
    #logmodel = LogisticRegression(random_state=2021)
    #log_cv = GridSearchCV(logmodel, param_grid_log, scoring='accuracy', n_jobs=-1, cv=cv)
    #log_cv.fit(X,y)
    #print('Best Score: %s' % log_cv.best_score_)
    #print('Best Hyperparameters: %s' % log_cv.best_params_)
    
    #SVM with gridsearchcv
    svm_cv = GridSearchCV(SVC(random_state=2021), param_grid_svm,refit=True, verbose=3)
    svm_cv.fit(X, y)
    print(svm_cv.best_params_)
    print(svm_cv.best_score_)

    #create an array with all the models
    models = [grid_tree, svm_cv]    #return models
    return models






data = pd.read_csv("training_data.csv")
unwanted_columns = ["AVERAGE_RAIN", "city_name","AVERAGE_PRECIPITATION"]
data = data.drop(unwanted_columns, axis=1)
data = data.dropna()
data = prepare_data(data)
data = data.drop(['record_date'],axis=1)#ORDINAL_SPEED_DIFF is AVERAGE_SPEED_DIFF (encoded)
models = create_models(data, "AVERAGE_SPEED_DIFF")

#%% get the best performing model
best_model = max(models, key=lambda x: x.best_score_)
print("O melhor modelo e", best_model)
best_model.best_score_
best_model.best_params_

test_data = pd.read_csv('test_data.csv')
#sort by the 'Unnamed: 0' column
#remove the IDENTIFIER Column
unwanted_columns = ["AVERAGE_RAIN", "city_name","AVERAGE_PRECIPITATION"]
test_data = test_data.drop(unwanted_columns, axis=1)
#test_data = test_data.dropna()
test_data = prepare_test_data(test_data)
test_data = test_data.drop(['record_date'],axis=1)#ORDINAL_SPEED_DIFF is AVERAGE_SPEED_DIFF (encoded)
#predictions = best_model.predict(test_data.drop(['Unnamed: 0'], axis=1))
predictions = best_model.predict(test_data)
#create a new "AVERAGE_SPEED_DIFF" column in the test_data 
test_data['AVERAGE_SPEED_DIFF'] = predictions

submit = pd.DataFrame()
submit['Speed_Diff'] = predictions
submit.index +=1
submit['RowId'] = submit.index
pd.unique(submit['Speed_Diff'])
submit['Speed_Diff'] = submit['Speed_Diff'].map({1: "None", 2: "Low", 3: "Medium", 4: "High", 5: "Very_High"})

submit = submit[['RowId', 'Speed_Diff']]
#rename AVERAGE_SPEED_DIFF to Speed_Diff
#submit = test_data.rename(columns={'AVERAGE_SPEED_DIFF': 'Speed_Diff'})

#%%export the test_data to a csv file
submit.to_csv('results_full.csv', index=False)












