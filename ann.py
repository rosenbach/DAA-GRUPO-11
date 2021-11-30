#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

RANDOM_SEED = 2021

print("TensorFlow version:", tf.__version__)
# %% load training_data_michael.csv
training_data = pd.read_csv('training_data_michael.csv')

#%%
test_data = pd.read_csv('test_data_michael.csv')
# create a new dataframe called final_results with the feature RowId
final_results = pd.DataFrame(test_data['RowId'], columns=['RowId'])

#%% in the RowId feature, add 1 to each element
final_results['RowId'] = final_results['RowId'] + 1

#%%drop IDENTIFIER and RowId
test_data = test_data.drop(['IDENTIFIER', 'RowId'], axis=1)

# Let's scale the features of test_data between [0,1]
scaler_test_data = MinMaxScaler(feature_range=(0, 1)).fit(test_data)
test_data_scaled = pd.DataFrame(scaler_test_data.transform(test_data[test_data.columns]), columns=test_data.columns)


#%% print the colnames of training_data
print(training_data.columns)
# %% define X and y
X = training_data.drop(['AVERAGE_SPEED_DIFF'], axis=1)
y = training_data[['AVERAGE_SPEED_DIFF']]

# Let's scale the features between [0,1]
scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(y)
X_scaled = pd.DataFrame(scaler_X.transform(X[X.columns]), columns=X.columns)
y_scaled = pd.DataFrame(scaler_y.transform(y[y.columns]), columns=y.columns)

# %%
def build_model(activation='relu', learning_rate=0.01):
    #Create a sequential model (with three layers - last one is the output layer)
    model = Sequential()
    model.add(Dense(16,input_dim=X_scaled.shape[1], activation=activation))
    model.add(Dense(8, activation=activation))
    model.add(Dense(1, activation='relu'))

    #Compile the model
    #Define the loss function, the optimizer and metrics to be used
    model.compile(
        loss='mae', 
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        metrics=['mae', 'mse']
        )
    return model

#%% split to test and training data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=RANDOM_SEED)
TUNING_DICT = {
    'activation': ['relu', 'sigmoid'],
    'learning_rate': [0.01, 0.001]
    }

#%% now use the KFold API with k=5, the KerasRegressor API and GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

model = KerasRegressor(build_fn=build_model, epochs=20,batch_size=32)
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=TUNING_DICT,
    scoring='neg_mean_absolute_error',
    cv=kf, 
    verbose=1)

grid_search.fit(X_train, y_train,validation_split=0.2,verbose=1)

# %% print the best parameters
print("Best parameters:", grid_search.best_params_)
# print the best score
print("Best score:", grid_search.best_score_)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# %%
best_mlp_model = grid_search.best_estimator_# Did the model overfit?

def plot_learning_curve(history, metric='neg_mean_absolute_error'):
    plt.figure(figsize=(8,4))
    plt.title("Training Loss vs Validation Loss")
    plt.plot(history.epoch, history.history['loss'],label="train")
    plt.plot(history.epoch, history.history['val_loss'],label="val")
    plt.ylabel("Training " + metric)
    plt.xlabel("Epochs")
    plt.legend()

plot_learning_curve(best_mlp_model.model.history, metric="neg_mean_absolute_error")


# %% obtain predictions
predictions = best_mlp_model.predict(X_test)
predictions = predictions.reshape(predictions.shape[0],1)

#%%
predictions_unscaled = scaler_y.inverse_transform(predictions)

y_test_unscaled = scaler_y.inverse_transform(y_test)
y_test_unscaled[:5]
#%%
predictions_unscaled[:5]


# %% predict on test_data_scaled
predictions_test = best_mlp_model.predict(test_data_scaled)
predictions_test = predictions_test.reshape(predictions_test.shape[0],1)

#%%unscale the predictions_test
predictions_test_unscaled = scaler_y.inverse_transform(predictions_test)

#%% print unique values of predictions_test
print(np.unique(predictions_test_unscaled))

#print the range of predictions_test_unscaled
print(np.min(predictions_test_unscaled), np.max(predictions_test_unscaled))


# %% round predictions_test_unscaled
predictions_test_unscaled_round = np.round(predictions_test_unscaled, decimals=0)

#%% save predictions_test_unscaled_round to csv
predictions_test_unscaled_df = pd.DataFrame(predictions_test_unscaled_round, columns=['AVERAGE_SPEED_DIFF'])
predictions_test_unscaled_df.to_csv('predictions_test_unscaled_round.csv', index=False)


#%%add a new column called AVERAGE_SPEED_DIFF
final_results['AVERAGE_SPEED_DIFF'] = predictions_test_unscaled_round

#%%#map AVERAGE_SPEED_DIFF to "None", "Low", "Medium", "High", "Very_High"
final_results['AVERAGE_SPEED_DIFF'] = final_results['AVERAGE_SPEED_DIFF'].map({0: "None", 1: "Low", 2: "Medium", 3: "High", 4: "Very_High"})


#save final_results to csv
final_results.to_csv('final_results.csv', index=False)


# %%
