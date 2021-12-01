#%%
import pandas as pd

#import marketing_campaign.csv, which is using tab as a separator
data = pd.read_csv('marketing_campaign.csv', sep='\t')

#  get all the features of type object
object_cols = [col for col in data.columns if data[col].dtype == 'object']
#drop object_cols from data
data.drop(object_cols, axis=1, inplace=True)
#drop id column
data.drop(['ID'], axis=1, inplace=True)
# remove all the rows with missing values
data.dropna(axis=0, inplace=True)

# create a now column Year_Tenths by taking Year_Birth and removing the last digit
data['Year_Tenths'] = data['Year_Birth'].apply(lambda x: int(str(x)[:-1]))

#add a zero as a new digit
data['Year_Tenths'] = data['Year_Tenths'].apply(lambda x: str(x) + '0')

#turn to int
data['Year_Tenths'] = data['Year_Tenths'].astype(int)

#%% print the unique values of Year_Tenths
print(data['Year_Tenths'].unique())

#%% train a knn model in order to predict the Year_Tenths feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Year_Tenths'], axis=1),
                                                    data['Year_Tenths'], test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# %% predict the income of the test set
y_pred = knn.predict(X_test)

#%% print the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# %%
