
# coding: utf-8

# In[96]:


# conventional way to import pandas
import pandas as pd
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

from sklearn.svm import SVC 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
import pylab as pl
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# DF TO EXCEL
from pandas import ExcelWriter
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from pandas import ExcelWriter


# In[97]:


# read CSV file from the 'data' subdirectory using a relative path
Dataset= pd.read_csv('C:\\Users\\Admin\\Desktop\\CS3123\\Data_Set\\Dataset.csv')

# display the first 5 rows
Dataset.head()


# In[98]:


# Drop index and unnessery featuer 
Dataset.drop(['0', '0.1','3','187'], axis=1,inplace=True)


# In[99]:


#Drop all features except (Age - gender -Height -weight)
#subset=Dataset.drop(Dataset.columns.to_series()["6":"445"], axis=1)
subset=Dataset


# In[100]:


subset.isna().sum()


# In[101]:



#Missing da


# In[102]:


import numpy as np

# Drop sample that has missing value in Gender feater 
H_Data=subset.drop(subset.index[1319])

H_Data.shape


# In[103]:


#removing outliers
H_Data = H_Data[(H_Data['1']> 0)]
H_Data= H_Data[(H_Data['1']<100)]


# In[104]:


H_Data.isna().sum()


# In[105]:


#Removing outliers from Hight
Data = H_Data[(H_Data['5']> 145)]
Data = H_Data[(H_Data['5']<195)]


# # Visualizing data using seaborn
# 
# **Seaborn:** Python library for statistical data visualization built on top of Matplotlib

# In[106]:


Data.isna().sum()


# In[107]:


#Removing outliers from Weight
Data = Data[(Data['4']> 38)]
Data = Data[(Data['4']<250)]


# In[108]:


Data.isna().sum()


# In[109]:


Data_BN=Data.dropna(axis='columns')


# In[110]:


Data_BN.isna().sum()


# In[111]:


# Female subset
F_Data=Data_BN[Data_BN['2'] == 0.0]
#Male subset
M_Data=Data_BN[Data_BN['2'] == 1.0]


# In[112]:


F_Data.shape


# In[113]:


M_Data.shape


# In[114]:


BF_Data=F_Data.sample(943, random_state=0)


# In[115]:


frames = [M_Data, BF_Data]


# In[116]:


MF_Data = pd.concat(frames)


# In[117]:


MF_Data.shape


# In[118]:


# select a Series from the DataFrame
y = MF_Data['2']
TT=MF_Data.drop(['2'],axis=1)
X=TT[:]
# print the first 5 values
y.shape


# # Splitting Data

# In[119]:


X_train, X_test, y_train, y_test = train_test_split( X,y,test_size=0.25,random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)


# # #  Feature selection_ with 5-fold CV Logistic Regression

# In[120]:


### Build RF classifier to use in feature selection
logreg = LogisticRegression()
# Build step forward feature selection
sfs_LR = sfs(logreg,
           k_features=48,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs_LR = sfs_LR.fit(X_train, y_train)


# In[121]:


# Which features?
feat_cols = list(sfs_LR.k_feature_idx_)
print(feat_cols)


# # Build ML model with best K Features
# 

# # Logistic Regression

# In[122]:


# Build full model with selected features
clf =LogisticRegression()
clf.fit(X_train.iloc[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test.iloc[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[123]:


# Build full model with all features
clf =LogisticRegression()
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# #  Decision Tree  with  28 FS

# In[124]:


#10-fold cross-validation with Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train.iloc[:, feat_cols], y_train)

y_train_pred = DT.predict(X_train.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = DT.predict(X_test.iloc[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[125]:


# Build full model with all features
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

y_train_pred = DT.predict(X_train)
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = DT.predict(X_test)
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# # SVM with 28 FS

# In[126]:



# Build full model with selected features

SVM= SVC(kernel='linear') 
SVM.fit(X_train.iloc[:, feat_cols], y_train)

y_train_pred = SVM.predict(X_train.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = SVM.predict(X_test.iloc[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[127]:


# Build full model on ALL features, for comparison
SVM= SVC(kernel='linear') 
SVM.fit(X_train, y_train)

y_train_pred = SVM.predict(X_train)
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = SVM.predict(X_test)
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# # Random Forest

# In[128]:



RF_C = RandomForestClassifier(n_estimators=100, n_jobs=-1)
RF_C.fit(X_train.iloc[:, feat_cols], y_train)

y_train_pred = RF_C.predict(X_train.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = RF_C.predict(X_test.iloc[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))


# In[129]:


RF_C = RandomForestClassifier(n_estimators=100, n_jobs=-1)
RF_C.fit(X_train, y_train)

y_train_pred = RF_C.predict(X_train)
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = RF_C.predict(X_test)
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))

