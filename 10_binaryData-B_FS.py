
# coding: utf-8

# In[2]:


# conventional way to import pandas
import pandas as pd
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

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


# In[3]:


# read CSV file from the 'data' subdirectory using a relative path
Dataset= pd.read_csv('C:\\Users\\Admin\\Desktop\\CS3123\\Data_Set\\Dataset.csv')


# In[4]:


Dataset


# In[5]:


# Drop index and unnessery featuer 
Dataset.drop(['0', '0.1','3','4','5','1','187'], axis=1,inplace=True)


# In[6]:


Bi_set=Dataset


# In[7]:


#Missing da
#Bi_set=Bi_set[(Bi_set< 2)]
Bi_set=Bi_set[Bi_set < 2]


# In[8]:


Bi_set


# In[9]:


import numpy as np

# Drop sample that has missing value in Gender feater 
BiN_set=Bi_set.drop(Bi_set.index[1319])

BiN_set.shape


# In[10]:


print(BiN_set)


# In[11]:


#Bi_set=Bi_set.dropna(axis='', how='any')
BiNF_set=BiN_set.fillna(0)


# # Visualizing data using seaborn
# 
# **Seaborn:** Python library for statistical data visualization built on top of Matplotlib

# In[12]:


# conventional way to import seaborn
import seaborn as sns

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


#sns.pairplot(Bi_set,  hue="2",palette="husl")


# In[14]:


# Female subset
F_Data=BiNF_set[BiNF_set['2'] == 0.0]
#Male subset
M_Data=BiNF_set[BiNF_set['2'] == 1.0]


# In[15]:


B_Data=F_Data.sample(1139, random_state=0)


# In[16]:


frames = [M_Data, B_Data]


# In[17]:


MFB_Data = pd.concat(frames)


# In[18]:


M_Data.shape
#writer = pd.ExcelWriter('output.xlsx')
#MFB_Data.to_excel(writer,'Sheet1')
#writer.save()


# In[19]:


F_Data.shape


# 
# # Task 7: Cross Validation CV

# In[20]:


# select a Series from the DataFrame
y = MFB_Data['2']
DT=MFB_Data.drop(['2'],axis=1)
X=DT[:]
# print the first 5 values
y.shape


# In[21]:


# Build step forward feature selection
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

sfs1 = sfs(clf,
           k_features=56,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)


# In[99]:


# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)


# In[126]:


# check the type and shape of y
print(type(y))
print(y.shape)


# # KNN ML Algorithm With Accuracy 79%

# In[127]:


from sklearn.model_selection import cross_val_score
# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#(scores)


# In[131]:


# use average accuracy as an estimate of out-of-sample accuracy
print('Accuracy of K-NN classifier on test set:',scores.mean())


# In[149]:


# search for an optimal value of K for KNN
k_range = list(range(1, 50))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[ ]:


# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=24)
y_pred_K = cross_val_predict(knn, X, y, cv=10)
conf_mat_KNN = confusion_matrix(y, y_pred_K)
print('Accuracy of KNN classifier on test set:',cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())


# In[78]:


plt.clf()
plt.imshow(conf_mat_KNN, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix - KNN - Test Data')
plt.ylabel('Actual')
plt.xlabel('Prediction')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat_KNN[i][j]))
plt.show()


# In[79]:


print(classification_report(y, y_pred_K))


# #  Decision Tree with Accuracy 67%

# In[134]:


#10-fold cross-validation with Decision Tree
DT = DecisionTreeClassifier()
scores = cross_val_score(DT, X, y, cv=10, scoring='accuracy')
y_pred_DT = cross_val_predict(DT, X, y, cv=10)
conf_mat_DT = confusion_matrix(y, y_pred_DT)
#print(scores)


# In[135]:


# use average accuracy as an estimate of out-of-sample accuracy
print('Accuracy of Decision Tree classifier on test set:',scores.mean())


# In[136]:


plt.clf()
plt.imshow(conf_mat_DT, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix - Decision Tree - Test Data')
plt.ylabel('Actual')
plt.xlabel('Prediction')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat_DT[i][j]))
plt.show()


# In[137]:


print(classification_report(y, y_pred_DT))


# #  Logistic Regrassion with Accuracy 73%

# In[138]:


# 10-fold cross-validation with logistic regression
logreg = LogisticRegression()
scores = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')
y_pred_LR= cross_val_predict(logreg, X, y, cv=10)
conf_mat_LR= confusion_matrix(y, y_pred_LR)
#print(scores)


# In[139]:


# use average accuracy as an estimate of out-of-sample accuracy
print('Accuracy of logistic regression classifier on test set:',scores.mean())


# In[140]:


plt.clf()
plt.imshow(conf_mat_LR, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix - logistic regression - Test Data')
plt.ylabel('Actual')
plt.xlabel('Prediction')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat_LR[i][j]))
plt.show()


# In[141]:


print(classification_report(y, y_pred_LR))


# #  Linear Discriminant Analysis with Accuracy 74%

# In[142]:


#10-fold cross-validation with Discriminant Analysis
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=10, scoring='accuracy')
y_pred_LDA= cross_val_predict(lda, X, y, cv=10)
conf_mat_LDA= confusion_matrix(y, y_pred_LDA)
#print(scores)


# In[143]:


# use average accuracy as an estimate of out-of-sample accuracy
print('Accuracy of LDA classifier on test set:',scores.mean())


# In[144]:


plt.clf()
plt.imshow(conf_mat_LDA, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['0','1']
plt.title('Confusion Matrix - LDA - Test Data')
plt.ylabel('Actual')
plt.xlabel('Prediction')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(conf_mat_LDA[i][j]))
plt.show()


# In[145]:


print(classification_report(y, y_pred_LDA))


# In[146]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
y_pred_clf= cross_val_predict(clf, X, y, cv=10)
conf_mat_clf= confusion_matrix(y, y_pred_clf)
#print(scores)
print('Accuracy of NN classifier on test set:',scores.mean())


# In[147]:


from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
scores = cross_val_score(NN, X, y, cv=10, scoring='accuracy')
y_pred_NN= cross_val_predict(NN, X, y, cv=10)
conf_mat_SVM= confusion_matrix(y, y_pred_NN)
#print(scores)
print('Accuracy of NN classifier on test set:',scores.mean())


# In[148]:


from sklearn.svm import SVC  
SVM= SVC(kernel='linear') 
scoresM = cross_val_score(SVM, X, y, cv=10, scoring='accuracy')
y_pred_SVM= cross_val_predict(SVM, X, y, cv=10)
conf_mat_SVM= confusion_matrix(y, y_pred_SVM)
#print(scores)
print('Accuracy of LDA classifier on test set:',scoresM.mean())


# In[98]:


sfs1 = sfs1.fit(X, y)


# In[167]:


# Build full model with selected features

clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X.iloc[:,feat_cols],y)

y_train_pred = clf.predict(X.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y, y_train_pred))


# In[169]:


# Build full model on ALL features, for comparison
clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X, y)

y_train_pred = clf.predict(X)
print('Training accuracy on all features: %.3f' % acc(y, y_train_pred))

