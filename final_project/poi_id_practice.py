#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

import sys
import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append("../tools/")


# In[2]:


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi" since it is the label
features_list = ['poi','salary', 'bonus',"to_messages","deferral_payments", "total_payments","loan_advances","restricted_stock_deferred","deferred_income","total_stock_value","from_poi_to_this_person","exercised_stock_options","from_messages","from_this_person_to_poi","long_term_incentive","shared_receipt_with_poi","restricted_stock","director_fees"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


# In[3]:


#Let's create a DataFrame to visualize data
df = pd.DataFrame.from_dict(my_dataset, orient='index', dtype=None)



#We transfom NaN to Null values (since NaN is a string)
df = df.replace("NaN", np.nan)

pd.options.display.float_format = '{:10,.0f}'.format
df.describe()

#How many mising values do we have
num_missing_values = df.isnull().sum()
print(num_missing_values)


#Fill missing values with 0
df = df.fillna(0)
df.head()

payment_categories = ['salary', 'bonus', 'long_term_incentive', 'deferred_income',
                      'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments']
stock_value_categories = ['exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']

# Look at the instances where the total we calculate is not equal to the total listed on the spreadsheet
df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories]

df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories]

#Some data has been shifted, let's order it

df.loc[('BELFER ROBERT','deferral_payments')] = 0
df.loc[('BELFER ROBERT','total_payments')] = 3285
df.loc[('BELFER ROBERT','restricted_stock_deferred')] = -44093
df.loc[('BELFER ROBERT','deferred_income')] = -102500
df.loc[('BELFER ROBERT','total_stock_value')] = 0
df.loc[('BELFER ROBERT','expenses')] = 3285
df.loc[('BELFER ROBERT','exercised_stock_options')] = 0
df.loc[('BELFER ROBERT','restricted_stock')] = 44093
df.loc[('BELFER ROBERT','director_fees')] = 102500

df.loc[('BHATNAGAR SANJAY','total_payments')] = 137864
df.loc[('BHATNAGAR SANJAY','restricted_stock_deferred')] = -2604490
df.loc[('BHATNAGAR SANJAY','total_stock_value')] = 15456290
df.loc[('BHATNAGAR SANJAY','expenses')] = 137864
df.loc[('BHATNAGAR SANJAY','exercised_stock_options')] = 15456290
df.loc[('BHATNAGAR SANJAY','other')] = 0
df.loc[('BHATNAGAR SANJAY','restricted_stock')] = 2604490
df.loc[('BHATNAGAR SANJAY','director_fees')] = 0

#Does that solve our problems ?
print(df[df[payment_categories[:-1]].sum(axis='columns') != df['total_payments']][payment_categories])
print(df[df[stock_value_categories[:-1]].sum(axis='columns') != df['total_stock_value']][stock_value_categories])


#Convert salary to float
df['salary'] = df['salary'].astype(float)

#droping rowS THAT WE DON'T NEED
df = df.drop("TOTAL")
df = df.drop("THE TRAVEL AGENCY IN THE PARK")

#The same for bonus
df['bonus'] = df['bonus'].astype(float)

#We drop columns that have the same value
to_drop = []
for column in df.columns:
    if len(pd.unique(df[column]))<2:
        to_drop.append(column)

df =df.drop(to_drop,axis=1)

#Description of the metrics
display(df.describe())
#Box plot of salary
display(sns.boxplot(y =df["salary"]))
#Box plot of Salary divided by POI
display(sns.boxplot(x="poi", y="salary", data=df))

#Simple linear Regression for bonus & salary
display(sns.lmplot(x="salary", y="bonus",hue="poi", data=df))

#How many POIs do we have ?
print(df["poi"].value_counts())

#If we want to use the cleaned data in other projects we can save it in a csv
clean_enron= "clean_enron_df_csv"
dirpath = os.getcwd()
df.to_csv(dirpath+clean_enron,index_label="name")

#we convert it back to dictionary
my_dataset = df.to_dict(orient="index")

### Extract features and labels from dataset for local testing
### We transform from dictionary to list ready for sklearn
data = featureFormat(my_dataset, features_list, sort_keys = True)

### We split the labels and features
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#This allows to scale features, substracting the mean from each feature and then scaling it to unit cariance
from sklearn.preprocessing import StandardScaler
#Pipeline allows to do a sequence of different transformations in just one line
from sklearn.pipeline import Pipeline
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import decomposition
#We will use Grid Search to automatically tune the parameters of our model 

from sklearn.model_selection import GridSearchCV

#Let's separate the data in test and 
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42,stratify=labels)


# In[21]:


#First we will scale the data, then apply PCA to reduce dimensionality and then our classifier
pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reduce_dims', PCA()),
        ('clf', SVC(gamma='auto'))])

param_grid = dict(reduce_dims__n_components=[4,6,8],
                  clf__C=np.logspace(-4, 1, 6),
                  clf__kernel=['rbf','linear'])


grid = GridSearchCV(pipe,param_grid=param_grid,cv=3, n_jobs=1, verbose=2, scoring="accuracy")

grid.fit(features_train, labels_train)

print("\n Best score %0.3f" % grid.best_score_)

print("\n Best parameters = ")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

grid.best_estimator_.score(features_train, labels_train)


# In[18]:


#we make the predictions with the best estimator obtained by Grid Search
preds = grid.best_estimator_.predict(features_test)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(labels_test,preds)))
print('Precision Score : ' + str(precision_score(labels_test,preds)))
print('Recall Score : ' + str(recall_score(labels_test,preds)))
print('F1 Score : ' + str(f1_score(labels_test,preds)))

#Dummy Classifier Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(labels_test,preds)))


# In[ ]:


#Doesn't look good, since the algorithm predicts all our values to be 0a


# In[28]:



from sklearn.linear_model import LogisticRegression



grid_values = {
    "C":[ 65,75, 85], 
    "penalty":["l1","l2"]

}

clf =LogisticRegression()
grid = GridSearchCV(clf, grid_values, scoring='roc_auc', n_jobs=-1, verbose=10, cv=3)
grid.fit(features_train, labels_train)


print('Achieved score %f' % grid.best_score_, 'with params:', grid.best_params_)


# In[30]:


#we make the predictions with the best estimator obtained by Grid Search
preds = grid.best_estimator_.predict(features_test)

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
print('Accuracy Score : ' + str(accuracy_score(labels_test,preds)))
print('Precision Score : ' + str(precision_score(labels_test,preds)))
print('Recall Score : ' + str(recall_score(labels_test,preds)))
print('F1 Score : ' + str(f1_score(labels_test,preds)))

#Dummy Classifier Confusion matrix
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(labels_test,preds)))


# In[34]:


from tester import test_classifier, dump_classifier_and_data

test_classifier(clf, my_dataset, features_list)


# In[31]:


dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:


#no consigues que el dump funcione :/
#ni tampoco el test. Mírate lo de StratifiedShuffleSplit


# 
# 
# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info: 
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 
# 
# 
# 
# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.
# 
# 
