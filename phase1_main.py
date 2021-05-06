
# ## EEE485 - Project Phase I
# ### Imports

import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as LA
import smote
import kNN
import fcn

# ### Load Data
raw_data = pd.read_csv("data.csv")
bankrupt_pd = raw_data["Bankrupt?"]
features_pd = raw_data.drop(["Bankrupt?"], axis=1)

# ### Check for NAN and Duplicate Values
print("NAN values:", [col for col in features_pd if features_pd[col].isna().sum() > 0])
print("Duplicates:", features_pd.duplicated().sum())
# We now know that we do not have any missing or duplicate data

# ### Evaluate Data Imbalance
unstable_initial = (raw_data["Bankrupt?"] == 1).sum()
stable_initial = (raw_data["Bankrupt?"] == 0).sum()
print("Data Size:", raw_data.shape[0])
print("# of stable companies:", stable_initial )
print("# of unstable companies:", unstable_initial )
print("Unstable to Stable Ratio: ", unstable_initial/stable_initial)

 ### PLOTS
raw_data.hist(figsize = (50,40), bins = 50)
#plt.show()
f, axes = plt.subplots(ncols=4, figsize = (24,6) )
sns.boxplot(x="Bankrupt?", y=" Cash/Total Assets", data=raw_data, ax = axes[0] )
axes[0].set_title("Bankrupt vs Cash/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Current Assets/Total Assets", data=raw_data, ax = axes[1] )
axes[1].set_title("Bankrupt vs Current Assets/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Net worth/Assets", data=raw_data, ax = axes[2] )
axes[2].set_title("Bankrupt vs Net worth/Assets")
sns.boxplot(x="Bankrupt?", y=" Cash/Current Liability", data=raw_data, ax = axes[3] )
axes[3].set_title("Bankrupt vs Cash/Current Liability")
#plt.show()

# ### Outlier Removal Using IQR
clean_data = raw_data.copy(deep=True)
for col in features_pd:
    clean_data = fcn.remove_outlier(raw_data[col], str(col), raw_data)
clean_data = clean_data.reset_index(drop=True)


 ### Plots with Outliers Removed
clean_data.hist(figsize = (50,40), bins = 50)
#plt.show()
f, axes = plt.subplots(ncols=4, figsize = (24,6) )
sns.boxplot(x="Bankrupt?", y=" Cash/Total Assets", data=clean_data, ax = axes[0] )
axes[0].set_title("Bankrupt vs Cash/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Current Assets/Total Assets", data=clean_data, ax = axes[1] )
axes[1].set_title("Bankrupt vs Current Assets/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Net worth/Assets", data=clean_data, ax = axes[2] )
axes[2].set_title("Bankrupt vs Net worth/Assets")
sns.boxplot(x="Bankrupt?", y=" Cash/Current Liability", data=clean_data, ax = axes[3] )
axes[3].set_title("Bankrupt vs Cash/Current Liability")
#plt.show()

# ### SMOTE
print("SMOTE")
minority = clean_data[clean_data["Bankrupt?"] == 1] # Extract minority samples from data
smt = smote.Smote( minority.to_numpy() ) # Initialize the SMOTE class
oversamples = smt.oversample(N=2600) # Employ SMOTE oversampling

smote_data = clean_data.copy(deep=True) # Cleared from outliers and oversampled
oversamples_pd = pd.DataFrame(oversamples, columns = clean_data.columns)
smote_data = smote_data.append(oversamples_pd)
smote_data = smote_data.reset_index(drop=True)

unstable_smote = (smote_data["Bankrupt?"] == 1).sum()
stable_smote = (smote_data["Bankrupt?"] == 0).sum()
print("Oversampled Data Size:", smote_data.shape[0])
print("Number of Stable Companies:", stable_smote)
print("Number of Unstable Companies (with SMOTE):", unstable_smote)
print("Unstable to Stable Ratio: ", unstable_smote/stable_smote, sep="")


smote_data["Bankrupt?"].hist()
#plt.show()
f, axes = plt.subplots(ncols=4, figsize = (24,6) )
sns.boxplot(x="Bankrupt?", y=" Cash/Total Assets", data=smote_data, ax = axes[0] )
axes[0].set_title("Bankrupt vs Cash/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Current Assets/Total Assets", data=smote_data, ax = axes[1] )
axes[1].set_title("Bankrupt vs Current Assets/Total Assets")
sns.boxplot(x="Bankrupt?", y=" Net worth/Assets", data=smote_data, ax = axes[2] )
axes[2].set_title("Bankrupt vs Net worth/Assets")
sns.boxplot(x="Bankrupt?", y=" Cash/Current Liability", data=smote_data, ax = axes[3] )
axes[3].set_title("Bankrupt vs Cash/Current Liability")
#plt.show()

# ### Test Train Split
test_ratio = 0.1
#Smote
train_sm, test_sm = fcn.test_train_split(smote_data, test_ratio )
X_train_sm = train_sm.drop(["Bankrupt?"], axis=1)
Y_train_sm = train_sm["Bankrupt?"]
X_test_sm = test_sm.drop(["Bankrupt?"], axis=1)
Y_test_sm = test_sm["Bankrupt?"]
#No Smote
train, test = fcn.test_train_split(clean_data, test_ratio )
X_train = train.drop(["Bankrupt?"], axis=1)
Y_train = train["Bankrupt?"]
X_test = test.drop(["Bankrupt?"], axis=1)
Y_test = test["Bankrupt?"]

# ### k-Nearest Neighbors Classifier (with and without SMOTE)
print("k-NN")
knn_classifier = kNN.k_NN_classifier(X_train.to_numpy(), Y_train.to_numpy() )
Y_test_pd = Y_test.to_numpy()
X_test_pd = X_test.to_numpy()
knn_preds = np.zeros_like(Y_test_pd)
classification_time_start = time.time()
for idx, test in enumerate(X_test_pd):
    knn_preds[idx] = knn_classifier.classify(test)
classification_time_end = time.time()
print("Time Spent on Classifying", Y_test.shape[0], "Samples is:", classification_time_end-classification_time_start)

knn_classifier_sm = kNN.k_NN_classifier(X_train_sm.to_numpy(), Y_train_sm.to_numpy() )
Y_test_sm_pd = Y_test_sm.to_numpy()
X_test_sm_pd = X_test_sm.to_numpy()
knn_sm_preds = np.zeros_like(Y_test_sm_pd)
classification_time_sm_start = time.time()
for idx, test in enumerate(X_test_sm_pd):
    knn_sm_preds[idx] = knn_classifier_sm.classify(test)
classification_time_sm_end = time.time()
print("Time Spent on Classifying", Y_test_sm.shape[0], "Samples is:", classification_time_sm_end-classification_time_sm_start)
print()
print("Confusion Matrix Without SMOTE")
conf_matrix = fcn.confusion_matrix(Y_test_pd, knn_preds, ret = True)
print("Recall:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0]) * 100 )
print("Precision:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]) * 100 )

print()
print("Confusion Matrix With SMOTE")
conf_matrix_sm = fcn.confusion_matrix(Y_test_sm_pd, knn_sm_preds, ret = True)
print("Recall:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[1,0]) * 100 )
print("Precision:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[0,1]) *100 )


