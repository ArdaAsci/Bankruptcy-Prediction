# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## EEE485 - Project Final
# %% [markdown]
# ### Imports

# %%
import csv
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.linalg as LA
import smote
import random
import kNN
import fcn
import PCA
from numba import njit, jit
import scipy
from logistic_regression import Logistic_regression
from GradBoost import BinaryGradientBoostClassify
from PCA import PCAnalyser

# %% [markdown]
# ### Load Data

# %%
raw_data = pd.read_csv("data.csv")
bankrupt_pd = raw_data["Bankrupt?"]
features_pd = raw_data.drop(["Bankrupt?"], axis=1)
raw_data

# %% [markdown]
# ### Check for NAN and Duplicate Values
# 

# %%
print("NAN values:", [col for col in features_pd if features_pd[col].isna().sum() > 0])
print("Duplicates:", features_pd.duplicated().sum())

# %% [markdown]
# ### Evaluate Data Imbalance

# %%
unstable_initial = (raw_data["Bankrupt?"] == 1).sum()
stable_initial = (raw_data["Bankrupt?"] == 0).sum()
print("Data Size:", raw_data.shape[0])
print("# of stable companies:", stable_initial )
print("# of unstable companies:", unstable_initial )
print("Unstable to Stable Ratio: ", unstable_initial/stable_initial)

# %% [markdown]
# ### PLOTS

# %%
raw_data.hist(figsize = (50,40), bins = 50)
plt.show()


# %%
f, axes = plt.subplots(ncols=4, figsize = (24,6) )

sns.boxplot(x="Bankrupt?", y=" Cash/Total Assets", data=raw_data, ax = axes[0] )
axes[0].set_title("Bankrupt vs Cash/Total Assets")

sns.boxplot(x="Bankrupt?", y=" Current Assets/Total Assets", data=raw_data, ax = axes[1] )
axes[1].set_title("Bankrupt vs Current Assets/Total Assets")

sns.boxplot(x="Bankrupt?", y=" Net worth/Assets", data=raw_data, ax = axes[2] )
axes[2].set_title("Bankrupt vs Net worth/Assets")

sns.boxplot(x="Bankrupt?", y=" Cash/Current Liability", data=raw_data, ax = axes[3] )
axes[3].set_title("Bankrupt vs Cash/Current Liability")

plt.show()

# %% [markdown]
# ### Outlier Removal Using IQR

# %%
clean_data = raw_data.copy(deep=True)
for col in features_pd:
    clean_data = fcn.remove_outlier(raw_data[col], str(col), raw_data)
clean_data = clean_data.reset_index(drop=True)
clean_data

# %% [markdown]
# ### PCA

# %%
clean_X = clean_data.drop(["Bankrupt?"], axis=1)
clean_Y = clean_data["Bankrupt?"]
centered_data = clean_X - np.mean(clean_X, axis=0)
pc_analyser = PCAnalyser(centered_data, data_centered=True)
eigen_vals, PCs = pc_analyser.analyse(k=7)
PCA_data = centered_data @ PCs


# %%
plt.figure(figsize=(6,4))
plt.plot(eigen_vals)
plt.xlabel("Index")
plt.ylabel("Eigen Value")
plt.title("Eigen Values of the Principal Components")
plt.xlim( (0, 30) )


# %%
PCA_data.columns = ("PC"+str(i) for i in range(1,8))
print("Shape of the Feature Matrix after PCA is:", PCA_data.shape)
print("PVE of the chosen PC's are:", pc_analyser.calc_PVE(m=7))
PCA_data = pd.concat([clean_Y, PCA_data], axis=1)

# %% [markdown]
# ### SMOTE

# %%
minority = PCA_data[PCA_data["Bankrupt?"] == 1] # Extract minority samples from data
smt = smote.Smote( minority.to_numpy() ) # Initialize the SMOTE class
oversamples = smt.oversample(N=2600) # Employ SMOTE oversampling


# %%
smote_data = PCA_data.copy(deep=True) # Cleared from outliers and dim reduced by PCA. Now oversample
oversamples_pd = pd.DataFrame(oversamples, columns = PCA_data.columns)
smote_data = smote_data.append(oversamples_pd)
smote_data = smote_data.reset_index(drop=True)


# %%
unstable_smote = (smote_data["Bankrupt?"] == 1).sum()
stable_smote = (smote_data["Bankrupt?"] == 0).sum()
print("Oversampled Data Size:", smote_data.shape[0])
print("Number of Stable Companies:", stable_smote)
print("Number of Unstable Companies (with SMOTE):", unstable_smote)
print("unstable to Stable Ratio: ", unstable_smote/stable_smote, sep="")


# %%
smote_data["Bankrupt?"].hist()
plt.show()

# %% [markdown]
# ### Test Train Split

# %%
test_ratio = 0.1
#Smote
smote_data = smote_data.sample(frac=1).reset_index(drop=True)
train_sm, test_sm = fcn.test_train_split(smote_data, test_ratio )
X_train_sm = train_sm.drop(["Bankrupt?"], axis=1)
Y_train_sm = train_sm["Bankrupt?"]
X_test_sm = test_sm.drop(["Bankrupt?"], axis=1)
Y_test_sm = test_sm["Bankrupt?"]
#No Smote
train, test = fcn.test_train_split(PCA_data, test_ratio )
X_train = train.drop(["Bankrupt?"], axis=1)
Y_train = train["Bankrupt?"]
X_test = test.drop(["Bankrupt?"], axis=1)
Y_test = test["Bankrupt?"]

X_train_np = X_train.to_numpy()
Y_train_np = Y_train.to_numpy()
X_test_np = X_test.to_numpy()
Y_test_np = Y_test.to_numpy()
X_train_sm_np = X_train_sm.to_numpy()
Y_train_sm_np = Y_train_sm.to_numpy()
X_test_sm_np = X_test_sm.to_numpy()
Y_test_sm_np = Y_test_sm.to_numpy()

# %% [markdown]
# ### k-Nearest Neighbors Classifier (with and without SMOTE)

# %%
knn_classifier = kNN.k_NN_classifier(X_train_np, Y_train_np )
knn_preds = np.zeros_like(Y_test_np)
for idx, test in enumerate(X_test_np):
    knn_preds[idx] = knn_classifier.classify(test)

knn_classifier_sm = kNN.k_NN_classifier(X_train_sm_np, Y_train_sm_np )
knn_sm_preds = np.zeros_like(Y_test_sm_np)
for idx, test in enumerate(X_test_sm_np):
    knn_sm_preds[idx] = knn_classifier_sm.classify(test)


# %%
print("Confusion Matrix Without SMOTE")
conf_matrix = fcn.confusion_matrix(Y_test_np, knn_preds, ret = True)

print("Recall:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0]) * 100 )
print("Precision:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]) * 100 )
print()
print("Confusion Matrix With SMOTE")
conf_matrix_sm = fcn.confusion_matrix(Y_test_sm_np, knn_sm_preds, ret = True)
print("Recall:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[1,0]) * 100 )
print("Precision:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[0,1]) *100 )

# %% [markdown]
# ### Logistic Regression

# %%
log_reg = Logistic_regression(7)
initial_weights = np.random.rand(7,1)
weights = Logistic_regression.gradient_descent(X_train_sm_np, Y_train_sm_np.reshape(len(Y_train_sm_np),1), initial_weights)
for idx in range(10000):
    weights = Logistic_regression.gradient_descent(X_train_sm_np, Y_train_sm_np.reshape(len(Y_train_sm_np),1), weights)
final_pred =Logistic_regression.pred(X_test_sm_np, weights)
result = Logistic_regression.classify(final_pred)


# %%
conf_matrix_sm_lr = fcn.confusion_matrix(Y_test_sm_np.T, result.T*1, ret = True)
print("Recall:", conf_matrix_sm_lr[0,0]/(conf_matrix_sm_lr[0,0]+conf_matrix_sm_lr[1,0]) * 100 )
print("Precision:", conf_matrix_sm_lr[0,0]/(conf_matrix_sm_lr[0,0]+conf_matrix_sm_lr[0,1]) *100 )

# %% [markdown]
# ### Gradient Boosting Classifier

# %%
grad_boost_sm = BinaryGradientBoostClassify(6, 0.065, 5, 1e-5, 17)
grad_boost_sm.fit(X_train_sm_np, Y_train_sm_np)
y_pred_sm = grad_boost_sm.predict(X_test_sm_np)
y_pred = grad_boost_sm.predict(X_test_np)


# %%
print("on SMOTE Data")
conf_matrix_sm = fcn.confusion_matrix(Y_test_sm_np, y_pred_sm, show=True)
print("Recall:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[1,0]) * 100 )
print("Precision:", conf_matrix_sm[0,0]/(conf_matrix_sm[0,0]+conf_matrix_sm[0,1]) *100 )
print("\non Real Data")
conf_matrix = fcn.confusion_matrix(Y_test_np, y_pred, show = True)
print("Recall:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[1,0]) * 100 )
print("Precision:", conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1]) *100 )


