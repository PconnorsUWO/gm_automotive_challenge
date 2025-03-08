
## Block 1: Importing libraries
# Standard imports
import numpy as np
from itertools import chain, combinations

# Data manipulation
import pandas as pd
import polars as pl

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector

# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

## Block 2: Data loading
# Load the data using polars
df = pl.read_csv("data_clean.parquet")

# Calculate the descriptive statistics
df.describe()

# Calculate X and y vectors
X = df.drop(["target"])
y = df.select("target")

## Block 3: Backward feature selection
# Define the logistic regression model
model = LogisticRegression(max_iter=1000)

# Define the sequential feature selector
sfs = SequentialFeatureSelector(model, n_features_to_select=1, direction='backward')

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Fit the sequential feature selector
sfs.fit(X_train, y_train)

## Block 4: Calculate bootstrapped AUC over the test set
# Define the number of bootstraps
n_bootstraps = 1000

# Define the list to store the AUC values
auc_values = []

# Loop over the number of bootstraps
for i in range(n_bootstraps):
    # Sample with replacement from the test set
    idx = np.random.choice(range(len(y_test)), len(y_test), replace=True)
    X_test_bootstrap = X_test.iloc[idx, :]
    y_test_bootstrap = y_test.iloc[idx]
    
    # Predict the probabilities
    y_pred = sfs.estimator_.predict_proba(X_test_bootstrap)[:, 1]
    
    # Calculate the AUC
    auc = roc_auc_score(y_test_bootstrap, y_pred)
    
    # Append the AUC to the list
    auc_values.append(auc)

## Block 5: Calculate the 95% confidence interval
# Calculate the test AUC
test_auc = roc_auc_score(y_test, sfs.estimator_.predict_proba(X_test)[:, 1])

# Get the residuals of the bootstrapped AUCs
residuals = np.array(auc_values) - test_auc

# Calculate the 95% confidence interval centered around the test AUC
ci = np.percentile(residuals, [2.5, 97.5]) + test_auc

# Print the confidence interval
print(f"The 95% confidence interval for the AUC is {ci}")

## Block 6: Decoy block. This block simulates the dataset is a regression and trains a LASSO model on it.
# Define the LASSO model
model = Lasso(alpha=0.1)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Fit the model
model.fit(X_train, y_train)


## Block 7: Decoy block. This block calculates the R^2 of the LASSO model.
# Calculate the R^2
r2 = model.score(X_test, y_test)

# Print the R^2
print(f"The R^2 of the LASSO model is {r2}")






