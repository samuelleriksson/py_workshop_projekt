#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 16:20:39 2025

@author: samuel
"""

from sklearn.datasets import fetch_openml
import pandas as pd

# Hämta datasetet från OpenML
housing = fetch_openml(name="house_prices", as_frame=True)

X = housing.data
y = housing.target

df = X.copy()
df["SalePrice"] = pd.to_numeric(y)

#%%

import matplotlib.pyplot as plt

plt.scatter(df["GrLivArea"], df["SalePrice"], alpha = 0.4, c = "green")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()

#%%

import missingno as msno

msno.matrix(X)
plt.title("Missing Values Plot")
plt.show()

#%%

missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_percent[missing_percent > 0]

missing_percent.head(20).plot(kind="barh", figsize=(10, 8))
plt.xlabel("Missing Percentage")
plt.title("Top Features by Missing Data")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%%

df_clean = df.copy().drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])

missing_percent = (df_clean.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_percent[missing_percent > 0]

missing_percent.head(20).plot(kind="barh", figsize=(10, 8))
plt.xlabel("Missing Percentage")
plt.title("Top Features by Missing Data")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
df_clean

#%%

msno.matrix(df_clean)
plt.title("Missing Values Plot")
plt.show()

#%%

from sklearn.preprocessing import LabelEncoder


X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == "category" or X_encoded[col].dtype == object:
        # Handle categorical variables
        X_encoded[col] = X_encoded[col].astype(str).fillna("Missing")
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])
    else:
        # Handle numeric variables with missing values
        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

#%%

msno.matrix(X_encoded)
plt.title("Missing Values Plot aftere preprocessing")
plt.show()


#%%




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


