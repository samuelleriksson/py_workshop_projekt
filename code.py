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
#Sorterar data baserat på förklaraingskraft delar upp i tränings och test data. Undersäker R^2 värde och sorterar baserat på det 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

r2_results = {}

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y, test_size=0.2, random_state=42)


for col in X_train.columns:
    
    model_house_price = LinearRegression()

    model_house_price.fit(X_train[[col]], y_train)
    
    y_pred = model_house_price.predict(X_test[[col]]) 
    
    r2 = r2_score(y_test, y_pred)
    
    r2_results[col] = r2
    
    print(f'Feature: {col:20} R^2 på testdata: {r2:.4f}')

#%%
r2 = pd.DataFrame(list(r2_results.items()), columns=["Feature", "R2"]) 

    
r2_sorted = r2.sort_values(by="R2", ascending=False)    

print(r2_sorted)

#Raderar negativa R^2 värden  sparar sedan 3 nya df:s en med all parametrar som har R^2>0.1, R^2>0.2, R^2>0.3 träna en multivariable model på dessa tre och ser vilken som presterar bäst. 
#%%    
r2_clean_1 = r2[r2["R2"]>= 0.1].copy().sort_values(by="R2", ascending=False)
r2_clean_2 = r2[r2["R2"]>= 0.2].copy().sort_values(by="R2", ascending=False)   
r2_clean_3 = r2[r2["R2"]>= 0.3].copy().sort_values(by="R2", ascending=False)

print(r2_clean_1) 

#%% 
from sklearn.metrics import mean_squared_error


X_train_1 = X_train[r2_clean_1["Feature"].tolist()]
X_test_1 = X_test[r2_clean_1["Feature"].tolist()]

# Filtrera X_train och X_test baserat på r2_clean_2
X_train_2 = X_train[r2_clean_2["Feature"].tolist()]
X_test_2 = X_test[r2_clean_2["Feature"].tolist()]

# Filtrera X_train och X_test baserat på r2_clean_3
X_train_3 = X_train[r2_clean_3["Feature"].tolist()]
X_test_3 = X_test[r2_clean_3["Feature"].tolist()]

model_1 = LinearRegression()
model_1.fit(X_train_1, y_train)
y_pred_1 = model_1.predict(X_test_1)
mse_1 = mean_squared_error(y_test, y_pred_1)
r2_1 = r2_score(y_test, y_pred_1)



print(f'R^2 är {r2_1} mse är {mse_1}')
#%%
model_2 = LinearRegression()
model_2.fit(X_train_2, y_train)
y_pred_2 = model_2.predict(X_test_2)
mse_2 = mean_squared_error(y_test, y_pred_2)
r2_2 = r2_score(y_test, y_pred_2)



print(f'R^2 är {r2_2} mse är {mse_2}')
#%%
model_3 = LinearRegression()
model_3.fit(X_train_3, y_train)
y_pred_3 = model_3.predict(X_test_3)
mse_3 = mean_squared_error(y_test, y_pred_3)
r2_3 = r2_score(y_test, y_pred_3)
    
print(f'R^2 är {r2_3} mse är {mse_3}')
    
    
    
    
    
    


