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
df["SalePrice"] = y



#%%

import matplotlib.pyplot as plt

plt.scatter(X["GrLivArea"], y, alpha = 0.4, c = "green")
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

missing_percent = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
missing_percent[missing_percent > 0]

missing_percent.head(20).plot(kind="barh", figsize=(10, 8))
plt.xlabel("Missing Percentage")
plt.title("Top Features by Missing Data")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%%

X_clean = X.copy().drop(columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"])

missing_percent = (X_clean.isnull().sum() / len(X) * 100).sort_values(ascending=False)
missing_percent[missing_percent > 0]

missing_percent.head(20).plot(kind="barh", figsize=(10, 8))
plt.xlabel("Missing Percentage")
plt.title("Top Features by Missing Data")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#%%

msno.matrix(X_clean)
plt.title("Missing Values Plot")
plt.show()

#%%

from sklearn.preprocessing import LabelEncoder


X_encoded = X_clean

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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded,y, test_size=0.2, random_state=42)


#%%

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


r2_scores = {}

for col in X_encoded:
    
    X = X_train[[col]]  # make it 2D
    model = LinearRegression().fit(X, y_train)
    y_pred = model.predict(X_test[[col]])
    r2 = r2_score(y_test,y_pred)
    r2_scores[col] = r2

# Create and sort DataFrame
r2_df = pd.DataFrame.from_dict(r2_scores, orient="index", columns=["R^2"])
r2_df = r2_df.sort_values(by="R^2", ascending=False)

print(r2_df.head(15))

#%%


r2_df.head(20).plot(kind="barh", figsize=(10, 8))
plt.xlabel("R^2")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#%%


for feature in r2_df.head(5).index:
    
    model = LinearRegression().fit(X_train[[feature]], y_train)
    y_pred = model.predict(X_test[[feature]])
    plt.scatter(X_test[feature], y_test, alpha = 0.4, c = "green")
    plt.plot(X_test[feature], y_pred, c = "red")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.show()
    
#%%

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")

#%%

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Pick the first tree in the forest
estimator = model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=X_train.columns, filled=True, max_depth=3, fontsize=10)
plt.title("Visualizing One Tree from the Random Forest")
plt.show()


#%%

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators = 100, random_state = 42).fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")

#%%

from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")
#%%
    
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")

#%%

import dash
from dash import html
from dash import dash_table


# Initialize the app
app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("Ames Housing Price Model", style={'textAlign': 'center'}),
    
    html.Div([
        html.H2("Full Dataset Preview"),
        html.P("This table shows the raw version of the Ames housing dataset, including 1460 data entries with 80 varibles and SalePrice")
    ], style={'width': '80%', 'margin': '0 auto'}),
    
    dash_table.DataTable(
    data=df.to_dict("records"),
    columns=[{"name": i, "id": i} for i in df.columns],
    
    # Layout and scrolling
    style_table={
        'overflowX': 'auto',         # Enables horizontal scrolling
        'overflowY': 'auto',         # Enables vertical scrolling
        'maxHeight': '300px',        # Limits the table height (adds vertical scroll if needed)
        'width': '80%',              # Table takes up 80% of the page width
        'margin': '0 auto'           # Centers the table horizontally
    },

    # Pagination
    page_size=50,  # Shows 50 rows per page with pagination controls

    # Cell styling
    style_cell={
        'textAlign': 'left',         # Left-aligns text in all cells
        'padding': '8px',            # Adds padding inside cells
        'minWidth': '100px',         # Ensures all columns have enough space
        'whiteSpace': 'normal'       # Allows text to wrap (instead of overflowing)
    },

    # Header styling
    style_header={
        'backgroundColor': '#f2f2f2',  # Light gray header background
        'fontWeight': 'bold',         # Bold text in headers
        'textAlign': 'left'           # Left-align headers
    },

    # Alternating row colors
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#fafafa'  # Light gray background for odd rows
        }
    ]
)
])

# Run the server
if __name__ == '__main__':
    app.run(debug=True, open_browser=True, use_reloader=False)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


