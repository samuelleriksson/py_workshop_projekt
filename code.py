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

for col in X_encoded.drop(columns = "Id"):
    
    X = X_train[[col]]  # make it 2D
    model = LinearRegression().fit(X, y_train)
    y_pred = model.predict(X_test[[col]])
    r2 = r2_score(y_test,y_pred)
    r2_scores[col] = r2

# Create and sort DataFrame
r2_df = pd.DataFrame.from_dict(r2_scores, orient="index", columns=["R^2"])
r2_df = r2_df.sort_values(by="R^2", ascending=False)

import plotly.express as px

# Plot top 20 features
r2_plot_df = r2_df.head(20).reset_index().rename(columns={"index": "Feature"})

r2_fig = px.bar(
    r2_plot_df,
    x="R^2",
    y="Feature",
    orientation="h",
    title="Top 20 Features by R² Score",
    labels={"Feature": "Feature", "R^2": "R² Score"}
)


# Reverse y-axis so best scores appear at top
r2_fig.update_layout(
    height = 500,
    yaxis=dict(autorange="reversed"),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='black')
)


print(r2_df.head(20))



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

X_train = X_train.drop(columns=["Id"])
X_test = X_test.drop(columns=["Id"])

    
#%%

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("RandomForest")
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")
print("RMSLE:", rmsle)
print("########################################")

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

model = LinearRegression().fit(X_train[["OverallQual"]],y_train)
y_pred = model.predict(X_test[["OverallQual"]])

mse = mean_squared_error(y_test, y_pred)

print("LinearRegression (OverallQual)")
print("R²:", r2_score(y_test, y_pred))
print(f"MSE: {mse:,.2f}")
print("########################################")





#%%

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators = 100, random_state = 42).fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("XGBRGradientBoosting")
print("R²:", r2_score(y_test, y_pred))
print("RMSLE:", rmsle)
print(f"MSE: {mse:,.2f}")
print("########################################")


#%%

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# 1. Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# 2. Set parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,            # try 0.01–0.1
    "max_depth": 5,                   # try 4–10
    "subsample": 0.7,                 # try 0.6–1.0
    "colsample_bytree": 0.7,          # try 0.5–1.0
    "gamma": 0.1,                     # add if overfitting
    "reg_alpha": 1,                   # L1 regularization
    "reg_lambda": 1,                  # L2 regularization
    "seed": 42
}

# 3. Use cross-validation to find best num_boost_round
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=30,
    metrics="rmse",
    seed=42,
    verbose_eval=False
)

best_rounds = len(cv_results)
print("Best num_boost_round:", best_rounds)

# 4. Train final model with optimal rounds
model = xgb.train(
    params,
    dtrain,
    num_boost_round=best_rounds,
    evals=[(dvalid, "eval")],
    early_stopping_rounds=30,
    verbose_eval=False
)

# 5. Evaluate
y_pred = model.predict(dvalid)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))


print("XGBRBoost")
print("R²:", r2)
print("RMSLE:", rmsle)
print(f"MSE: {mse:,.2f}")

#%%

xgb.plot_importance(model, max_num_features=20)
plt.tight_layout()
plt.show()
#%%

import io
import base64
import matplotlib.pyplot as plt
import xgboost as xgb

# Plot top 20 features by importance
fig, ax = plt.subplots()
xgb.plot_importance(model, max_num_features=20, ax=ax)
plt.tight_layout()

# Save as transparent image to buffer
buf = io.BytesIO()
plt.savefig(buf, format="png", transparent=True)
plt.close(fig)
buf.seek(0)

# Encode to base64
importance_img_base64 = base64.b64encode(buf.read()).decode("utf-8")


#%%


cv_results["test-rmse-mean"].plot(title="CV RMSE over boosting rounds")
plt.xlabel("Boosting Round")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

import io
import base64
import matplotlib.pyplot as plt

# Create the plot
fig, ax = plt.subplots()
cv_results["test-rmse-mean"].plot(ax=ax, title="CV RMSE over Boosting Rounds")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("RMSE")
ax.grid(True)
plt.tight_layout()

# Save the plot to a buffer
buf = io.BytesIO()
plt.savefig(buf, format="png", transparent = True)
plt.close(fig)  # Close the plot to avoid displaying it outside Dash
buf.seek(0)

# Encode the image to base64
img_base64 = base64.b64encode(buf.read()).decode("utf-8")


#%%

xgb.plot_importance(model, importance_type="gain", max_num_features=20)
plt.grid(True)
plt.tight_layout()
plt.show()


#%%

import shap
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)

#%%

import shap
import matplotlib.pyplot as plt
import io
import base64

# 1. Compute SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# 2. Create SHAP beeswarm plot
fig = plt.figure()
shap.plots.beeswarm(shap_values, show=False)  # show=False prevents auto-display

# 3. Save the plot to buffer with transparent background
buf = io.BytesIO()
plt.savefig(buf, format="png", transparent=True, bbox_inches='tight')
plt.close(fig)
buf.seek(0)

# 4. Encode to base64
shap_img_base64 = base64.b64encode(buf.read()).decode("utf-8")


#%%

from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("HistGradientBoosting")
print("R²:", r2_score(y_test, y_pred))
print("RMSLE:", rmsle)
print(f"MSE: {mse:,.2f}")
print("########################################")



#%%

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

# Storage dictionaries with lists
metrics = {"R_2": [], "MSE": []}
alphas = np.logspace(-3, 3, 50)

for alpha in alphas:
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics["R_2"].append(r2)
    metrics["MSE"].append(mse)

# Plot R²
plt.plot(alphas, metrics["R_2"], label="R²")
plt.xlabel("Alpha")
plt.ylabel("R²")
plt.title("R² for different alpha values")
plt.grid(True)
plt.show()

# Plot MSE
plt.plot(alphas, metrics["MSE"], label="MSE", color="red")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("MSE for different alpha values")
plt.grid(True)
plt.show()

model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("LassoRidge")
print("Best alpha: ", model.alpha_)
print("R²:", r2_score(y_test, y_pred))
print("RMSLE:", rmsle)
print(f"MSE: {mse:,.2f}")
print("########################################")

#%%

model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("LinearRegression (multivariate)")
print("R²:", r2_score(y_test, y_pred))
print("RMSLE:", rmsle)
print(f"MSE: {mse:,.2f}")
print("########################################")


#%%

import pandas as pd

model_results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "R² Score": 0.65,
        "MSE": "2 681 026 164",
        "Notes": "Baseline model on OverallQual"
    },
    {
        "Model": "Multivariate Linear Regression",
        "R² Score": 0.85,
        "MSE": "1 171 287 710",
        "Notes": "Basline model with all predictors"
    },
    {
        "Model": "XGBR Gradient Boosting",
        "R² Score": 0.91,
        "MSE": "689 401 639",
        "Notes": "n_estimators=100, learning_rate=0.05"
    },
    {
        "Model": "Hist Gradient Boosting",
        "R² Score": 0.89,
        "MSE": "844 265 288",
        "Notes": "-"
    },
    {
        "Model": "Random Forest",
        "R² Score": 0.90,
        "MSE": "801 632 622",
        "Notes": "n_estimators=100, learning_rate=0.05"
    },
    {
        "Model": "Lasso Ridge",
        "R² Score": 0.85,
        "MSE": "1 170 244 722",
        "Notes": "alpha = 10"
    },
    {
        "Model": "Optimizes XGBR Boost",
        "R² Score": 0.92,
        "MSE": "602 496 859",
        "Notes": "RMSLE: 0.13"
    }
    
])


#%%

import dash
from dash import html, dash_table, dcc, Output, Input
from sklearn.datasets import fetch_openml
import pandas as pd
import plotly.express as px
import statsmodels.api as sm



numeric_columns = df.select_dtypes(include='number').columns

# Initialize the app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([  # <--- NEW WRAPPER

         html.H1(
             "Ames Housing Price Model",
             style={
                 "textAlign": "center",
                 "color": "black"
             }
         ),
         
         html.P(
"This dashboard provides visual analytics and modeling results for the Ames housing dataset. Firstly the data were either reformatted into useable format or removed if deemed irrelevant to the project. A number of different approaches were used to extract patterns from the data in order to predict real estate value.",
    style={
        "textAlign": "center",
        "fontSize": "18px",
        "margin": "10px auto 30px auto",
        "width": "70%",
        "color": "black"
    }
),
     
         html.Div([
             html.H2("Full Dataset Preview"),
             html.P("This table shows the first 50 rows of the Ames housing dataset.")
         ], style={
             "width": "80%",
             "margin": "0 auto",
             "color": "black"
         }),
     
         dash_table.DataTable(
             data=df.head(50).to_dict("records"),
             columns=[{"name": i, "id": i} for i in df.columns],
             page_size=50,
             style_table={
                 "overflowX": "auto",
                 "overflowY": "auto",
                 "maxHeight": "300px",
                 "width": "80%",
                 "margin": "0 auto"
             },
             style_cell={
                 "textAlign": "left",
                 "padding": "8px",
                 "minWidth": "100px",
                 "whiteSpace": "normal"
             },
             style_header={
                 "backgroundColor": "#f2f2f2",
                 "fontWeight": "bold",
                 "textAlign": "left"
             },
             style_data_conditional=[
                 {
                     "if": {"row_index": "odd"},
                     "backgroundColor": "#fafafa"
                 }
             ]
         ),
     
         html.H2(
             "Explore Variable Relationships",
             style={
                 "textAlign": "center",
                 "marginTop": "40px",
                 "color": "black"
             }
         ),
     
         html.Div([
             # Graph (left)
             dcc.Graph(id='scatter-plot', style={"width": "90%"}),
     
             # Dropdown controls (right)
             html.Div([
                 html.Label("Select X-axis variable:"),
                 dcc.Dropdown(
                     id='x-axis',
                     options=[{'label': col, 'value': col} for col in numeric_columns],
                     value ='GrLivArea'
                 ),
     
                 html.Label("Select Y-axis variable:", style={"marginTop": "20px"}),
                 dcc.Dropdown(
                     id='y-axis',
                     options=[{'label': col, 'value': col} for col in numeric_columns],
                     value='SalePrice'
                 )
             ], style={
                 "width": "20%",
                 "paddingLeft": "10px",
                 "display": "flex",
                 "flexDirection": "column",
                 "justifyContent": "center",
                 "color": "black"
             })
         ], style={
             "display": "flex",
             "justifyContent": "center",
             "alignItems": "center",
             "marginTop": "40px",
             "width": "80%",
             "marginLeft": "auto",
             "marginRight": "auto"
         }),
     
     
         html.H3(
             "Top 20 Features by R² Score",
             style={
                 "textAlign": "center",
                 "marginTop": "40px",
                 "color": "black",
             }
         ),
             
         html.Div([
         dcc.Graph(
             id="r2-barplot",
             figure=r2_fig,
             style={"width": "70%", "margin": "0 auto"}
         )
     ]),
         
         html.H3("Model Comparison", style={
        "textAlign": "center",
        "marginTop": "40px",
        "color": "black"
    }),
    
    dash_table.DataTable(
        data = model_results.sort_values(by="R² Score", ascending=False).to_dict("records"),
        columns=[{"name": i, "id": i} for i in model_results.columns],
        style_table={
            "width": "80%",
            "margin": "0 auto",
            "overflowX": "auto"
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "whiteSpace": "normal"
        },
        style_header={
            "backgroundColor": "#f2f2f2",
            "fontWeight": "bold"
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafafa"
            }
        ]
    ),
    html.Div([
    html.H2("Cross-Validation Performance", style={"textAlign": "center"}),

    html.P("This plot shows the average RMSE from 5-fold cross-validation "
           "over boosting rounds during XGBoost model training. "
           "Early stopping is applied to prevent overfitting.",
           style={"width": "70%", "margin": "auto", "textAlign": "center"}),

    html.Img(src="data:image/png;base64," + img_base64,
             style={"display": "block", "margin": "30px auto", "width": "80%", "maxWidth": "700px"})
]),
    html.Div([
    html.H2("Feature Importance (XGBoost)", style={"textAlign": "center"}),

    html.P(
        "This chart shows the top 20 features ranked by how frequently XGBoost used them to split decision trees. "
        "A higher importance score indicates the feature was used more often across boosting rounds.",
        style={"width": "70%", "margin": "auto", "textAlign": "center"}
    ),

    html.Img(
        src="data:image/png;base64," + importance_img_base64,
        style={"display": "block", "margin": "30px auto", "width": "80%", "maxWidth": "700px"}
    )
]), 
    html.Div([
    html.H2("SHAP Beeswarm Plot (Global Explanation)", style={"textAlign": "center"}),

    html.P(
        "This SHAP beeswarm plot shows how each feature influenced individual predictions. "
        "Each dot represents a single prediction. Color indicates the feature value "
        "(red = high, blue = low), and position on the x-axis shows the impact on the model's output.",
        style={"width": "70%", "margin": "auto", "textAlign": "center"}
    ),

    html.Img(
        src="data:image/png;base64," + shap_img_base64,
        style={"display": "block", "margin": "30px auto", "width": "90%", "maxWidth": "800px"}
    ),

    html.Div([
    html.H2("Conclusion", style={
        "textAlign": "center",
        "marginTop": "60px",
        "color": "black"
    }),
    
    html.P(
        "The analysis highlights the most important features influencing housing prices in Ames. "
        "Through regression modeling, cross-validation, and SHAP analysis, we gain insights into "
        "both the predictive power and interpretability of machine learning models used.",
        style={
            "textAlign": "center",
            "fontSize": "16px",
            "width": "80%",
            "margin": "20px auto",
            "color": "black"
        }
    )
])


])
        

    ], style={  # Style for the center panel
        "backgroundColor": "rgba(255, 255, 255, 0.9)",  # white with slight transparency
        "width": "70%",
        "margin": "40px auto",
        "padding": "15px",
        "borderRadius": "12px",
        "boxShadow": "0px 0px 16px rgba(0, 0, 0, 0.4)"
    })
              
                  
              
], style={  # Background image stays on page
    "backgroundImage": 'url("https://dystewilliams.com/wp-content/uploads/2020/07/iStock-1181134074-neighborhood-1536x864.jpg")',
    "backgroundRepeat": "repeat",
    "backgroundSize": "cover",
    "minHeight": "100vh",
    "paddingTop": "0",
    "marginTop": "0",
    "padding": "0",
    "margin": "0",
    "border": "none",
    "outline": "none",
    "boxSizing": "border-box"
})


# Callback for updating the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value')
)
def update_plot(x_col, y_col):
    # Create scatter plot with OLS trendline
    fig = px.scatter(
        df, x=x_col, y=y_col,
        title="",  # we'll set it below
        trendline='ols'
    )

    # Extract the regression result to get R²
    results = px.get_trendline_results(fig)
    if not results.empty:
        model = results.iloc[0]["px_fit_results"]
        r2 = model.rsquared
    else:
        r2 = None

    # Set the title with R² value
    if r2 is not None:
        fig.update_layout(
            title=f"{y_col} vs {x_col}  (R² = {r2:.3f})"
        )
    else:
        fig.update_layout(title=f"{y_col} vs {x_col}")

    # Customize layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        title_font=dict(size=20, color='black')
    )

    # Customize regression line color
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.color = 'red'

    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, open_browser=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


