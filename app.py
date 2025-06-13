# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:46:48 2025

@author: edvin
"""

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px

    
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('House Pricing Data')
    ],style={
    "background-image": 'url("https://dystewilliams.com/wp-content/uploads/2020/07/iStock-1181134074-neighborhood-1536x864.jpg")',
    "background-size": "cover",
    "height": "100vh"}, 
        
        )

    
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)