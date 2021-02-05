import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash.dependencies
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# ---------------------------------------------------------------
# csv file call
DF_RFM = pd.read_csv("RFM_data.csv")
dfb = pd.read_csv("bird-window-collision-death.csv")

# # Data exploration with plotly
# # ---------------------------------------
fig = px.scatter(DF_RFM, x="CustomerID", y="frequency",
                 size="CustomerID", color="monetary_value", hover_name="recency",
                 log_x=True, size_max=60)

app.layout = html.Div([

    dbc.Container([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Page 1", href="#")),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("More pages", header=True, ),
                        dbc.DropdownMenuItem("Page 2", href="#"),
                        dbc.DropdownMenuItem("Page 3", href="#"),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="More",
                ),

            ],
            brand="NavbarSimple",
            brand_href="#",
            color="primary",
            dark=True,

        ),
        # NavbarSimple Complete
        dbc.Row([
            dbc.Col(html.H1("Stock Market DashBord", className='text-center text-primary , mb-4'), width=12),

        ]),
        dbc.Row([

            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Total customer", className="card-title"),
                    ])

                ], style={"width": "21rem", 'textAlign': 'center', }, ),

            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Recency", className="card-title center"),

                    ])

                ], style={"width": "21rem", 'textAlign': 'center', 'color': 'success'}, ),

            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Frequency", className="card-title "),

                    ])

                ], style={"width": "21rem", 'textAlign': 'center'}, ),

            ]),

        ],

        ),
        # First Row complete

        dbc.Row([
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Recency", className="card-title, "),
                        dcc.Dropdown(
                            id='Recency', multi=False, value=1,
                            options=[{'label': x, 'value': x}
                                     for x in sorted(DF_RFM['recency'].unique())]

                        ),

                    ])

                ], style={"width": "21rem", 'textAlign': 'center'}, )
            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Frequency", className="card-title, "),
                        dcc.Dropdown(
                            id='Frequency', multi=False, value=1,
                            options=[{'label': x, 'value': x}
                                     for x in sorted(DF_RFM['recency'].unique())]

                        ),

                    ])

                ], style={"width": "21rem", 'textAlign': 'center'}, ),

            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Monetary value", className="card-title, "),
                        dcc.Dropdown(
                            id='Monetary_value', multi=False, value=1,
                            options=[{'label': x, 'value': x}
                                     for x in sorted(DF_RFM['recency'].unique())]

                        ),

                    ])

                ], style={"width": "21rem", 'textAlign': 'center'}, ),

            ]),

        ], ),
        # Second Row complete

        dbc.Row([
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Recency Out", className="card-title, "),
                        dcc.Graph(id='line-fig1', figure={}),

                    ])

                ], style={'textAlign': 'center'}, )
            ]),

        ], ),
        dbc.Row([

            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("frequency Out", className="card-title, "),
                        dcc.Graph(
                            id='life-exp-vs-gdp',
                            figure=fig
                        )

                    ])

                ], style={'textAlign': 'center'}, )
            ]),

        ], ),

    ])

])


# Callback section: connecting the components
# ************************************************************************

# Line chart - Recency out
@app.callback(
    Output('line-fig1', 'figure'),
    Input('Recency', 'value')
)
def update_graph(Recency):
    dff = DF_RFM[DF_RFM['recency'] == Recency]
    figln = px.line(dff, x='CustomerID', y='frequency')
    return figln


# if __name__ == '__main__':
app.run_server(debug=True)
