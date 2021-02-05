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

DF_RFM = pd.read_csv("RFM_data.csv")

# # Data exploration with plotly
# # ---------------------------------------
# # print(dfd.Genre.nunique())
# # print(dfd.Genre.unique())
#
#
# # Data Visualization with plotly
# # ---------------------------------------
# # fig = px.pie(data_frame=dfd, values='Japan Sales', names='Genre',  title='Population of European continent')
# # fig.show()
#
# #
# # fig = px.histogram(data_frame=dfd, x='Japan Sales', y='Genre',  title='Population of European continent')
# # fig.show()
# #
# # fig = px.bar(data_frame=dfd, x='Japan Sales', y='Genre',  title='Population of European continent')
# # fig.show()
# #
# # Data Interactive with plotly
# # ---------------------------------------
print(DF_RFM)
#
# DF_RFM = pd.read_csv('RFM_data.csv', encoding='utf-8')
# DF_RFM['recency_normalized'] = pd.qcut(DF_RFM['recency'], 50, labels=False)
# DF_RFM['recency_normalized'] = DF_RFM['recency_normalized'] + 1
# DF_RFM['frequency_normalized'] = pd.qcut(DF_RFM['frequency'], 50, labels=False)
# DF_RFM['frequency_normalized'] = DF_RFM['frequency_normalized'] + 1
# DF_RFM['monetary_value_normalized'] = pd.qcut(DF_RFM['monetary_value'], 50, labels=False)
# DF_RFM['monetary_value_normalized'] = DF_RFM['monetary_value_normalized'] + 1
# print(DF_RFM)
#
# DF_ARRAY = np.array(DF_RFM.iloc[:, 4:8])  # Getting only the numeric features from the dataset
# DF_NORM = preprocessing.normalize(DF_ARRAY)  # Normalizing the data
# print(DF_ARRAY)
# print(DF_NORM)
#
# # Creating our Model
# kmeans = KMeans(n_clusters=10)
#
# # Training our model
# kmeans.fit(DF_NORM)
# # Amount of values to be tested for K
# Ks = range(2, 30)
#
# # List to hold on the metrics for each value of K
# results = []
#
# # Executing the loop
# for K in Ks:
#     model = KMeans(n_clusters=K)
#     model.fit(DF_NORM)
#
#     results.append(model.inertia_)

# Plotting the final result
# plt.plot(Ks, results, 'o-')
# plt.xlabel("Values of K")
# plt.ylabel("SSE")
# plt.show()

# # Creating our Model
# kmeans = KMeans(n_clusters=15)
#
# # Training our model
# kmeans.fit(DF_NORM)
#
# # You can see the labels (clusters) assigned for each data point with the function labels_
# kmeans.labels_
#
# # Assigning the labels to the initial dataset
# DF_RFM['cluster'] = kmeans.labels_
# print(DF_RFM)
# PLOT = go.Figure()
#
# PLOT.add_trace(go.Scatter3d(x=DF_RFM['recency_normalized'],
#                             y=DF_RFM['frequency_normalized'],
#                             z=DF_RFM['monetary_value_normalized']
#                             )
#                )
# PLOT = go.Figure()
#
# for C in list(DF_RFM.cluster.unique()):
#     PLOT.add_trace(go.Scatter3d(x=DF_RFM[DF_RFM.cluster == C]['recency_normalized'],
#                                 y=DF_RFM[DF_RFM.cluster == C]['frequency_normalized'],
#                                 z=DF_RFM[DF_RFM.cluster == C]['monetary_value_normalized'],
#                                 mode='markers', marker_size=8, marker_line_width=1,
#                                 name='RFM Segment ' + str(C)))
# DF_RFM[DF_RFM.cluster==6]
# cluster_energy=DF_RFM.groupby(by='cluster').sum('recency')
#
# cluster_energy['strength']=cluster_energy['frequency_normalized']+cluster_energy['monetary_value_normalized']-cluster_energy['recency_normalized']
# cluster_energy
#
# cluster_energy.sort_values(['strength'], ascending=[False], inplace=True)
#
# cluster_energy.sort_values('strength',ascending=False,inplace=True)
# Layout section DashBord
# ---------------------------------------------------


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
                        html.H4("Recency", className="card-title"),
                        dcc.Dropdown(
                            id='DropDown1', multi=False, value=1,
                            options=[{'label': x, 'value': x}
                                     for x in sorted(DF_RFM['recency'].unique())]

                        ),

                    ])

                ], style={"width": "18rem"}, )
            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Customer", className="card-title"),
                        dcc.Dropdown(
                            id='DropDown2', multi=False, value=1,
                            options=[{'label': x, 'value': x}
                                     for x in sorted(DF_RFM['CustomerID'].unique())]

                        ),

                    ])

                ], style={"width": "18rem"}, )

            ]),

        ],

        ),
        # First Row complete

        dbc.Row([

            dbc.Col([


                dbc.Card([

                    dbc.CardBody([
                        html.H4("Out put", className="card-title"),

                        dcc.Graph(id='line-fig1', figure={}),
                    ])

                ],
                    style={"width": "32rem"})
            ]),
            dbc.Col([
                dbc.Card([

                    dbc.CardBody([
                        html.H4("Out put", className="card-title"),

                        dcc.Graph(id='line-fig2', figure={}),
                    ])

                ],
                    style={"width": "32rem"})

            ]),

        ],

        ),

    ])

])


# Callback section: connecting the components
# ************************************************************************
# Line chart - Single
@app.callback(
    Output('line-fig1', 'figure'),
    Input('DropDown1', 'value')
)
def update_graph(DropDown1):
    dff = DF_RFM[DF_RFM['recency'] == DropDown1]
    figln = px.line(dff, x='CustomerID', y='frequency')
    return figln


# Line chart - multiple
@app.callback(
    Output('line-fig2', 'figure'),
    Input('DropDown2', 'value')
)
def update_graph(DropDown2):
    dff = DF_RFM[DF_RFM['CustomerID'] == DropDown2]
    figln = px.line(dff, x='recency', y='monetary_value')
    return figln


# # Histogram
# @app.callback(
#     Output('my-hist', 'figure'),
#     Input('my-checklist', 'value')
# )
# def update_graph(stock_slctd):
#     dff = DF_RFM[DF_RFM['frequency'].isin(stock_slctd)]
#     dff = dff[dff['Date'] == '2020-12-03']
#     fighist = px.histogram(dff, x='Symbols', y='Close')
#     return fighist


if __name__ == '__main__':
    app.run_server(debug=True, port=4050)
