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
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.layout = html.Div([

    html.Div([
        html.Div([
            html.Img(src=app.get_asset_url('corona-logo-1.jpg'),
                     id='corona-image',
                     style={
                         "height": "60px",
                         "width": "auto",
                         "margin-bottom": "25px",
                     },
                     )
        ],
            className="one-third column",
        ),
        html.Div([
            html.Div([
                html.H3("Covid - 19", style={"margin-bottom": "0px", 'color': 'white'}),
                html.H5("Track Covid - 19 Cases", style={"margin-top": "0px", 'color': 'white'}),
            ])
        ], className="one-half column", id="title"),

    ], id="header", className="row flex-display", style={"margin-bottom": "25px"}),

    # ----------Cards View Start -----------------
    html.Div([
        # ---------- Start Cards View of  CustomerID-----------------
        html.Div([
            html.H6(children='Tolale Customer',
                    style={
                        'textAlign': 'center',
                        'color': 'white'}
                    ),

            html.P(f"{DF_RFM['CustomerID'].iloc[-1]:,.0f}",
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 40}
                   ),

            html.P('new:  ' + f"{DF_RFM['CustomerID'].iloc[-1] - DF_RFM['CustomerID'].iloc[-2]:,.0f} "
                   + ' (' + str(round(((DF_RFM['CustomerID'].iloc[-1] - DF_RFM['CustomerID'].iloc[-2]) /
                                       DF_RFM['CustomerID'].iloc[-1]) * 100, 2)) + '%)',
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 15,
                       'margin-top': '-18px'}
                   )], className="card_container three columns",
        ),
        # ---------- End Cards View of  CustomerID-----------------
        # ---------- Start Cards View of  recency ----------------
        html.Div([
            html.H6(children='Recency',
                    style={
                        'textAlign': 'center',
                        'color': 'white'}
                    ),

            html.P(f"{DF_RFM['recency'].iloc[-1]:,.0f}",
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 40}
                   ),

            html.P('new:  ' + f"{DF_RFM['recency'].iloc[-1] - DF_RFM['recency'].iloc[-2]:,.0f} "
                   + ' (' + str(round(((DF_RFM['recency'].iloc[-1] - DF_RFM['recency'].iloc[-2]) /
                                       DF_RFM['recency'].iloc[-1]) * 100, 2)) + '%)',
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 15,
                       'margin-top': '-18px'}
                   )], className="card_container three columns",
        ),
        # ---------- End Cards View of  recency-----------------
        # ---------- Start Cards View of  frequency ----------------
        html.Div([
            html.H6(children='Frequency',
                    style={
                        'textAlign': 'center',
                        'color': 'white'}
                    ),

            html.P(f"{DF_RFM['recency'].iloc[-1]:,.0f}",
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 40}
                   ),

            html.P('new:  ' + f"{DF_RFM['frequency'].iloc[-1] - DF_RFM['frequency'].iloc[-2]:,.0f} "
                   + ' (' + str(round(((DF_RFM['frequency'].iloc[-1] - DF_RFM['frequency'].iloc[-2]) /
                                       DF_RFM['frequency'].iloc[-1]) * 100, 2)) + '%)',
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 15,
                       'margin-top': '-18px'}
                   )], className="card_container three columns",
        ),
        # ---------- End Cards View of  frequency-----------------
        # ---------- Start Cards View of  monetary_value ----------------
        html.Div([
            html.H6(children='Monetary value',
                    style={
                        'textAlign': 'center',
                        'color': 'white'}
                    ),

            html.P(f"{DF_RFM['monetary_value'].iloc[-1]:,.0f}",
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 40}
                   ),

            html.P('new:  ' + f"{DF_RFM['monetary_value'].iloc[-1] - DF_RFM['monetary_value'].iloc[-2]:,.0f} "
                   + ' (' + str(round(((DF_RFM['monetary_value'].iloc[-1] - DF_RFM['monetary_value'].iloc[-2]) /
                                       DF_RFM['monetary_value'].iloc[-1]) * 100, 2)) + '%)',
                   style={
                       'textAlign': 'center',
                       'color': 'orange',
                       'fontSize': 15,
                       'margin-top': '-18px'}
                   )], className="card_container three columns",
        ),
        # ---------- End Cards View of  monetary_value-----------------
    ], className="row flex-display"),
    # ----------Cards View Start with DropDown -----------------
    html.Div([
        html.Div([

            html.P('Select Recency:', className='fix_label', style={'color': 'white'}),

            dcc.Dropdown(id='w_countries',
                         multi=False,
                         clearable=True,
                         value=1,
                         placeholder='Select Recency',
                         options=[{'label': c, 'value': c}
                                  for c in sorted(DF_RFM['recency'].unique())], className='dcc_compon'),

        ], className="create_container three columns", id="cross-filter-options"),
        # ----------Graph chart Start with DropDown -----------------
        html.Div([

            dcc.Graph(id='line-fig1', figure={}),
        ], className="create_container four columns"),

        html.Div([
            dcc.Graph(id="bar-chart"),

        ], className="create_container five columns"),
        # ----------Graph chart End with DropDown -----------------

    ], className="row flex-display"),
    # ----------Cards View Ends with DropDown 1-----------------
    # ----------Cards View Start with DropDown2 -----------------
    html.Div([

        # ----------Graph chart Start with DropDown -----------------
        html.Div([
            dcc.Graph(id="scatter-plot"),
            html.P("Petal Width:"),
            dcc.RangeSlider(
                id='range-slider',
                min=0, max=2.5, step=0.1,
                marks={0: '0', 2.5: '2.5'},
                value=[0.5, 2]
            ),
            # dcc.Graph(id='line-fig2', figure={}),
        ], className="create_container four columns"),

        html.Div([
            # dcc.Graph(id="bar-chart2"),
            dcc.Graph(id="pie-chart")

        ], className="create_container five columns"),
        # ----------Graph chart End with DropDown -----------------

    ], className="row flex-display"),
    # ----------Cards View Ends with DropDown2 -----------------

    html.Div([
        html.Div([
            dcc.Graph(id="map")], className="create_container1 twelve columns"),

    ], className="row flex-display"),

], id="mainContainer",
    style={"display": "flex", "flex-direction": "column"})


# Callback section: connecting the components
# ************************************************************************
# Line chart - Single
@app.callback(
    Output('line-fig1', 'figure'),
    Input('w_countries', 'value')
)
def update_graph(w_countries):
    dff = DF_RFM[DF_RFM['recency'] == w_countries]
    figln = px.line(dff, x='CustomerID', y='monetary_value')
    return figln


# Bar chart - Single

@app.callback(
    Output("bar-chart", "figure"),
    [Input("w_countries", "value")])
def update_bar_chart(w_countries):
    mask = DF_RFM["recency"] == w_countries
    fig = px.bar(DF_RFM[mask], x="frequency", y="monetary_value",
                 color="CustomerID", barmode="group")
    return fig
# pie-chart - Single
@app.callback(
    Output("pie-chart", "figure"),
    [Input("names", "value"),
     Input("values", "value")])
def generate_chart(w_countries, values):
    fig = px.pie(DF_RFM, values=values, names=w_countries)
    return fig
# pie-chart - Single
@app.callback(
    Output("scatter-plot", "figure"),
    [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (DF_RFM['petal_width'] > low) & (DF_RFM['petal_width'] < high)
    fig = px.scatter(
        DF_RFM[mask], x="CustomerID", y="recency",
        color="frequency", size='petal_length',
        hover_data=['petal_width'])
    return fig

# # Create pie chart (total casualties)
# @app.callback(Output('pie_chart', 'figure'),
#               [Input('w_countries', 'value')])
# def update_graph(w_countries):
#     # covid_data_2 = covid_data.groupby(['date', 'Country/Region'])[['confirmed', 'death', 'recovered', 'active']].sum().reset_index()
#     new_confirmed = DF_RFM[DF_RFM['CustomerID'] == w_countries]['recency'].iloc[-1]
#     new_death = DF_RFM[DF_RFM['CustomerID'] == w_countries]['frequency'].iloc[-1]
#     new_recovered = DF_RFM[DF_RFM['CustomerID'] == w_countries]['monetary_value'].iloc[-1]
#     new_active = DF_RFM[DF_RFM['CustomerID'] == w_countries]['active'].iloc[-1]
#     colors = ['orange', '#dd1e35', 'green', '#e55467']
#
#     return {
#         'data': [go.Pie(labels=['recency', 'frequency', 'monetary_value'],
#                         values=[new_confirmed, new_death, new_recovered],
#                         marker=dict(colors=colors),
#                         hoverinfo='label+value+percent',
#                         textinfo='label+value',
#                         textfont=dict(size=13),
#                         hole=.7,
#                         rotation=45
#                         # insidetextorientation='radial',
#
#
#                         )],
#
#         'layout': go.Layout(
#             # width=800,
#             # height=520,
#             plot_bgcolor='#1f2c56',
#             paper_bgcolor='#1f2c56',
#             hovermode='closest',
#             title={
#                 'text': 'Total Cases : ' + (w_countries),
#
#
#                 'y': 0.93,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'},
#             titlefont={
#                        'color': 'white',
#                        'size': 20},
#             legend={
#                 'orientation': 'h',
#                 'bgcolor': '#1f2c56',
#                 'xanchor': 'center', 'x': 0.5, 'y': -0.07},
#             font=dict(
#                 family="sans-serif",
#                 size=12,
#                 color='white')
#             ),
#
#
#         }


if __name__ == '__main__':
    app.run_server(debug=True)
