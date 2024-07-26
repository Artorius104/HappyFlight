import dash_bootstrap_components as dbc
from dash import html, dcc

from utils_data import get_non_loyal_figures


# GET ANALYSES FIGURES
local_path = "csv/"
fig = get_non_loyal_figures(local_path)

# ANALYSES CLIENTS NON-LOYAUX
layout_non_loyal = dbc.Container(
    [
        html.H1(
            "Clients Non Loyaux",
            className="text-center",
            style={
                "font-size": "48px",
                "font-family": "Gotham, Tahoma, sans-serif",
                "color": "#158cba"
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=fig["travel_type"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=fig["travel_type_satisfaction"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=fig["business_class_satisfaction"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=12
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(figure=fig["per_services_satisfaction_eco"]),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=12
            ),
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(figure=fig["per_services_satisfaction_business"]),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=12
            ),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=fig["flight_distance_eco"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=4
                ),
                dbc.Col(
                    dcc.Graph(figure=fig["flight_distance_business"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=4
                ),
                dbc.Col(
                    dcc.Graph(figure=fig["flight_distance_satisfaction"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=4
                ),
            ],
            className="mb-4",
        ),
        html.Div(fig["services_comparison"])
    ],
    fluid=True,
)
