import dash_bootstrap_components as dbc
from dash import html, dcc

from utils_data import get_loyal_figures


# GET ANALYSES FIGURES
local_path = "csv/"
fig = get_loyal_figures(local_path)

# ANALYSES CLIENTS LOYAUX
layout_loyal = dbc.Container(
    [
        html.H1(
            "Clients Loyaux",
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
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=fig["personal_class_satisfaction"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(figure=fig["per_services_satisfaction"]),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=12
            ),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=fig["flight_distance"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=fig["flight_distance_satisfaction"]),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        html.Div(fig["services_comparison"])
    ],
    fluid=True,
)
