import dash_bootstrap_components as dbc
from dash import html, dcc
from pandas import read_csv
from utils_app import create_services_comparison_layout
from display_module import (
    travel_type_pie_chart, travel_type_satisfaction_bar_chart, business_class_satisfaction_bar_chart,
    personal_class_satisfaction_bar_chart, per_services_satisfaction_bar_chart, flight_distance_histogram,
    flight_distance_satisfaction_histogram, services_comparison_graphs)


# GET ANALYSES FIGURES
local_path = "csv/"
travel_type_fig = travel_type_pie_chart(read_csv(f'{local_path}travel_type_distribution_loyal.csv'))
travel_type_satisfaction_fig = travel_type_satisfaction_bar_chart(read_csv(f'{local_path}travel_type_satisfaction_distribution_loyal.csv'))
business_class_satisfaction_fig = business_class_satisfaction_bar_chart(read_csv(f'{local_path}business_class_satisfaction_distribution_loyal.csv'))
personal_class_satisfaction_fig = personal_class_satisfaction_bar_chart(read_csv(f'{local_path}personal_class_satisfaction_distribution.csv'))
per_services_satisfaction_fig = per_services_satisfaction_bar_chart(read_csv(f'{local_path}services_satisfaction_loyal.csv'))
flight_distance_fig = flight_distance_histogram(read_csv(f'{local_path}flight_distribution_loyal.csv'))
flight_distance_satisfaction_fig = flight_distance_satisfaction_histogram(read_csv(f'{local_path}satisfaction_per_distance_loyal.csv'))
services_comparison_fig_list = services_comparison_graphs(read_csv(f'{local_path}services_satisfaction_per_distance_loyal.csv'))
services_comparison_layout = create_services_comparison_layout(services_comparison_fig_list)

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
                    dcc.Graph(figure=travel_type_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=travel_type_satisfaction_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=business_class_satisfaction_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=personal_class_satisfaction_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(figure=per_services_satisfaction_fig),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=12
            ),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=flight_distance_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=flight_distance_satisfaction_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        html.Div(services_comparison_layout)
    ],
    fluid=True,
)
