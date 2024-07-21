import dash_bootstrap_components as dbc
from dash import html, dcc
from pandas import read_csv
from display_module import (
    travel_type_pie_chart, travel_type_satisfaction_bar_chart, business_class_satisfaction_bar_chart,
    personal_class_satisfaction_bar_chart, per_services_satisfaction_bar_chart, flight_distance_histogram,
    flight_distance_satisfaction_histogram, services_comparison_graphs)


# GET ANALYSES FIGURES
local_path = "csv/"
travel_type_fig = travel_type_pie_chart(read_csv(f'{local_path}travel_type_distribution_disloyal.csv'))
travel_type_satisfaction_fig = travel_type_satisfaction_bar_chart(read_csv(f'{local_path}travel_type_satisfaction_distribution_disloyal.csv'))
business_class_satisfaction_fig = business_class_satisfaction_bar_chart(read_csv(f'{local_path}business_class_satisfaction_distribution_disloyal.csv'))
per_services_satisfaction_eco_fig = per_services_satisfaction_bar_chart(read_csv(f'{local_path}services_satisfaction_disloyal_eco.csv'))
per_services_satisfaction_business_fig = per_services_satisfaction_bar_chart(read_csv(f'{local_path}services_satisfaction_disloyal_business.csv'))
flight_distance_eco_fig = flight_distance_histogram(read_csv(f'{local_path}flight_distribution_disloyal_eco.csv'))
flight_distance_business_fig = flight_distance_histogram(read_csv(f'{local_path}flight_distribution_disloyal_business.csv'))
flight_distance_satisfaction_fig = flight_distance_satisfaction_histogram(read_csv(f'{local_path}satisfaction_per_distance_disloyal.csv'))
services_comparison_fig = services_comparison_graphs(read_csv(f'{local_path}services_satisfaction_per_distance_disloyal.csv'))

# ANALYSES CLIENTS NON-LOYAUX
layout_non_loyal = dbc.Container(
    [
        html.H1("Clients Non Loyaux", className="text-center"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=travel_type_fig), width=6),
                dbc.Col(dcc.Graph(figure=travel_type_satisfaction_fig), width=6),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=business_class_satisfaction_fig), width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(figure=per_services_satisfaction_eco_fig), width=12),
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(figure=per_services_satisfaction_business_fig), width=12),
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=flight_distance_eco_fig), width=4),
                dbc.Col(dcc.Graph(figure=flight_distance_business_fig), width=4),
                dbc.Col(dcc.Graph(figure=flight_distance_satisfaction_fig), width=4),
            ],
            className="mb-4",
        ),
        dbc.Row(
            # dbc.Col(dcc.Graph(figure=services_comparison_fig), width=12),
            className="mb-4",
        ),
    ],
    fluid=True,
)
