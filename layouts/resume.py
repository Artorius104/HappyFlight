import dash_bootstrap_components as dbc
from dash import html, dcc
from pandas import read_csv
from display_module import (satisfaction_pie_chart, customer_type_pie_chart, age_distribution_chart, plot_correlation_heatmap)


# GET ANALYSES FIGURES
local_path = "csv/"
satisfaction_fig = satisfaction_pie_chart(read_csv(f'{local_path}satisfaction_distribution.csv'), False)
customer_type_fig = customer_type_pie_chart(read_csv(f'{local_path}client_type_distribution.csv'), False)
age_fig = age_distribution_chart(read_csv(f'{local_path}age_distribution.csv'), False)
age_filtered_fig = age_distribution_chart(read_csv(f'{local_path}age_distribution_filtered.csv'), True)
satisfaction_filtered_fig = satisfaction_pie_chart(read_csv(f'{local_path}filtered_satisfaction_distribution.csv'), True)
customer_type_filtered_fig = customer_type_pie_chart(read_csv(f'{local_path}filtered_customer_type_distribution.csv'), True)
corr_matrix_fig = plot_correlation_heatmap(read_csv(f'{local_path}correlation_matrix.csv'))

# PAGE DU RESUME
layout_resume = dbc.Container(
    [
        html.H1(
            "Résumé",
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
                    dcc.Graph(figure=satisfaction_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=customer_type_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=age_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=age_filtered_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=satisfaction_filtered_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(figure=customer_type_filtered_fig),
                    style={"box-shadow": "10px 5px 5px #C1C6CF"},
                    width=6
                ),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(figure=corr_matrix_fig),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=12
            ),
            className="mb-4",
        ),
    ],
    fluid=True,
)
