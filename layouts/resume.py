import dash_bootstrap_components as dbc
from dash import html, dcc
from data_processing import get_global_satisfaction
from display_module import generate_pie_chart


df_satisfaction = get_global_satisfaction()
pie_chart_figure = generate_pie_chart(df_satisfaction)

# PAGE DU RESUME
layout_resume = dbc.Container(
    [
        html.H1("Résumé", className="text-center"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=pie_chart_figure), width=4),
                dbc.Col(dcc.Graph(id="graph2"), width=4),
                dbc.Col(dcc.Graph(id="graph3"), width=4),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph4"), width=4),
                dbc.Col(dcc.Graph(id="graph5"), width=4),
                dbc.Col(dcc.Graph(id="graph6"), width=4),
            ],
            className="mb-4",
        ),
        dbc.Row(
            dbc.Col(dcc.Graph(id="graph7"), width=12),
            className="mb-4",
        ),
    ],
    fluid=True,
)
