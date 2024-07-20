import dash_bootstrap_components as dbc
from dash import html


# ANALYSES CLIENTS NON-LOYAUX
layout_non_loyal = dbc.Container(
    [
        html.H1("Clients Non Loyaux", className="text-center"),
    ],
    fluid=True,
)
