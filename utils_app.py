import dash_bootstrap_components as dbc
from dash import dcc, html

def create_services_comparison_layout(fig_list):
    # Liste pour stocker les rangées
    rows = []

    # Itérer sur les figures deux par deux
    for i in range(0, len(fig_list), 2):
        # Créer les colonnes pour la rangée actuelle
        cols = []

        # Ajouter la première figure
        cols.append(dbc.Col(
            dcc.Graph(figure=fig_list[i]),
            style={"box-shadow": "10px 5px 5px #C1C6CF"},
            width=6
        ))
        # Ajouter la deuxième figure si elle existe
        if i + 1 < len(fig_list):
            cols.append(dbc.Col(
                dcc.Graph(figure=fig_list[i + 1]),
                style={"box-shadow": "10px 5px 5px #C1C6CF"},
                width=6
            ))
        # Ajouter la rangée complète à la liste des rangées
        rows.append(
            dbc.Row(
                cols,
                className="mb-4",
            )
        )

    return rows