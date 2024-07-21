from dash import Dash, html, Output, Input, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# RUN ANALYSES
from data_processing import get_spark_analyses
get_spark_analyses()

# Import des layouts
from layouts.resume import layout_resume
from layouts.loyal import layout_loyal
from layouts.non_loyal import layout_non_loyal

# APP INITIALISATION
app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
app.title = "HF Dashboard"


# BARRE DE NAVIGATION
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Résumé", href="/resume")),
        dbc.NavItem(dbc.NavLink("Clients Loyaux", href="/loyaux")),
        dbc.NavItem(dbc.NavLink("Clients Non Loyaux", href="/non-loyaux")),
    ],
    brand="Happy Flight",
    brand_href="/",
    color="primary",
    dark=True,
)

# FOOTER
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                "© 2024 - Happy Flight",
                className="text-center"
            )
        )
    ),
    fluid=True,
)

# LAYOUT
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        navbar,
        html.Div(id="page-content"),
        footer,
    ]
)

# CALLBACKS
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/resume":
        return layout_resume
    elif pathname == "/loyaux":
        return layout_loyal
    elif pathname == "/non-loyaux":
        return layout_non_loyal
    else:
        return layout_resume  # Layout par défaut

# LANCEMENT
if __name__ == "__main__":
    app.run(debug=True, dev_tools_hot_reload=True)
