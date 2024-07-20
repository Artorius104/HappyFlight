import plotly.graph_objs as go

def generate_pie_chart(df):
    fig = go.Figure(data=[go.Pie(
        labels=df['satisfaction'],
        values=df['percentage'],
        hole=0.3,
        textinfo='label+percent',
        marker=dict(colors=['red', 'green'])
    )])
    fig.update_layout(
        title_text="Répartition de la satisfaction en pourcentage",
        title_x=0.5
    )
    return fig