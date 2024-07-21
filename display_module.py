import plotly.graph_objs as go
import plotly.express as px

def satisfaction_pie_chart(df):
    fig = go.Figure(
        data=[go.Pie(
            labels=df['satisfaction'],
            values=df['percentage'],
            hole=.3,
            marker=dict(colors=['red', 'green'])
        )]
    )

    fig.update_layout(
        title="Répartition de la satisfaction en pourcentage",
        annotations=[dict(text='Satisfaction', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def customer_type_pie_chart(df):
    fig = go.Figure(
        data=[go.Pie(
            labels=df['Customer Type'],
            values=df['percentage'],
            hole=.3,
            marker=dict(colors=['green', 'red'])
        )]
    )
    fig.update_layout(
        title="Répartition des types de clients",
        annotations=[dict(text='Customer Type', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def age_distribution_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Age'],
        nbinsx=30,
        name='Distribution de l\'âge',
        marker_color='blue',
        opacity=0.75
    ))

    fig.update_layout(
        title="Distribution de l'âge des clients",
        xaxis_title="Âge",
        yaxis_title="Nombre de clients",
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig

def age_distribution_filtered_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['Age'],
        nbinsx=30,
        name='Distribution de l\'âge (20-60 ans)',
        marker_color='green',
        opacity=0.75
    ))
    fig.update_layout(
        title="Distribution de l'âge des clients (entre 20 et 60 ans)",
        xaxis_title="Âge",
        yaxis_title="Nombre de clients",
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig

def plot_correlation_heatmap(df_pd):
    # Calculer la matrice de corrélation
    correlation_matrix = df_pd.corr()

    # Créer la carte thermique
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title='Corrélation'),
        zmin=-1,  # Valeur minimale pour les couleurs
        zmax=1    # Valeur maximale pour les couleurs
    ))

    fig.update_layout(
        title='Matrice de Corrélation',
        xaxis_title='Variables',
        yaxis_title='Variables',
        xaxis=dict(ticks='', side='top'),
        yaxis=dict(ticks=''),
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    return fig

def travel_type_pie_chart(df):
    fig = go.Figure(
        data=[go.Pie(
            labels=df['Type of Travel'],
            values=df['percentage'],
            hole=.3,
            marker=dict(colors=['yellow', 'skyblue'])
        )]
    )
    fig.update_layout(
        title="Répartition des types de voyage parmi les clients loyaux",
        annotations=[dict(text='Type of Travel', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def travel_type_satisfaction_bar_chart(df):
    types_of_travel = df['Type of Travel'].unique()
    satisfaction_levels = df['Satisfaction'].unique()

    data = []

    for satisfaction in satisfaction_levels:
        filtered_df = df[df['Satisfaction'] == satisfaction]
        data.append(
            go.Bar(
                x=filtered_df['Type of Travel'],
                y=filtered_df['Count'],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title="Satisfaction des clients loyaux pour chaque type de voyage",
        xaxis_title="Type de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )
    return fig

def business_class_satisfaction_bar_chart(df):
    classes = df['Class'].unique()
    satisfaction_levels = df['satisfaction'].unique()

    data = []

    for satisfaction in satisfaction_levels:
        filtered_df = df[df['satisfaction'] == satisfaction]
        data.append(
            go.Bar(
                x=filtered_df['Class'],
                y=filtered_df['count'],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        title="Répartition des classes de voyage et satisfaction pour les clients loyaux et qui voyagent pour affaires",
        xaxis_title="Classe de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )

    return fig

def personal_class_satisfaction_bar_chart(df):
    classes = df['Class'].unique()
    satisfaction_levels = df['satisfaction'].unique()

    data = []

    for satisfaction in satisfaction_levels:
        filtered_df = df[df['satisfaction'] == satisfaction]
        data.append(
            go.Bar(
                x=filtered_df['Class'],
                y=filtered_df['count'],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='stack',
        title="Répartition des classes de voyage et satisfaction pour les clients loyaux et qui voyagent pour autre chose",
        xaxis_title="Classe de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )

    return fig

def per_services_satisfaction_bar_chart(df):
    # Préparer les données pour le graphique
    df_melted = df.melt(
        id_vars='Parameter',
        value_vars=['Satisfied', 'Dissatisfied'],
        var_name='Satisfaction',
        value_name='Average Score'
    )

    # Créer le graphique en barres
    fig = go.Figure()

    for satisfaction in df_melted['Satisfaction'].unique():
        df_subset = df_melted[df_melted['Satisfaction'] == satisfaction]
        fig.add_trace(
            go.Bar(
                x=df_subset['Parameter'],
                y=df_subset['Average Score'],
                name=satisfaction,
                marker_color='green' if satisfaction == 'Satisfied' else 'red'
            )
        )

    fig.update_layout(
        title="Moyenne des notes par paramètre pour les clients satisfaits et insatisfaits",
        xaxis_title="Paramètre",
        yaxis_title="Note moyenne",
        barmode='group',
        xaxis_tickangle=-75
    )

    return fig


def flight_distance_histogram(df):
    # Créer l'histogramme de la distribution des distances de vols
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df['Flight Distance'],
            nbinsx=60,  # Nombre de bins pour l'histogramme
            name='Distance de vol',
            marker_color='blue',
            opacity=0.75
        )
    )

    # Ajouter une ligne KDE (Kernel Density Estimate)
    fig.add_trace(
        go.Histogram(
            x=df['Flight Distance'],
            nbinsx=60,
            histnorm='probability density',
            name='KDE',
            marker_color='red',
            opacity=0.5
        )
    )

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title="Distribution des distances de vols des clients loyaux",
        xaxis_title="Distance de vol (km)",
        yaxis_title="Nombre de vols",
        barmode='overlay',
        legend_title="Type de données"
    )

    return fig

def flight_distance_satisfaction_histogram(df):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Distance Bin'],
            y=df['Percentage'],
            mode='lines+markers',
            name='Satisfaction',
            line=dict(color='orange'),
            marker=dict(size=8)
        )
    )

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        title="Satisfaction en pourcentage des clients loyaux selon la distance de vol",
        xaxis_title="Catégorie de distance de vol (km)",
        yaxis_title="Pourcentage de satisfaits",
        xaxis_tickangle=-45
    )

    # Ajouter les pourcentages au-dessus de chaque point
    for i in range(len(df)):
        fig.add_annotation(
            x=df['Distance Bin'][i],
            y=df['Percentage'][i],
            text=f'{df['Percentage'][i]:.1f}%',
            showarrow=False,
            font=dict(size=12, color='black'),
            align='center'
        )

    return fig

def services_comparison_graphs(avg_scores):
    fig_dicts = []
    params = [
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink", "Gate location",
        "Inflight wifi service", "Inflight entertainment", "Online support", "Ease of Online booking",
        "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
        "Online boarding"
    ]

    for param in params:
        # Préparer les données pour le graphique
        avg_scores_melted = avg_scores.melt(
            id_vars='Distance Bin',
            value_vars=[f'{param}_satisfied', f'{param}_dissatisfied'],
            var_name='Satisfaction',
            value_name='Average Score'
        )

        # Créer le graphique
        fig = go.Figure()

        for satisfaction in ['satisfied', 'dissatisfied']:
            subset = avg_scores_melted[avg_scores_melted['Satisfaction'] == satisfaction]
            fig.add_trace(
                go.Bar(
                    x=subset['Distance Bin'],
                    y=subset['Average Score'],
                    name=satisfaction.capitalize(),
                    marker_color='green' if satisfaction == 'satisfied' else 'red'
                )
            )

        fig.update_layout(
            title=f"Moyenne des notes pour {param} selon la distance de vol",
            xaxis_title="Catégorie de distance de vol (km)",
            yaxis_title="Note moyenne",
            xaxis_tickangle=-45,
            barmode='group',
            legend_title='Satisfaction'
        )

        # Ajouter les moyennes au-dessus des barres
        for trace in fig.data:
            for i, value in enumerate(trace.y):
                fig.add_annotation(
                    x=trace.x[i],
                    y=value,
                    text=f'{value:.2f}',
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    align='center'
                )

        fig_dicts.append(fig)

    return fig_dicts