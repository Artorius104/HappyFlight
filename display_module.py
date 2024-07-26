import plotly.graph_objs as go
from pyspark.sql import functions as F


def satisfaction_pie_chart(df, filtered):
    # Collect data from PySpark DataFrame
    data = df.collect()
    labels = [row['satisfaction'] for row in data]
    values = [row['percentage'] for row in data]

    # Create the pie chart
    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=['red', 'green'])
        )]
    )
    if filtered:
        fig.update_layout(
            title="Répartition de la satisfaction (entre 20 et 60 ans)",
            annotations=[dict(text='Satisfaction', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    else:
        fig.update_layout(
            title="Répartition de la satisfaction",
            annotations=[dict(text='Satisfaction', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    return fig

def customer_type_pie_chart(df, filtered):
    # Collect data from PySpark DataFrame
    data = df.collect()
    labels = [row['Customer Type'] for row in data]
    values = [row['percentage'] for row in data]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=['green', 'red'])
        )]
    )
    if filtered is True:
        fig.update_layout(
            title="Répartition des types de clients (entre 20 et 60 ans)",
            annotations=[dict(text='Customer Type', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    else:
        fig.update_layout(
            title="Répartition des types de clients",
            annotations=[dict(text='Customer Type', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    return fig

def age_distribution_chart(df, filtered):
    data = df.select("Age").rdd.flatMap(lambda x: x).collect()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Distribution de l\'âge',
        marker_color='blue',
        opacity=0.75
    ))
    if filtered is True:
        fig.update_layout(
            title="Distribution de l'âge des clients (entre 20 et 60 ans)",
            xaxis_title="Âge",
            yaxis_title="Nombre de clients",
            bargap=0.2,
            bargroupgap=0.1
        )
    else:
        fig.update_layout(
            title="Distribution de l'âge des clients",
            xaxis_title="Âge",
            yaxis_title="Nombre de clients",
            bargap=0.2,
            bargroupgap=0.1
        )
    return fig

def plot_correlation_heatmap(df):
    # Convert PySpark DataFrame to a list of rows
    data = df.collect()

    # Extract unique column names
    cols = sorted(set(row['col1'] for row in data).union(set(row['col2'] for row in data)))

    # Create a dictionary to hold correlation values
    correlation_dict = {(row['col1'], row['col2']): row['correlation'] for row in data}
    correlation_dict.update({(row['col2'], row['col1']): row['correlation'] for row in data})

    # Create a 2D list for correlation matrix values
    correlation_matrix = [[correlation_dict.get((col1, col2), 0) for col2 in cols] for col1 in cols]

    # Créer la carte thermique
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=cols,
        y=cols,
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
    data = df.collect()
    labels = [row['Type of Travel'] for row in data]
    values = [row['percentage'] for row in data]

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=['yellow', 'skyblue'])
        )]
    )
    fig.update_layout(
        title="Répartition types de voyage (clients loyaux)",
        annotations=[dict(text='Type of Travel', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def travel_type_satisfaction_bar_chart(df):
    # Collect data from PySpark DataFrame
    data = df.collect()

    # Extract unique values for Type of Travel and Satisfaction
    types_of_travel = sorted(set(row['Type of Travel'] for row in data))
    satisfaction_levels = sorted(set(row['Satisfaction'] for row in data))

    # Prepare data for the bar chart
    chart_data = {satisfaction: {travel: 0 for travel in types_of_travel} for satisfaction in satisfaction_levels}
    for row in data:
        chart_data[row['Satisfaction']][row['Type of Travel']] = row['Count']

    traces = []
    for satisfaction in satisfaction_levels:
        traces.append(
            go.Bar(
                x=types_of_travel,
                y=[chart_data[satisfaction][travel] for travel in types_of_travel],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )

        # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='group',
        title="Satisfaction par type de voyage (clients loyaux)",
        xaxis_title="Type de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )
    return fig

def business_class_satisfaction_bar_chart(df):
    # Collect data from PySpark DataFrame
    data = df.collect()

    # Extract unique values for Class and Satisfaction
    classes = sorted(set(row['Class'] for row in data))
    satisfaction_levels = sorted(set(row['satisfaction'] for row in data))

    # Prepare data for the bar chart
    chart_data = {satisfaction: {cls: 0 for cls in classes} for satisfaction in satisfaction_levels}
    for row in data:
        chart_data[row['satisfaction']][row['Class']] = row['Count']

    # Create traces for the bar chart
    traces = []
    for satisfaction in satisfaction_levels:
        traces.append(
            go.Bar(
                x=classes,
                y=[chart_data[satisfaction][cls] for cls in classes],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )

    # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='stack',
        title="Satisfaction pour les clients qui voyagent pour affaires (clients loyaux)",
        xaxis_title="Classe de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )
    return fig


def personal_class_satisfaction_bar_chart(df):
    # Collect data from PySpark DataFrame
    data = df.collect()

    # Extract unique values for Class and Satisfaction
    classes = sorted(set(row['Class'] for row in data))
    satisfaction_levels = sorted(set(row['satisfaction'] for row in data))

    # Prepare data for the bar chart
    chart_data = {satisfaction: {cls: 0 for cls in classes} for satisfaction in satisfaction_levels}
    for row in data:
        chart_data[row['satisfaction']][row['Class']] = row['Count']

    # Create traces for the bar chart
    traces = []
    for satisfaction in satisfaction_levels:
        traces.append(
            go.Bar(
                x=classes,
                y=[chart_data[satisfaction][cls] for cls in classes],
                name=satisfaction,
                marker_color='green' if satisfaction == 'satisfied' else 'red'
            )
        )

    # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='stack',
        title="Satisfaction pour les clients qui voyagent pour autre chose (clients loyaux)",
        xaxis_title="Classe de voyage",
        yaxis_title="Nombre de clients",
        legend_title="Satisfaction"
    )
    return fig


def per_services_satisfaction_bar_chart(df, classe):
    # Collect data from PySpark DataFrame
    data = df.collect()

    # Extract unique satisfaction levels
    satisfaction_levels = sorted(set(row['Satisfaction'] for row in data))

    # Prepare traces for the bar chart
    traces = []
    for satisfaction in satisfaction_levels:
        # Filter data for the current satisfaction level
        subset_data = [(row['Parameter'], row['Average_Score']) for row in data if row['Satisfaction'] == satisfaction]
        parameters, scores = zip(*subset_data)

        # Add bar trace to the figure
        traces.append(
            go.Bar(
                x=parameters,
                y=scores,
                name=satisfaction,
                marker_color='green' if satisfaction == 'Satisfied' else 'red'
            )
        )

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout based on the class parameter
    if classe == "eco":
        fig.update_layout(
            title="Moyenne des notes par paramètre pour les clients non-loyaux en Eco",
            xaxis_title="Paramètre",
            yaxis_title="Note moyenne",
            barmode='group',
            xaxis_tickangle=-75
        )
    elif classe == "business":
        fig.update_layout(
            title="Moyenne des notes par paramètre pour les clients non-loyaux en Business",
            xaxis_title="Paramètre",
            yaxis_title="Note moyenne",
            barmode='group',
            xaxis_tickangle=-75
        )
    else:
        fig.update_layout(
            title="Moyenne des notes par paramètre pour les clients loyaux",
            xaxis_title="Paramètre",
            yaxis_title="Note moyenne",
            barmode='group',
            xaxis_tickangle=-75
        )

    return fig


def flight_distance_histogram(df, classe):
    # Collect data from PySpark DataFrame
    data = df.select('Flight Distance').rdd.flatMap(lambda x: x).collect()

    # Create histogram
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=60,  # Number of bins for the histogram
            name='Distance de vol',
            marker_color='blue',
            opacity=0.75
        )
    )

    # Add KDE (Kernel Density Estimate) trace
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=60,
            histnorm='probability density',
            name='KDE',
            marker_color='red',
            opacity=0.5
        )
    )

    # Update layout based on the class parameter
    if classe == "eco":
        fig.update_layout(
            title="Distribution des distances de vols des clients non-loyaux en Eco",
            xaxis_title="Distance de vol (km)",
            yaxis_title="Nombre de vols",
            barmode='overlay',
            legend_title="Type de données"
        )
    elif classe == "business":
        fig.update_layout(
            title="Distribution des distances de vols des clients non-loyaux en Business",
            xaxis_title="Distance de vol (km)",
            yaxis_title="Nombre de vols",
            barmode='overlay',
            legend_title="Type de données"
        )
    else:
        fig.update_layout(
            title="Distribution des distances de vols des clients loyaux",
            xaxis_title="Distance de vol (km)",
            yaxis_title="Nombre de vols",
            barmode='overlay',
            legend_title="Type de données"
        )

    return fig


def flight_distance_satisfaction_histogram(df, loyal):
    # Collect data from PySpark DataFrame
    data = df.collect()

    # Extract values for the plot
    distance_bins = [row['Distance Bin'] for row in data]
    percentages = [row['Percentage'] for row in data]

    # Create the figure
    fig = go.Figure()

    # Add scatter plot trace
    fig.add_trace(
        go.Scatter(
            x=distance_bins,
            y=percentages,
            mode='lines+markers',
            name='Satisfaction',
            line=dict(color='orange'),
            marker=dict(size=8)
        )
    )

    # Update layout based on the loyalty parameter
    if loyal:
        fig.update_layout(
            title="Satisfaction des clients loyaux selon la distance de vol",
            xaxis_title="Catégorie de distance de vol (km)",
            yaxis_title="Pourcentage de satisfaits",
            xaxis_tickangle=-45
        )
    else:
        fig.update_layout(
            title="Satisfaction des clients non-loyaux selon la distance de vol",
            xaxis_title="Catégorie de distance de vol (km)",
            yaxis_title="Pourcentage de satisfaits",
            xaxis_tickangle=-45
        )

    # Add annotations for each point
    for i in range(len(distance_bins)):
        fig.add_annotation(
            x=distance_bins[i],
            y=percentages[i],
            text=f'{percentages[i]:.1f}%',
            showarrow=False,
            font=dict(size=12, color='black'),
            align='center'
        )

    return fig


def services_comparison_graphs(df):
    fig_list = []
    params = [
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink", "Gate location",
        "Inflight wifi service", "Inflight entertainment", "Online support", "Ease of Online booking",
        "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
        "Online boarding"
    ]

    for param in params:
        # Préparer les colonnes pour la transformation
        satisfied_col = f"{param}_satisfied"
        dissatisfied_col = f"{param}_dissatisfied"

        # Sélectionner les colonnes et transformer en format long
        df_long = df.select(
            F.col("Distance Bin"),
            F.expr(f"'{param}_satisfied' AS Parameter"),
            F.col(satisfied_col).alias("Average_Score"),
            F.lit('satisfied').alias("Satisfaction")
        ).union(
            df.select(
                F.col("Distance Bin"),
                F.expr(f"'{param}_dissatisfied' AS Parameter"),
                F.col(dissatisfied_col).alias("Average_Score"),
                F.lit('dissatisfied').alias("Satisfaction")
            )
        )

        # Convertir en Pandas DataFrame pour Plotly
        df_pandas = df_long.toPandas()

        # Créer le graphique
        fig = go.Figure()

        for satisfaction in ['satisfied', 'dissatisfied']:
            subset = df_pandas[df_pandas['Satisfaction'] == satisfaction]
            fig.add_trace(
                go.Bar(
                    x=subset['Distance Bin'],
                    y=subset['Average_Score'],
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

        fig_list.append(fig)

    return fig_list