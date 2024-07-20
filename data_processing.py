from pyspark.sql import SparkSession
import pandas as pd


hdfs_address = "hdfs://localhost:9000/"
local_path = "csv/"

# FONCTIONS UTILITAIRES SPARK
def load_data(spark):
    # Lire le fichier CSV depuis HDFS
    df = spark.read.csv(
        f'{hdfs_address}airline_data/Airline_customer_satisfaction.csv',
        header=True,
        inferSchema=True
    )
    print("ZER GOOD")

    return df

def global_satisfaction(df):
    satisfaction_dist = df.groupBy("satisfaction").count().toPandas()
    total_count = satisfaction_dist['count'].sum()
    satisfaction_dist['percentage'] = (satisfaction_dist['count'] / total_count) * 100
    satisfaction_dist.to_csv(f'{local_path}satisfaction_distribution.csv', index=False)

def global_client_type_distrib(df):
    customer_type_dist = df.groupBy("Customer Type").count().toPandas()
    total_count = customer_type_dist['count'].sum()
    customer_type_dist['percentage'] = (customer_type_dist['count'] / total_count) * 100
    customer_type_dist.to_csv(f'{local_path}satisfaction_distribution.csv', index=False)

def age_distrib(df):
    age_dist = df.select("Age").toPandas()
    age_dist.to_csv(f'{local_path}satisfaction_distribution.csv', index=False)

def age_filtered_distrib(df_age_filtered):
    age_dist_filtered = df_age_filtered.select("Age").toPandas()

def filtered_satisfaction(df_age_filtered):
    filtered_satisfaction_dist = df_age_filtered.groupBy("satisfaction").count().toPandas()
    total_count = filtered_satisfaction_dist['count'].sum()
    filtered_satisfaction_dist['percentage'] = (filtered_satisfaction_dist['count'] / total_count) * 100

def filtered_client_type_distrib(df_age_filtered):
    filtered_customer_type_dist = df_age_filtered.groupBy("Customer Type").count().toPandas()
    total_count = filtered_customer_type_dist['count'].sum()
    filtered_customer_type_dist['percentage'] = (filtered_customer_type_dist['count'] / total_count) * 100

def correlation_matrix(df):
    numeric_cols = [
        'Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
        'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment',
        'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    df_pd = df.select(numeric_cols).toPandas()
    correlation_matrix = df_pd.corr()

def travel_type_distrib(df, loyal):
    if loyal is True:
        df_filtered = df.filter(df["Customer Type"] == "Loyal Customer")
    else:
        df_filtered = df.filter(df["Customer Type"] == "disloyal Customer")

    travel_type_dist = df_filtered.groupBy("Type of Travel").count().toPandas()
    total_count = travel_type_dist['count'].sum()
    travel_type_dist['percentage'] = (travel_type_dist['count'] / total_count) * 100

def travel_type_satisfaction(df, loyal):
    travel_satisfaction_dist = df.groupBy("Type of Travel", "satisfaction").count().toPandas()
    travel_satisfaction_dist.columns = ['Type of Travel', 'Satisfaction', 'Count']

def business_satisfaction_per_class(df, loyal):
    filtered_df = df.filter(df["Type of Travel"] == "Business travel")
    class_satisfaction_distribution = filtered_df.groupBy("Class", "satisfaction").count().toPandas()

def other_satisfaction_per_class(df, loyal):
    filtered_df = df.filter(df["Type of Travel"] == "Personal Travel")
    class_satisfaction_distribution = filtered_df.groupBy("Class", "satisfaction").count().toPandas()

def given_notes_per_services(df, loyal, classe):
    # Séparer les clients satisfaits et insatisfaits
    df_satisfied = df.filter(df["satisfaction"] == "satisfied")
    df_dissatisfied = df.filter(df["satisfaction"] == "dissatisfied")

    # Paramètres à analyser
    parameters = ["Seat comfort", "Departure/Arrival time convenient", "Food and drink",
                  "Gate location", "Inflight wifi service", "Inflight entertainment",
                  "Online support", "Ease of Online booking", "On-board service",
                  "Leg room service", "Baggage handling", "Checkin service",
                  "Cleanliness", "Online boarding"]

    # Calculer la moyenne des notes pour chaque paramètre pour les clients satisfaits
    satisfied_means = df_satisfied.select(parameters).groupBy().mean().toPandas().transpose()
    satisfied_means.columns = ['Satisfied']
    satisfied_means['Parameter'] = satisfied_means.index
    satisfied_means['Parameter'] = satisfied_means['Parameter'].str.replace("avg(", "").str.replace(")", "")

    # Calculer la moyenne des notes pour chaque paramètre pour les clients insatisfaits
    dissatisfied_means = df_dissatisfied.select(parameters).groupBy().mean().toPandas().transpose()
    dissatisfied_means.columns = ['Dissatisfied']
    dissatisfied_means['Parameter'] = dissatisfied_means.index
    dissatisfied_means['Parameter'] = dissatisfied_means['Parameter'].str.replace("avg(", "").str.replace(")", "")

    # Combiner les moyennes dans un seul DataFrame
    means_df = pd.merge(satisfied_means, dissatisfied_means, on='Parameter')
    means_melted = means_df.melt(
        id_vars='Parameter',
        value_vars=['Satisfied', 'Dissatisfied'],
        var_name='Satisfaction',
        value_name='Average Score'
    )

def flight_distrib(df, loyal, classe):
    distance_dist = df.select("Flight Distance").toPandas()

def satisfaction_per_distance(df, loyal):
    # On garde la distance de vol et la satisfaction
    df_pd = df.select("Flight Distance", "satisfaction").toPandas()

    # Supprimer les lignes avec des valeurs manquantes
    df_pd.dropna(subset=["Flight Distance", "satisfaction"], inplace=True)

    # Créer des bins pour l'histogramme
    if loyal is True:
        bins = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3500, 4000, 5000, 6000, 7000, 8000]
        labels = ['0-300', '300-600', '600-900', '900-1200', '1200-1500', '1500-1800',
                  '1800-2100', '2100-2400', '2400-2700', '2700-3000', '3000-3500',
                  '3500-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000']
    else:
        bins = [1100, 1400, 1700, 2000, 2300, 2600, 3000]
        labels = ['1100-1400', '1400-1700', '1700-2000', '2000-2300', '2300-2600', '2600-3000']

    # Ajouter une colonne pour les bins de distance de vol
    df_pd['Distance Bin'] = pd.cut(df_pd['Flight Distance'], bins=bins, labels=labels)

    # Calculer le nombre de clients satisfaits et total dans chaque bin
    satisfaction_counts = df_pd.groupby(['Distance Bin', 'satisfaction']).size().unstack(fill_value=0)
    total_counts = satisfaction_counts.sum(axis=1)

    # Calculer le pourcentage de clients satisfaits dans chaque bin
    satisfaction_percentage = (satisfaction_counts['satisfied'] / total_counts) * 100

    # Convertir en DataFrame Pandas pour la visualisation
    satisfaction_percentage = satisfaction_percentage.reset_index()

def satisfaction_per_distance_per_param(df, loyal):
    # Convertir en DataFrame Pandas pour faciliter la manipulation
    df_pd = df.select(
        "Flight Distance", "satisfaction", "Seat comfort", "Departure/Arrival time convenient",
        "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness", "Online boarding").toPandas()

    # Supprimer les lignes avec des valeurs manquantes
    df_pd.dropna(subset=["Flight Distance", "satisfaction"], inplace=True)

    # Créer des bins pour l'histogramme
    if loyal is True:
        bins = [
            0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400,
            2700, 3000, 3500, 4000, 5000, 6000, 7000, 8000
        ]
        labels = [
            '0-300', '300-600', '600-900', '900-1200', '1200-1500', '1500-1800',
            '1800-2100', '2100-2400', '2400-2700', '2700-3000', '3000-3500',
            '3500-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000'
        ]
    else:
        bins = [1100, 1400, 1700, 2000, 2300, 2600, 3000]
        labels = ['1100-1400', '1400-1700', '1700-2000', '2000-2300', '2300-2600', '2600-3000']

    df_pd['Distance Bin'] = pd.cut(
        df_pd['Flight Distance'],
        bins=bins,
        labels=labels,
        right=False
    )

    # Séparer les clients satisfaits et insatisfaits
    df_satisfied_pd = df_pd[df_pd['satisfaction'] == 'satisfied']
    df_dissatisfied_pd = df_pd[df_pd['satisfaction'] == 'dissatisfied']

    # Calculer les notes moyennes pour chaque paramètre et chaque bin
    params = [
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink", "Gate location",
        "Inflight wifi service", "Inflight entertainment", "Online support", "Ease of Online booking",
        "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
        "Online boarding"
    ]

    avg_satisfied = df_satisfied_pd.groupby('Distance Bin')[params].mean().reset_index()
    avg_dissatisfied = df_dissatisfied_pd.groupby('Distance Bin')[params].mean().reset_index()
    # Fusionner les deux DataFrames pour comparaison
    avg_scores = pd.merge(
        avg_satisfied,
        avg_dissatisfied,
        on='Distance Bin',
        suffixes=('_satisfied', '_dissatisfied')
    )

# MAIN FUNCTION
def get_spark_analyses():
    spark = SparkSession.builder \
        .appName("Airline Satisfaction Analysis") \
        .getOrCreate()
    # Charger les données du HDFS
    df = load_data(spark)

    # Distribution de la satisfaction
    global_satisfaction(df)
    # Distribution des types de clients
    global_client_type_distrib(df)
    # Distribution de l'âge
    age_distrib(df)

    # Distribution de la majorité entre 20 ans et 60 ans
    df_age_filtered = df.filter((df.Age >= 20) & (df.Age <= 60))
    age_filtered_distrib(df_age_filtered)
    # Répartition de la satisfaction (filtrée)
    filtered_satisfaction(df_age_filtered)
    # Répartition des types de clients (filtrée)
    filtered_client_type_distrib(df_age_filtered)
    # Matrice de Corrélation
    correlation_matrix(df)

    # CLIENTS LOYAUX
    travel_type_distrib(df_age_filtered, True)
    df_loyal = df_age_filtered.filter(df_age_filtered["Customer Type"] == "Loyal Customer")
    travel_type_satisfaction(df_loyal, True)
    business_satisfaction_per_class(df_loyal, True)
    other_satisfaction_per_class(df_loyal, True)

    df_loyal_eco = df_loyal.filter(df_loyal["Class"] == "Eco")
    given_notes_per_services(df_loyal_eco, True, "eco")
    flight_distrib(df_loyal_eco, True, "eco")
    satisfaction_per_distance(df_loyal_eco, True)
    satisfaction_per_distance_per_param(df_loyal_eco, True)

    # CLIENTS NON LOYAUX
    travel_type_distrib(df_age_filtered, False)
    df_disloyal = df_age_filtered.filter(df_age_filtered["Customer Type"] == "disloyal Customer")
    travel_type_satisfaction(df_disloyal, False)
    business_satisfaction_per_class(df_disloyal, False)

    df_disloyal_eco = df_disloyal.filter(
        (df_disloyal["Type of Travel"] == "Business travel") &
        (df_disloyal["Class"] == "Eco")
    )
    df_disloyal_busi = df_disloyal.filter(
        (df_disloyal["Type of Travel"] == "Business travel") &
        (df_disloyal["Class"] == "Business")
    )
    given_notes_per_services(df_disloyal_eco, False, "eco")
    given_notes_per_services(df_disloyal_busi, False, "business")
    flight_distrib(df_disloyal_eco, False, "eco")
    flight_distrib(df_disloyal_busi, False, "business")

    df_disloyal_distance = df_disloyal_busi.filter(
        (df['Flight Distance'] >= 1100) &
        (df['Flight Distance'] <= 3000)
    )
    satisfaction_per_distance(df_disloyal_distance, False)
    satisfaction_per_distance_per_param(df_disloyal_distance, False)

    # Arrêter la session Spark
    spark.stop()
    # return satisfaction_dist
