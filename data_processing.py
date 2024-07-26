from os import path

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import col, round, sum, count, mean, when

from utils_data import create_folder, load_data_from_hdfs, save_dataframe


def satisfaction_distrib(df: DataFrame, filtered: bool, local_path: str) -> None:
    """
    Process data to get satisfaction distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        filtered (bool): If the dataframe has been filtered.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # Grouper par la colonne 'satisfaction'
    satisfaction_dist = (
        df.groupBy("satisfaction")
        .agg(count("*").alias("count"))
    )
    # Compter le nombre d'occurences
    total_count = satisfaction_dist.agg(sum("count")).collect()[0][0]
    # Ajouter une colonne pour les pourcentages
    satisfaction_dist = satisfaction_dist.withColumn(
        "percentage",
        round((col("count") / total_count) * 100, 2)
    )

    if filtered is not True:
        save_dataframe(satisfaction_dist, f'{local_path}satisfaction_distribution')
    else:
        save_dataframe(satisfaction_dist, f'{local_path}filtered_satisfaction_distribution')

def client_type_distrib(df: DataFrame, filtered: bool, local_path: str):
    """
    Process data to get client type distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        filtered (bool): If the dataframe has been filtered.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # Grouper par la colonne 'Customer Type'
    customer_type_dist = (
        df.groupBy("Customer Type")
        .agg(count("*").alias("count"))
    )
    # Compter le nombre d'occurences
    total_count = customer_type_dist.agg(sum("count")).collect()[0][0]
    # Ajouter une colonne pour les pourcentages
    customer_type_dist = customer_type_dist.withColumn(
        "percentage",
        round((col("count") / total_count) * 100, 2)
    )

    if filtered is not True:
        save_dataframe(customer_type_dist, f'{local_path}client_type_distribution')
    else:
        save_dataframe(customer_type_dist, f'{local_path}filtered_customer_type_distribution')

def age_distrib(df: DataFrame, filtered: bool, local_path: str):
    """
    Process data to get age distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        filtered (bool): If the dataframe has been filtered.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    age_dist = df.select("Age")
    # Calculer la distribution d'âge (compter les occurrences de chaque âge)
    # age_distribution = age_dist.groupBy("Age").count()

    if filtered is not True:
        save_dataframe(age_dist, f'{local_path}age_distribution')
    else:
        save_dataframe(age_dist, f'{local_path}age_distribution_filtered')

def global_mean_notes(df, rating_columns, local_path):
    mean_ratings = df[rating_columns].mean()
    mean_ratings_sorted = mean_ratings.sort_values(ascending=True)
    mean_ratings_sorted.to_csv(f'{local_path}global_means_notes.csv', index=False)
    print(f'{local_path}global_means_notes.csv LOADED')

def global_ecart_type(df, rating_columns, local_path):
    std_devs = df[rating_columns].std()
    std_devs.to_csv(f'{local_path}ecart_type.csv', index=False)
    print(f'{local_path}ecart_type.csv LOADED')

def global_variance(df, rating_columns, local_path):
    std_devs = df[rating_columns].var()
    std_devs.to_csv(f'{local_path}variance.csv', index=False)
    print(f'{local_path}variance.csv LOADED')

def correlation_matrix(df: DataFrame, numeric_cols: list[str], local_path: str, spark: SparkSession):
    """
    Process global data to get correlation matrix into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        numeric_cols (list[str]): Columns with numeric values.
        local_path (str): The path to save the CSV file.
        spark (SparkSession): The Spark session.

    Returns:
        None
    """
    # Créer une matrice de corrélation en PySpark
    correlation_matrix = {}
    for col1 in numeric_cols:
        correlation_matrix[col1] = {}
        for col2 in numeric_cols:
            corr_value = df.stat.corr(col1, col2)
            correlation_matrix[col1][col2] = corr_value

    # Convertir la matrice de corrélations en liste de Row
    rows = []
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            corr_value = correlation_matrix[col1][col2]
            rows.append(Row(col1=col1, col2=col2, correlation=corr_value))

    # Créer un DataFrame PySpark à partir de la liste de Row
    correlation_df = spark.createDataFrame(rows)
    # # Convertir la matrice de corrélation en DataFrame Pandas pour la visualisation
    # correlation_df = pd.DataFrame(correlation_matrix)
    save_dataframe(correlation_df, f'{local_path}correlation_matrix')

def travel_type_distrib(df, loyal, local_path):
    """
    Process data to get the Travel Type distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    if loyal is True:
        df_filtered = df.filter(df["Customer Type"] == "Loyal Customer")
    else:
        df_filtered = df.filter(df["Customer Type"] == "disloyal Customer")

    # Grouper par la colonne 'Type of Travel'
    travel_type_dist = (
        df.groupBy("Type of Travel")
        .agg(count("*").alias("count"))
    )
    # Compter le nombre d'occurences
    total_count = travel_type_dist.agg(sum("count")).collect()[0][0]
    # Ajouter une colonne pour les pourcentages
    travel_type_dist = travel_type_dist.withColumn(
        "percentage",
        round((col("count") / total_count) * 100, 2)
    )

    if loyal is True:
        save_dataframe(travel_type_dist, f'{local_path}travel_type_distribution_loyal')
    else:
        save_dataframe(travel_type_dist, f'{local_path}travel_type_distribution_disloyal')

def travel_type_satisfaction(df, loyal, local_path):
    """
    Process data to get the Travel Type satisfaction distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # Grouper par la colonne 'Type of Travel'
    travel_satisfaction_dist = (
        df.groupBy("Type of Travel", "satisfaction")
        .agg(count("*").alias("Count"))
    )
    # Renommer la colonne 'satisfaction'
    travel_satisfaction_dist = (
        travel_satisfaction_dist.withColumnRenamed("satisfaction", "Satisfaction")
    )

    if loyal is True:
        save_dataframe(travel_satisfaction_dist, f'{local_path}travel_type_satisfaction_distribution_loyal')
    else:
        save_dataframe(travel_satisfaction_dist, f'{local_path}travel_type_satisfaction_distribution_disloyal')

def business_satisfaction_per_class(df, loyal, local_path):
    """
    Process data to get the satisfaction per class for business travel into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # Filtrer les données pour "Business travel"
    filtered_df = df.filter(df["Type of Travel"] == "Business travel")

    # Regrouper par "Class" et "satisfaction", et compter les occurrences
    class_satisfaction_distribution = (
        filtered_df.groupBy("Class", "satisfaction")
        .agg(count("*").alias("Count"))
    )

    if loyal is True:
        save_dataframe(class_satisfaction_distribution, f'{local_path}business_class_satisfaction_distribution_loyal')
    else:
        save_dataframe(class_satisfaction_distribution, f'{local_path}business_class_satisfaction_distribution_disloyal')

def personal_satisfaction_per_class(df, local_path):
    """
    Process data to get the satisfaction per class for personal travel into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    filtered_df = df.filter(df["Type of Travel"] == "Personal Travel")
    # Regrouper par "Class" et "satisfaction", et compter les occurrences
    class_satisfaction_distribution = (
        filtered_df.groupBy("Class", "satisfaction")
        .agg(count("*").alias("Count"))
    )
    save_dataframe(class_satisfaction_distribution, f'{local_path}personal_class_satisfaction_distribution')

def given_notes_per_services(df, loyal, classe, local_path, spark):
    """
    Process data to get the notes' mean for each service into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        classe (str): Classe to study (eco | business).
        local_path (str): The path to save the CSV file.
        spark (SparkSession): The Spark session.

    Returns:
        None
    """
    # Séparer les clients satisfaits et insatisfaits
    df_satisfied = df.filter(df["satisfaction"] == "satisfied")
    df_dissatisfied = df.filter(df["satisfaction"] == "dissatisfied")

    # Paramètres à analyser
    parameters = [
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink",
        "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service",
        "Leg room service", "Baggage handling", "Checkin service",
        "Cleanliness", "Online boarding"
    ]

    # Calculer la moyenne des notes pour chaque paramètre pour les clients satisfaits
    satisfied_means = df_satisfied.select(parameters).agg(*[mean(col(p)).alias(p) for p in parameters])
    # Calculer la moyenne des notes pour chaque paramètre pour les clients insatisfaits
    dissatisfied_means = df_dissatisfied.select(parameters).agg(*[mean(col(p)).alias(p) for p in parameters])
    # Convertir les DataFrames en RDDs et les joindre
    satisfied_means_rdd = satisfied_means.rdd.flatMap(
        lambda row: [(parameter, row[parameter]) for parameter in parameters])
    dissatisfied_means_rdd = dissatisfied_means.rdd.flatMap(
        lambda row: [(parameter, row[parameter]) for parameter in parameters])

    satisfied_means_df = spark.createDataFrame(satisfied_means_rdd.map(lambda x: Row(Parameter=x[0], Satisfied=x[1])))
    dissatisfied_means_df = spark.createDataFrame(
        dissatisfied_means_rdd.map(lambda x: Row(Parameter=x[0], Dissatisfied=x[1])))
    # Joindre les DataFrames sur le paramètre
    means_df = satisfied_means_df.join(dissatisfied_means_df, on="Parameter", how="outer")
    # Réorganiser les données avec melt (ou l'équivalent en PySpark)
    from pyspark.sql.functions import expr
    # Ajouter une colonne 'Satisfaction' avec des valeurs pour le melting
    means_df = means_df.withColumn(
        "Satisfaction",
        expr("CASE WHEN Satisfied IS NOT NULL THEN 'Satisfied' ELSE 'Dissatisfied' END"))
    # Convertir les colonnes de satisfaction en une seule colonne
    means_df = means_df.selectExpr("Parameter", "Satisfaction", "coalesce(Satisfied, Dissatisfied) as Average_Score")

    if loyal is True:
        save_dataframe(means_df, f'{local_path}services_satisfaction_loyal')
    else:
        if classe == "eco":
            save_dataframe(means_df, f'{local_path}services_satisfaction_disloyal_eco')
        else:
            save_dataframe(means_df, f'{local_path}services_satisfaction_disloyal_business')

def flight_distrib(df, loyal, classe, local_path):
    """
    Process data to get the flight distance distribution into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        classe (str): Classe to study (eco | business).
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    distance_dist = df.select("Flight Distance")
    if loyal is True:
        save_dataframe(distance_dist, f'{local_path}flight_distribution_loyal')
    else:
        if classe == "eco":
            save_dataframe(distance_dist, f'{local_path}flight_distribution_disloyal_eco')
        else:
            save_dataframe(distance_dist, f'{local_path}flight_distribution_disloyal_business')

def satisfaction_per_distance(df, loyal, local_path):
    """
    Process data to get the satisfaction per flight distance into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        classe (str): Classe to study (eco | business).
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # On garde la distance de vol et la satisfaction
    df = df.select("Flight Distance", "satisfaction")

    # Supprimer les lignes avec des valeurs manquantes
    df = df.dropna(subset=["Flight Distance", "satisfaction"])

    # Créer des bins pour l'histogramme
    if loyal is True:
        # Utiliser expr pour créer des bins en utilisant des conditions
        df = df.withColumn(
            "Distance Bin",
            when((col("Flight Distance") >= 0) & (col("Flight Distance") <= 300), '0-300')
            .when((col("Flight Distance") > 300) & (col("Flight Distance") <= 600), '300-600')
            .when((col("Flight Distance") > 600) & (col("Flight Distance") <= 900), '600-900')
            .when((col("Flight Distance") > 900) & (col("Flight Distance") <= 1200), '900-1200')
            .when((col("Flight Distance") > 1200) & (col("Flight Distance") <= 1500), '1200-1500')
            .when((col("Flight Distance") > 1500) & (col("Flight Distance") <= 1800), '1500-1800')
            .when((col("Flight Distance") > 1800) & (col("Flight Distance") <= 2100), '1800-2100')
            .when((col("Flight Distance") > 2100) & (col("Flight Distance") <= 2400), '2100-2400')
            .when((col("Flight Distance") > 2400) & (col("Flight Distance") <= 2700), '2400-2700')
            .when((col("Flight Distance") > 2700) & (col("Flight Distance") <= 3000), '2700-3000')
            .when((col("Flight Distance") > 3000) & (col("Flight Distance") <= 3500), '3000-3500')
            .when((col("Flight Distance") > 3500) & (col("Flight Distance") <= 4000), '3500-4000')
            .when((col("Flight Distance") > 4000) & (col("Flight Distance") <= 5000), '4000-5000')
            .when((col("Flight Distance") > 5000) & (col("Flight Distance") <= 6000), '5000-6000')
            .when((col("Flight Distance") > 6000) & (col("Flight Distance") <= 7000), '6000-7000')
            .when((col("Flight Distance") > 7000) & (col("Flight Distance") <= 8000), '7000-8000')
        )
    else:
        # Utiliser expr pour créer des bins en utilisant des conditions
        df = df.withColumn(
            "Distance Bin",
            when((col("Flight Distance") >= 1100) & (col("Flight Distance") <= 1400), '1100-1400')
            .when((col("Flight Distance") > 1400) & (col("Flight Distance") <= 1700), '1400-1700')
            .when((col("Flight Distance") > 1700) & (col("Flight Distance") <= 2000), '1700-2000')
            .when((col("Flight Distance") > 2000) & (col("Flight Distance") <= 2300), '2000-2300')
            .when((col("Flight Distance") > 2300) & (col("Flight Distance") <= 2600), '2300-2600')
            .when((col("Flight Distance") > 2600) & (col("Flight Distance") <= 3000), '2600-3000')
        )

    # Calculer le nombre de clients satisfaits et total dans chaque bin
    satisfaction_counts = (
        df.groupBy("Distance Bin", "satisfaction")
        .count()
        .withColumnRenamed("count", "Count")
    )
    # Calculer les totaux pour chaque bin
    total_counts = (
        satisfaction_counts.groupBy("Distance Bin")
        .agg(sum("Count").alias("Total"))
    )
    # Joindre les totaux avec les comptages de satisfaction
    joined_df = satisfaction_counts.join(total_counts, on="Distance Bin")
    # Calculer le pourcentage de clients satisfaits dans chaque bin
    satisfaction_percentage = (
        joined_df.withColumn("Percentage",
                             (col("Count") / col("Total")) * 100)
        .select("Distance Bin", "satisfaction", "Percentage")
    )

    if loyal is True:
        save_dataframe(satisfaction_percentage, f'{local_path}satisfaction_per_distance_loyal')
    else:
        save_dataframe(satisfaction_percentage, f'{local_path}satisfaction_per_distance_disloyal')

def satisfaction_per_distance_per_service(df, loyal, local_path):
    """
    Process data to get the notes' mean per flight distance bin for each service into a CSV file

    Args:
        df (DataFrame): The Spark dataframe.
        loyal (bool): To study loyal or non-loyal clients.
        classe (str): Classe to study (eco | business).
        local_path (str): The path to save the CSV file.

    Returns:
        None
    """
    # Convertir en DataFrame Pandas pour faciliter la manipulation
    df = df.select(
        "Flight Distance", "satisfaction", "Seat comfort", "Departure/Arrival time convenient",
        "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness", "Online boarding")

    # Supprimer les lignes avec des valeurs manquantes
    df.dropna(subset=["Flight Distance", "satisfaction"])

    # Créer des bins pour l'histogramme
    if loyal is True:
        # Utiliser expr pour créer des bins en utilisant des conditions
        df = df.withColumn(
            "Distance Bin",
            when((col("Flight Distance") >= 0) & (col("Flight Distance") <= 300), '0-300')
            .when((col("Flight Distance") > 300) & (col("Flight Distance") <= 600), '300-600')
            .when((col("Flight Distance") > 600) & (col("Flight Distance") <= 900), '600-900')
            .when((col("Flight Distance") > 900) & (col("Flight Distance") <= 1200), '900-1200')
            .when((col("Flight Distance") > 1200) & (col("Flight Distance") <= 1500), '1200-1500')
            .when((col("Flight Distance") > 1500) & (col("Flight Distance") <= 1800), '1500-1800')
            .when((col("Flight Distance") > 1800) & (col("Flight Distance") <= 2100), '1800-2100')
            .when((col("Flight Distance") > 2100) & (col("Flight Distance") <= 2400), '2100-2400')
            .when((col("Flight Distance") > 2400) & (col("Flight Distance") <= 2700), '2400-2700')
            .when((col("Flight Distance") > 2700) & (col("Flight Distance") <= 3000), '2700-3000')
            .when((col("Flight Distance") > 3000) & (col("Flight Distance") <= 3500), '3000-3500')
            .when((col("Flight Distance") > 3500) & (col("Flight Distance") <= 4000), '3500-4000')
            .when((col("Flight Distance") > 4000) & (col("Flight Distance") <= 5000), '4000-5000')
            .when((col("Flight Distance") > 5000) & (col("Flight Distance") <= 6000), '5000-6000')
            .when((col("Flight Distance") > 6000) & (col("Flight Distance") <= 7000), '6000-7000')
            .when((col("Flight Distance") > 7000) & (col("Flight Distance") <= 8000), '7000-8000')
        )
    else:
        # Utiliser expr pour créer des bins en utilisant des conditions
        df = df.withColumn(
            "Distance Bin",
            when((col("Flight Distance") >= 1100) & (col("Flight Distance") <= 1400), '1100-1400')
            .when((col("Flight Distance") > 1400) & (col("Flight Distance") <= 1700), '1400-1700')
            .when((col("Flight Distance") > 1700) & (col("Flight Distance") <= 2000), '1700-2000')
            .when((col("Flight Distance") > 2000) & (col("Flight Distance") <= 2300), '2000-2300')
            .when((col("Flight Distance") > 2300) & (col("Flight Distance") <= 2600), '2300-2600')
            .when((col("Flight Distance") > 2600) & (col("Flight Distance") <= 3000), '2600-3000')
        )

    # Séparer les clients satisfaits et insatisfaits
    df_satisfied = df.filter(df["satisfaction"] == "satisfied")
    df_dissatisfied = df.filter(df["satisfaction"] == "dissatisfied")

    # Calculer les notes moyennes pour chaque paramètre et chaque bin
    params = [
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink", "Gate location",
        "Inflight wifi service", "Inflight entertainment", "Online support", "Ease of Online booking",
        "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness",
        "Online boarding"
    ]

    # Moyennes pour les clients
    avg_satisfied = df_satisfied.groupBy("Distance Bin").agg(*[mean(col(param)).alias(f"{param}_satisfied") for param in params])
    avg_dissatisfied = df_dissatisfied.groupBy("Distance Bin").agg(*[mean(col(param)).alias(f"{param}_dissatisfied") for param in params])

    # Fusionner les deux DataFrames pour comparaison
    avg_scores = avg_satisfied.join(avg_dissatisfied, on="Distance Bin", how="outer")
    if loyal is True:
        save_dataframe(avg_scores, f'{local_path}services_satisfaction_per_distance_loyal')
    else:
        save_dataframe(avg_scores, f'{local_path}services_satisfaction_per_distance_disloyal')

# ANALYSES MAIN FUNCTION
def get_spark_analyses():
    # Build a Spark session
    spark = SparkSession.builder \
        .appName("Airline Satisfaction Analysis") \
        .getOrCreate()

    # Load data from HDFS
    hdfs_address = "hdfs://localhost:9000/"
    df = load_data_from_hdfs(spark, hdfs_address)
    print("ZER GOOD")

    # Global analyses
    create_folder('csv')
    local_path = "csv/"
    if not path.exists('csv/satisfaction_distribution.csv'):
        satisfaction_distrib(df, False, local_path)
    if not path.exists('csv/client_type_distribution.csv'):
        client_type_distrib(df, False, local_path)
    if not path.exists('csv/age_distribution.csv'):
        age_distrib(df, False, local_path)
    # rating_columns = [
    #     'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink',
    #     'Gate location', 'Online support', 'Ease of Online booking', 'On-board service',
    #     'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness',
    #     'Online boarding'
    # ]
    # if not path.exists('csv/global_means_notes.csv'):
    #     global_mean_notes(df, rating_columns, local_path)
    # if not path.exists('csv/ecart_type.csv'):
    #     global_ecart_type(df, rating_columns, local_path)
    # if not path.exists('csv/variance.csv'):
    #     global_variance(df, rating_columns, local_path)

    # Age distribution between 20 and 60
    df_age_filtered = df.filter((df.Age >= 20) & (df.Age <= 60))
    if not path.exists('csv/age_distribution_filtered.csv'):
        age_distrib(df_age_filtered, True, local_path)
    if not path.exists('csv/filtered_satisfaction_distribution.csv'):
        satisfaction_distrib(df_age_filtered, True, local_path)
    if not path.exists('csv/filtered_customer_type_distribution.csv'):
        client_type_distrib(df_age_filtered, True, local_path)
    numeric_cols = [
        'Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
        'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment',
        'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding',
        'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]
    if not path.exists('csv/correlation_matrix.csv'):
        correlation_matrix(df, numeric_cols, local_path, spark)

    # Loyal clients
    if not path.exists('csv/travel_type_distribution_loyal.csv'):
        travel_type_distrib(df_age_filtered, True, local_path)
    df_loyal = df_age_filtered.filter(df_age_filtered["Customer Type"] == "Loyal Customer")
    if not path.exists('csv/travel_type_satisfaction_distribution_loyal.csv'):
        travel_type_satisfaction(df_loyal, True, local_path)
    if not path.exists('csv/business_class_satisfaction_distribution_loyal.csv'):
        business_satisfaction_per_class(df_loyal, True, local_path)
    if not path.exists('csv/personal_class_satisfaction_distribution.csv'):
        personal_satisfaction_per_class(df_loyal, local_path)

    # Loyal clients - Eco class
    df_loyal_eco = df_loyal.filter(df_loyal["Class"] == "Eco")
    if not path.exists('csv/services_satisfaction_loyal.csv'):
        given_notes_per_services(df_loyal_eco, True, "eco", local_path, spark)
    if not path.exists('csv/flight_distribution_loyal.csv'):
        flight_distrib(df_loyal_eco, True, "eco", local_path)
    if not path.exists('csv/satisfaction_per_distance_loyal.csv'):
        satisfaction_per_distance(df_loyal_eco, True, local_path)
    if not path.exists('csv/services_satisfaction_per_distance_loyal.csv'):
        satisfaction_per_distance_per_service(df_loyal_eco, True, local_path)

    # Non-loyal clients
    if not path.exists('csv/travel_type_distribution_disloyal.csv'):
        travel_type_distrib(df_age_filtered, False, local_path)
    df_disloyal = df_age_filtered.filter(df_age_filtered["Customer Type"] == "disloyal Customer")
    if not path.exists('csv/travel_type_satisfaction_distribution_disloyal.csv'):
        travel_type_satisfaction(df_disloyal, False, local_path)
    if not path.exists('csv/business_class_satisfaction_distribution_disloyal.csv'):
        business_satisfaction_per_class(df_disloyal, False, local_path)

    # Non-loyal clients - Business & Eco classes
    df_disloyal_eco = df_disloyal.filter(
        (df_disloyal["Type of Travel"] == "Business travel") &
        (df_disloyal["Class"] == "Eco")
    )
    df_disloyal_busi = df_disloyal.filter(
        (df_disloyal["Type of Travel"] == "Business travel") &
        (df_disloyal["Class"] == "Business")
    )
    if not path.exists('csv/services_satisfaction_disloyal_eco.csv'):
        given_notes_per_services(df_disloyal_eco, False, "eco", local_path, spark)
    if not path.exists('csv/services_satisfaction_disloyal_business.csv'):
        given_notes_per_services(df_disloyal_busi, False, "business", local_path, spark)
    if not path.exists('csv/flight_distribution_disloyal_eco.csv'):
        flight_distrib(df_disloyal_eco, False, "eco", local_path)
    if not path.exists('csv/flight_distribution_disloyal_business.csv'):
        flight_distrib(df_disloyal_busi, False, "business", local_path)

    df_disloyal_distance = df_disloyal_busi.filter(
        (df['Flight Distance'] >= 1100) &
        (df['Flight Distance'] <= 3000)
    )
    if not path.exists('csv/satisfaction_per_distance_disloyal.csv'):
        satisfaction_per_distance(df_disloyal_distance, False, local_path)
    if not path.exists('csv/services_satisfaction_per_distance_disloyal.csv'):
        satisfaction_per_distance_per_service(df_disloyal_distance, False, local_path)

    # Stop the Spark session
    print("*************************")
    print("ANALYSES COMPLETED")
    print("*************************")
    spark.stop()
