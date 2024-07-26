from os import makedirs, path, listdir
from shutil import move, rmtree

from pyspark.sql import SparkSession, DataFrame

from display_module import (satisfaction_pie_chart, customer_type_pie_chart, age_distribution_chart,
                            plot_correlation_heatmap, travel_type_pie_chart, travel_type_satisfaction_bar_chart,
                            business_class_satisfaction_bar_chart, personal_class_satisfaction_bar_chart,
                            per_services_satisfaction_bar_chart, flight_distance_histogram,
                            flight_distance_satisfaction_histogram, services_comparison_graphs)
from utils_app import create_services_comparison_layout


def create_folder(folder_path: str) -> None:
    """Creates a folder if it does not exist."""
    if not path.exists(folder_path):
        makedirs(folder_path)

def load_data_from_hdfs(spark: SparkSession, hdfs_address: str) -> DataFrame:
    """
    Load data from the HDFS address.

    Args:
        spark (SparkSession): The Spark session.
        hdfs_address (str): The HDFS address.

    Returns:
        DataFrame: The Spark dataframe with the data.
    """
    df = spark.read.csv(
        f'{hdfs_address}airline_data/Airline_customer_satisfaction.csv',
        header=True,
        inferSchema=True
    )

    return df

def load_csv_to_dataframe(file_path: str, spark: SparkSession) -> DataFrame:
    df = spark.read.csv(f'{file_path}', header=True, inferSchema=True)

    return df

def save_dataframe(df, local_file):
    """
    Save the dataframe into a CSV file in the specified path

    Args:
        df (DataFrame): The Spark dataframe.
        local_file (str): The path to save the CSV file with the filename (without .csv).

    Returns:
        None
    """
    # Réduire à une seule partition
    df = df.coalesce(1)

    # Définir le chemin temporaire et le chemin final
    temp_path = f'{local_file}'
    final_path = f'{local_file}.csv'

    # Sauvegarder le résultat en un fichier CSV temporaire
    df.write.csv(temp_path, header=True, mode="overwrite")

    # Trouver le fichier CSV temporaire généré
    temp_file = [file for file in listdir(temp_path) if file.endswith('.csv')][0]

    # Déplacer et renommer le fichier temporaire
    move(path.join(temp_path, temp_file), final_path)

    # Supprimer le dossier temporaire
    rmtree(temp_path)
    print(f'{local_file}.csv SAVED')

def get_resume_figures(folder_path):
    spark = SparkSession.builder \
        .appName("Get Resume Analysis CSVs") \
        .getOrCreate()

    df_satisfaction = load_csv_to_dataframe(f'{folder_path}satisfaction_distribution.csv', spark)
    df_customer_type = load_csv_to_dataframe(f'{folder_path}client_type_distribution.csv', spark)
    df_age = load_csv_to_dataframe(f'{folder_path}age_distribution.csv', spark)
    df_age_filtered = load_csv_to_dataframe(f'{folder_path}age_distribution_filtered.csv', spark)
    df_satisfaction_filtered = load_csv_to_dataframe(f'{folder_path}filtered_satisfaction_distribution.csv', spark)
    df_customer_type_filtered = load_csv_to_dataframe(f'{folder_path}filtered_customer_type_distribution.csv', spark)
    df_corr_matrix = load_csv_to_dataframe(f'{folder_path}correlation_matrix.csv', spark)

    satisfaction_fig = satisfaction_pie_chart(df_satisfaction, False)
    customer_type_fig = customer_type_pie_chart(df_customer_type,False)
    age_fig = age_distribution_chart(df_age,False)
    age_filtered_fig = age_distribution_chart(df_age_filtered,True)
    satisfaction_filtered_fig = satisfaction_pie_chart(df_satisfaction_filtered,True)
    customer_type_filtered_fig = customer_type_pie_chart(df_customer_type_filtered,True)
    corr_matrix_fig = plot_correlation_heatmap(df_corr_matrix)

    res = {
        "satisfaction": satisfaction_fig,
        "customer_type": customer_type_fig,
        "age": age_fig,
        "age_filtered": age_filtered_fig,
        "satisfaction_filtered": satisfaction_filtered_fig,
        "customer_type_filtered": customer_type_filtered_fig,
        "corr_matrix": corr_matrix_fig
    }
    spark.stop()

    return res

def get_loyal_figures(folder_path):
    spark = SparkSession.builder \
        .appName("Get Loyal Analysis CSVs") \
        .getOrCreate()

    df_travel_type = load_csv_to_dataframe(f'{folder_path}travel_type_distribution_loyal.csv', spark)
    df_travel_type_satisfaction = load_csv_to_dataframe(f'{folder_path}travel_type_satisfaction_distribution_loyal.csv', spark)
    df_business_class_satisfaction = load_csv_to_dataframe(f'{folder_path}business_class_satisfaction_distribution_loyal.csv', spark)
    df_personal_class_satisfaction = load_csv_to_dataframe(f'{folder_path}personal_class_satisfaction_distribution.csv', spark)
    df_per_services_satisfaction = load_csv_to_dataframe(f'{folder_path}services_satisfaction_loyal.csv', spark)
    df_flight_distance = load_csv_to_dataframe(f'{folder_path}flight_distribution_loyal.csv', spark)
    df_flight_distance_satisfaction = load_csv_to_dataframe(f'{folder_path}satisfaction_per_distance_loyal.csv', spark)
    df_services_satisfaction_distance = load_csv_to_dataframe(f'{folder_path}services_satisfaction_per_distance_disloyal.csv', spark)

    travel_type_fig = travel_type_pie_chart(df_travel_type)
    travel_type_satisfaction_fig = travel_type_satisfaction_bar_chart(df_travel_type_satisfaction)
    business_class_satisfaction_fig = business_class_satisfaction_bar_chart(df_business_class_satisfaction)
    personal_class_satisfaction_fig = personal_class_satisfaction_bar_chart(df_personal_class_satisfaction)
    per_services_satisfaction_fig = per_services_satisfaction_bar_chart(df_per_services_satisfaction, "")
    flight_distance_fig = flight_distance_histogram(df_flight_distance, "")
    flight_distance_satisfaction_fig = flight_distance_satisfaction_histogram(df_flight_distance_satisfaction, True)
    services_comparison_fig_list = services_comparison_graphs(df_services_satisfaction_distance)
    services_comparison_layout = create_services_comparison_layout(services_comparison_fig_list)

    res = {
        "travel_type": travel_type_fig,
        "travel_type_satisfaction": travel_type_satisfaction_fig,
        "business_class_satisfaction": business_class_satisfaction_fig,
        "personal_class_satisfaction": personal_class_satisfaction_fig,
        "per_services_satisfaction": per_services_satisfaction_fig,
        "flight_distance": flight_distance_fig,
        "flight_distance_satisfaction": flight_distance_satisfaction_fig,
        "services_comparison": services_comparison_layout
    }
    spark.stop()

    return res

def get_non_loyal_figures(folder_path):
    spark = SparkSession.builder \
        .appName("Get Non-Loyal Analysis CSVs") \
        .getOrCreate()

    df_travel_type = load_csv_to_dataframe(f'{folder_path}travel_type_distribution_disloyal.csv', spark)
    df_travel_type_satisfaction = load_csv_to_dataframe(f'{folder_path}travel_type_satisfaction_distribution_disloyal.csv', spark)
    df_business_class_satisfaction = load_csv_to_dataframe(f'{folder_path}business_class_satisfaction_distribution_disloyal.csv', spark)
    df_per_services_satisfaction_eco = load_csv_to_dataframe(f'{folder_path}services_satisfaction_disloyal_eco.csv', spark)
    df_per_services_satisfaction_business = load_csv_to_dataframe(f'{folder_path}services_satisfaction_disloyal_business.csv', spark)
    df_flight_distance_eco = load_csv_to_dataframe(f'{folder_path}flight_distribution_disloyal_eco.csv', spark)
    df_flight_distance_business = load_csv_to_dataframe(f'{folder_path}flight_distribution_disloyal_business.csv', spark)
    df_flight_distance_satisfaction = load_csv_to_dataframe(f'{folder_path}satisfaction_per_distance_disloyal.csv', spark)
    df_services_satisfaction_distance = load_csv_to_dataframe(f'{folder_path}services_satisfaction_per_distance_disloyal.csv', spark)

    travel_type_fig = travel_type_pie_chart(df_travel_type)
    travel_type_satisfaction_fig = travel_type_satisfaction_bar_chart(df_travel_type_satisfaction)
    business_class_satisfaction_fig = business_class_satisfaction_bar_chart(df_business_class_satisfaction)
    per_services_satisfaction_eco_fig = per_services_satisfaction_bar_chart(df_per_services_satisfaction_eco, "eco")
    per_services_satisfaction_business_fig = per_services_satisfaction_bar_chart(df_per_services_satisfaction_business, "business")
    flight_distance_eco_fig = flight_distance_histogram(df_flight_distance_eco, "eco")
    flight_distance_business_fig = flight_distance_histogram(df_flight_distance_business, "business")
    flight_distance_satisfaction_fig = flight_distance_satisfaction_histogram(df_flight_distance_satisfaction, False)
    services_comparison_fig_list = services_comparison_graphs(df_services_satisfaction_distance)
    services_comparison_layout = create_services_comparison_layout(services_comparison_fig_list)

    res = {
        "travel_type": travel_type_fig,
        "travel_type_satisfaction": travel_type_satisfaction_fig,
        "business_class_satisfaction": business_class_satisfaction_fig,
        "per_services_satisfaction_eco": per_services_satisfaction_eco_fig,
        "per_services_satisfaction_business": per_services_satisfaction_business_fig,
        "flight_distance_eco": flight_distance_eco_fig,
        "flight_distance_business": flight_distance_business_fig,
        "flight_distance_satisfaction": flight_distance_satisfaction_fig,
        "services_comparison": services_comparison_layout
    }
    spark.stop()

    return res
