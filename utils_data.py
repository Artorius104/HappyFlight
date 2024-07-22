from os import makedirs, path, listdir
from shutil import move, rmtree

from pyspark.sql import SparkSession, DataFrame


def create_folder(folder_path: str) -> None:
    """Creates a folder if it does not exist."""
    if not path.exists(folder_path):
        makedirs(folder_path)

def load_data(spark: SparkSession, hdfs_address: str) -> DataFrame:
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
