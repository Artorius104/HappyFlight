from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os


# os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
hdfs_address = "hdfs://localhost:9000/"


def main() -> None:
    spark = SparkSession.builder \
                        .appName("Airline Customer Satisfaction Analysis") \
                        .getOrCreate()
    
    # Lire le fichier CSV depuis HDFS
    df = spark.read.csv(f'{hdfs_address}airline_data/Airline_customer_satisfaction.csv', header=True, inferSchema=True)

    # Afficher le schéma du DataFrame
    print("-------- SCHEMA --------")
    df.printSchema()

    # Afficher les premières lignes du DataFrame
    print("-------- FIRST 5 LINES --------")
    df.show(5)
    
    # Sélectionner certaines colonnes
    selected_df = df.select("Flight Distance", "Departure Delay in Minutes", "Satisfaction")

    # Filtrer les données
    print("-------- FILTERED FIRST 5 --------")
    filtered_df = df.filter(df["Satisfaction"] == "satisfied")
    filtered_df.show(5)

    # Calculer la moyenne du retard au départ
    print("-------- AVG DEPARTURE DELAY --------")
    avg_departure_delay = df.groupBy("Satisfaction").avg("Departure Delay in Minutes")
    avg_departure_delay.show()

    # Mapper avec la distance de vol
    mapped_df = df.select(col("Flight Distance").alias("distance")).rdd.map(lambda row: (row.distance, 1))

    # Reduce
    reduced_df = mapped_df.reduceByKey(lambda a, b: a + b)

    # Convertir en DataFrame pour afficher
    print("-------- MAP REDUCE --------")
    result_df = reduced_df.toDF(["Flight Distance", "Count"])
    result_df.show()

    # # Sauvegarder le DataFrame dans HDFS en format CSV
    # result_df.write.csv('/airline_data/flight_distance_count', header=True)

    # Arrêter la session Spark
    spark.stop()


if __name__ == "__main__":
    main()
