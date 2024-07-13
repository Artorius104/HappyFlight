from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, corr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import plotly.express as px
import pandas as pd


hdfs_address = "hdfs://localhost:9000/"

# FONCTIONS UTILITAIRES SPARK
def load_data(spark):
    # Lire le fichier CSV depuis HDFS
    df = spark.read.csv(f'{hdfs_address}airline_data/Airline_customer_satisfaction.csv', header=True, inferSchema=True)

    return df

# FONCTIONS D'ANALYSES
def satisfaction_by_client_type(df):
    avg_satisfaction_by_customer_type = df.groupBy("Customer Type").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_customer_type.show()

def satisfaction_by_travel_type(df):
    avg_satisfaction_by_travel_type = df.groupBy("Type of Travel").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_travel_type.show()

def satisfaction_by_travel_class(df):
    avg_satisfaction_by_class = df.groupBy("Class").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_class.show()

def satisfaction_by_flight_distance(df):
    avg_satisfaction_by_distance = df.groupBy("Flight Distance").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_distance.show()

def satisfaction_by_delays(df):
    avg_satisfaction_by_departure_delay = df.groupBy("Departure Delay in Minutes").agg(
        avg("Satisfaction").alias("avg_satisfaction")
    )
    avg_satisfaction_by_departure_delay.show()

    avg_satisfaction_by_arrival_delay = df.groupBy("Arrival Delay in Minutes").agg(
        avg("Satisfaction").alias("avg_satisfaction")
    )
    avg_satisfaction_by_arrival_delay.show()

def correlation_satisfaction_and_services(df):
    services_columns = [
        "Seat comfort", "Food and drink", "Inflight wifi service",
        "Inflight entertainment", "Online support", "Ease of Online booking",
        "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Cleanliness", "Online boarding"
    ]

    for col_name in services_columns:
        correlation = df.stat.corr(col_name, "Satisfaction")
        print(f"Correlation between {col_name} and Satisfaction: {correlation}")

def satisfaction_per_age_group(df):
    avg_satisfaction_by_age = df.groupBy("Age").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_age.show()

def satisfaction_by_seat_comfort(df):
    avg_satisfaction_by_seat_comfort = df.groupBy("Seat comfort").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_seat_comfort.show()

def satisfaction_by_cleanliness(df):
    avg_satisfaction_by_cleanliness = df.groupBy("Cleanliness").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_cleanliness.show()

def satisfaction_by_booking_service(df):
    avg_satisfaction_by_online_booking = df.groupBy("Ease of Online booking").agg(
        avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_online_booking.show()

def satisfaction_by_onboard_services(df):
    avg_satisfaction_by_onboard_service = df.groupBy("On-board service").agg(
        avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_onboard_service.show()

def satisfaction_by_online_support(df):
    avg_satisfaction_by_online_support = df.groupBy("Online support").agg(avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_online_support.show()

def satisfaction_by_punctuality(df):
    # Assuming there is a column indicating on-time performance, e.g., "On-time arrival"
    avg_satisfaction_by_on_time_arrival = df.groupBy("On-time arrival").agg(
        avg("Satisfaction").alias("avg_satisfaction"))
    avg_satisfaction_by_on_time_arrival.show()

def satisfaction_combined_analysis(df):
    # Select relevant features and assemble them into a feature vector
    feature_columns = [
        "Age", "Flight Distance", "Seat comfort", "Departure/Arrival time convenient",
        "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Cleanliness", "Online boarding",
        "Departure Delay in Minutes", "Arrival Delay in Minutes"
    ]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_features = assembler.transform(df)

    # Fit a linear regression model
    lr = LinearRegression(featuresCol="features", labelCol="Satisfaction")
    lr_model = lr.fit(df_features)

    # Print the coefficients and intercept for linear regression
    print(f"Coefficients: {lr_model.coefficients}")
    print(f"Intercept: {lr_model.intercept}")

    # Summarize the model over the training set and print out some metrics
    training_summary = lr_model.summary
    print(f"RMSE: {training_summary.rootMeanSquaredError}")
    print(f"R2: {training_summary.r2}")

# FONCTIONS D'ANALYSES
# def satisfaction_by_client_type(df):
#     avg_satisfaction_by_customer_type = df.groupBy("Customer Type").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_customer_type.show()
#
# def satisfaction_by_travel_type(df):
#     avg_satisfaction_by_travel_type = df.groupBy("Type of Travel").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_travel_type.show()
#
# def satisfaction_by_travel_class(df):
#     avg_satisfaction_by_class = df.groupBy("Class").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_class.show()
#
# def satisfaction_by_flight_distance(df):
#     avg_satisfaction_by_distance = df.groupBy("Flight Distance").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_distance.show()
#
# def satisfaction_by_delays(df):
#     avg_satisfaction_by_departure_delay = df.groupBy("Departure Delay in Minutes").agg(
#         avg("Satisfaction").alias("avg_satisfaction")
#     )
#     avg_satisfaction_by_departure_delay.show()
#
#     avg_satisfaction_by_arrival_delay = df.groupBy("Arrival Delay in Minutes").agg(
#         avg("Satisfaction").alias("avg_satisfaction")
#     )
#     avg_satisfaction_by_arrival_delay.show()
#
# def correlation_satisfaction_and_services(df):
#     services_columns = [
#         "Seat comfort", "Food and drink", "Inflight wifi service",
#         "Inflight entertainment", "Online support", "Ease of Online booking",
#         "On-board service", "Leg room service", "Baggage handling",
#         "Checkin service", "Cleanliness", "Online boarding"
#     ]
#
#     for col_name in services_columns:
#         correlation = df.stat.corr(col_name, "Satisfaction")
#         print(f"Correlation between {col_name} and Satisfaction: {correlation}")
#
# def satisfaction_per_age_group(df):
#     avg_satisfaction_by_age = df.groupBy("Age").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_age.show()
#
# def satisfaction_by_seat_comfort(df):
#     avg_satisfaction_by_seat_comfort = df.groupBy("Seat comfort").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_seat_comfort.show()
#
#
# def satisfaction_by_cleanliness(df):
#     avg_satisfaction_by_cleanliness = df.groupBy("Cleanliness").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_cleanliness.show()
#
# def satisfaction_by_booking_service(df):
#     avg_satisfaction_by_online_booking = df.groupBy("Ease of Online booking").agg(
#         avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_online_booking.show()
#
# def satisfaction_by_onboard_services(df):
#     avg_satisfaction_by_onboard_service = df.groupBy("On-board service").agg(
#         avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_onboard_service.show()
#
# def satisfaction_by_online_support(df):
#     avg_satisfaction_by_online_support = df.groupBy("Online support").agg(avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_online_support.show()
#
# def satisfaction_by_punctuality(df):
#     # Assuming there is a column indicating on-time performance, e.g., "On-time arrival"
#     avg_satisfaction_by_on_time_arrival = df.groupBy("On-time arrival").agg(
#         avg("Satisfaction").alias("avg_satisfaction"))
#     avg_satisfaction_by_on_time_arrival.show()
#
# def satisfaction_combined_analysis(df):
#     # Select relevant features and assemble them into a feature vector
#     feature_columns = [
#         "Age", "Flight Distance", "Seat comfort", "Departure/Arrival time convenient",
#         "Food and drink", "Gate location", "Inflight wifi service", "Inflight entertainment",
#         "Online support", "Ease of Online booking", "On-board service", "Leg room service",
#         "Baggage handling", "Checkin service", "Cleanliness", "Online boarding",
#         "Departure Delay in Minutes", "Arrival Delay in Minutes"
#     ]
#
#     assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
#     df_features = assembler.transform(df)
#
#     # Fit a linear regression model
#     lr = LinearRegression(featuresCol="features", labelCol="Satisfaction")
#     lr_model = lr.fit(df_features)
#
#     # Print the coefficients and intercept for linear regression
#     print(f"Coefficients: {lr_model.coefficients}")
#     print(f"Intercept: {lr_model.intercept}")
#
#     # Summarize the model over the training set and print out some metrics
#     training_summary = lr_model.summary
#     print(f"RMSE: {training_summary.rootMeanSquaredError}")
#     print(f"R2: {training_summary.r2}")


# MAIN FUNCTION
def main() -> None:
    # Initialiser une session Spark
    spark = SparkSession.builder \
        .appName("Airline Satisfaction Analysis") \
        .getOrCreate()

    # Charger les données du HDFS
    df = load_data(spark)

    # Analyses

    # Sauvegarder le DataFrame dans HDFS en format CSV
    # result_df.write.csv('/airline_data/flight_distance_count', header=True)

    # Arrêter la session Spark
    spark.stop()


if __name__ == "__main__":
    main()
