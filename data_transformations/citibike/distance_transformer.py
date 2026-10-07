from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

METERS_PER_FOOT = 0.3048
FEET_PER_MILE = 5280
EARTH_RADIUS_IN_METERS = 6371e3
METERS_PER_MILE = METERS_PER_FOOT * FEET_PER_MILE
EARTH_RADIUS_IN_MILES = EARTH_RADIUS_IN_METERS / METERS_PER_MILE


def compute_distance(_spark: SparkSession, dataframe: DataFrame) -> DataFrame:
    # Convert latitude and longitude from degrees to radians
    lat1 = F.radians(F.col("start_station_latitude"))
    lon1 = F.radians(F.col("start_station_longitude"))
    lat2 = F.radians(F.col("end_station_latitude"))
    lon2 = F.radians(F.col("end_station_longitude"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        F.pow(F.sin(dlat / 2.0), 2)
        + F.cos(lat1) * F.cos(lat2) * F.pow(F.sin(dlon / 2.0), 2)
    )
    c = 2.0 * F.atan2(F.sqrt(a), F.sqrt(1.0 - a))

    # Calculate distance in miles and round to 2 decimal places
    distance = F.round(F.lit(EARTH_RADIUS_IN_MILES) * c, 2).cast(DoubleType())

    return dataframe.withColumn("distance", distance)


def run(
    spark: SparkSession, input_dataset_path: str, transformed_dataset_path: str
) -> None:
    input_dataset = spark.read.parquet(input_dataset_path)
    input_dataset.show()

    dataset_with_distances = compute_distance(spark, input_dataset)
    dataset_with_distances.show()

    dataset_with_distances.write.parquet(transformed_dataset_path, mode="append")