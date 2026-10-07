import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run(spark: SparkSession, input_path: str, output_path: str) -> None:
    logging.info("Reading text file from: %s", input_path)
    input_df = spark.read.text(input_path)

    # 1. Lowercase the entire line
    # 2. Extract words (sequences of letters/numbers possibly containing apostrophes)
    #    or split on non-word characters except apostrophe.
    #    The regex pattern [a-z0-9]+(?:'[a-z0-9]+)? matches words and contractions like "haven't", "sailors'"
    #    Alternatively, we can replace all punctuation except letters, digits, apostrophes and spaces with space,
    #    or use regexp_extract_all / split.
    
    # Clean non-alphanumeric characters except apostrophe and space:
    cleaned_df = input_df.select(
        F.lower(F.col("value")).alias("text")
    ).select(
        # Replace dashes, commas, quotes, periods, semicolons with space
        F.regexp_replace(F.col("text"), r"[^a-z0-9'\s]", " ").alias("text")
    ).select(
        # Handle trailing/isolated quotes or double dashes cleaned to spaces
        # Split text by one or more whitespace characters
        F.split(F.trim(F.col("text")), r"\s+").alias("words")
    )

    # Explode words into individual rows
    words_df = cleaned_df.select(F.explode(F.col("words")).alias("word"))

    # Trim any stray quotes/apostrophes at the edges of words (e.g. "word" or 'word')
    # Note: sailors' in the expected dataset retains the apostrophe: ["sailors'", 1]
    # But quotes like "Whenever -> whenever, and \"just -> just
    # Let's clean leading/trailing standard quotes if any, and filter out empty strings
    words_df = words_df.filter(F.col("word") != "")

    # Group by word and count occurrences
    word_count_df = (
        words_df.groupBy("word")
        .agg(F.count("*").alias("count"))
        .orderBy("word")
    )

    logging.info("Writing csv to directory: %s", output_path)
    word_count_df.coalesce(1).write.csv(output_path, header=True)