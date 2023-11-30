from datasets import load_dataset, load_metric
from configs.train_config import env_vars
from pyspark.sql import SparkSession
import argparse
import datasets


def main(config):
    """
    Create sample training and validation datasets

    @param config: configuration dictionary
    @return: None
    """
    spark = SparkSession.builder.getOrCreate()

    # Load sample dataset
    datasets.utils.logging.disable_progress_bar()
    dataset = load_dataset("glue", "cola")

    # Split data into train and validation
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    # Transform the datasets into Spark DataFrames
    train_df = spark.createDataFrame(train_dataset.to_pandas())
    val_data_df = spark.createDataFrame(validation_dataset.to_pandas())

    """
    ETL Transformations in Spark go here:
    """

    # Create the Unity Catalog table names from the configuration dict
    train_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['training_data']}"
    val_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['validation_data']}"

    # For example purposes, the datasets will be written out to the
    # training and validation tables in overwrite
    (train_df
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(train_data_table)
     )

    (val_data_df
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(val_data_table)
     )


if __name__ == "__main__":
    # Get environment variables
    parser = argparse.ArgumentParser()

    envs = ["dev", "stg", "tst", "prd"]
    parser.add_argument("--env",
                        type=str,
                        choices=envs,
                        default="prd")

    args = parser.parse_args()
    env = args.env

    # Load config for the environment
    config = env_vars[env]

    # Run preprocessing job
    main(config)
