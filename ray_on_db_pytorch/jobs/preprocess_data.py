from datasets import load_dataset, load_metric
from configs.train_config import env_vars
import argparse
import datasets


def main(config):
    datasets.utils.logging.disable_progress_bar()
    dataset = load_dataset("glue", "cola")

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    train_df = spark.createDataFrame(train_dataset.to_pandas())
    val_data_df = spark.createDataFrame(validation_dataset.to_pandas())

    train_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['training_data']}"
    val_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['validation_data']}"

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
    parser = argparse.ArgumentParser()

    envs = ["dev", "stg", "tst", "prd"]
    parser.add_argument("--env",
                        type=str,
                        choices=envs,
                        default="prd")

    args = parser.parse_args()
    env = args.env

    config = env_vars[env]
    main(config)
