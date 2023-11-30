import argparse
import os

import pytorch_lightning as pl
import ray
import ray.train
import torch
import mlflow
from mlflow.models import infer_signature

from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer

from .configs.train_config import env_vars
from .model import SentimentModel
from .shared.ray_utils import init_ray, delta_to_raydata
from .shared.pytorch_utils import tokenize_sentence
from .shared.spark_utils import get_n_cpus_per_node


# Global variables for the particular training job
# Username to store MLflow values
# CHANGE TO PERSONAL USERNAME IN THE DATABRICKS WORKSPACE
USER = "stephen.offer@databricks.com"
# Model name to store in the model registry
MLFLOW_MODEL_NAME = "RAY_PYTORCH_DEMO"
# Model name
MODEL_TYPE = "BERT_BASE_CASED"
# Create the experiment name
EXPERIMENT_NAME = f"{MODEL_TYPE}_{MLFLOW_MODEL_NAME}"
# Generate MLflow path
MLFLOW_PATH = f"/Users/{USER}"

# Training configuration for the train function
train_func_config = {
    "lr": 1e-5,
    "eps": 1e-8,
    "batch_size": 16,
    "max_epochs": 5,
}


def train_func(config):
    """

    @param config:
    @return:
    """
    # Unpack the input configs passed from `TorchTrainer(train_loop_config)`
    lr = config["lr"]
    eps = config["eps"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")

    # Create a dataloader for Ray Datasets
    train_ds_loader = train_ds.iter_torch_batches(batch_size=batch_size)
    val_ds_loader = val_ds.iter_torch_batches(batch_size=batch_size)

    # Create the Model
    model = SentimentModel(lr=lr, eps=eps)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=val_ds_loader)


def main(config, use_gpu):
    """

    @param config: Configuration dict
    @param use_gpu: Boolean value of whether to use GPUs
    @return:
    """
    # Get the training and validation datasets from Unity Catalog
    train_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['training_data']}"
    val_data_table = f"{config['uc_catalog']}.{config['schema']}.{config['validation_data']}"

    # Read the Delta Lake tables in Unityb Catalog and load into Ray Data
    train_dataset = delta_to_raydata(train_data_table)
    validation_dataset = delta_to_raydata(val_data_table)

    # Map the tokenize_sentence across the datasets
    train_dataset = train_dataset.map_batches(tokenize_sentence, batch_format="numpy")
    validation_dataset = validation_dataset.map_batches(tokenize_sentence, batch_format="numpy")

    # Save the top-2 checkpoints according to the evaluation metric
    # The checkpoints and metrics are reported by `RayTrainReportCallback`
    run_config = RunConfig(
        name="ptl-sent-classification",
        callbacks=[
            MLflowLoggerCallback(
                experiment_name=os.path.join(MLFLOW_PATH, EXPERIMENT_NAME)
            )
        ],
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="matthews_correlation",
            checkpoint_score_order="max",
        ),
    )

    scaling_config = ScalingConfig(num_workers=4, use_gpu=use_gpu)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_func_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset, "validation": validation_dataset},
    )

    result = trainer.fit()
    checkpoint_path = result.best_checkpoints

    model = torch.load(checkpoint_path)

    # signature = infer_signature(validation_dataset.numpy(), model(validation_dataset).detach().numpy())
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, MLFLOW_MODEL_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    envs = ["dev", "stg", "tst", "prd"]
    parser.add_argument("--env",
                        type=str,
                        choices=envs,
                        default="prd")

    parser.add_argument("--use_gpu",
                        type=str,
                        choices=["True", "true", "TRUE", "False", "false", "FALSE", "None"],
                        default="None")

    parser.add_argument("--n_workers",
                        type=int,
                        default=None)

    parser.add_argument("--use_autoscaler",
                        type=str,
                        choices=["True", "true", "TRUE", "False", "false", "FALSE"],
                        default="False")

    args = parser.parse_args()
    env = args.env
    use_gpu = args.use_gpu
    use_autoscaler = args.use_autoscaler
    n_workers = args.n_workers

    if use_autoscaler.lower() == "true":
        use_autoscaler = True
    elif use_autoscaler.lower() == "false":
        use_autoscaler = False

    if use_gpu.lower() == "true":
        use_gpu = True
    elif use_gpu.lower() == "false":
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    if use_gpu is True:
        n_gpus_per_node = 1
    else:
        n_gpus_per_node = 0

    n_cpus_per_node = get_n_cpus_per_node()

    init_ray(n_gpus_per_node=n_gpus_per_node,
             n_cpus_per_node=n_cpus_per_node,
             autoscale=use_autoscaler)

    config = env_vars[env]
    main(config, use_gpu)
