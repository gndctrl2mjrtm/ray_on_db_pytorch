import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
from ray.data import read_parquet
from ray.data.datasource import DefaultParquetMetadataProvider

from pyspark.sql import SparkSession


def delta_to_raydata(table_name,
                     filesystem=None,
                     columns=None,
                     parallelism=-1,
                     ray_remote_args=None,
                     tensor_column_schema=None,
                     meta_provider=DefaultParquetMetadataProvider(),
                     **arrow_parquet_args):
    """

    @param table_name:
    @param filesystem:
    @param columns:
    @param parallelism:
    @param ray_remote_args:
    @param tensor_column_schema:
    @param meta_provider:
    @param arrow_parquet_args:
    @return:
    """
    spark = SparkSession.builder.getOrCreate()
    files = spark.table(table_name).inputFiles()
    return read_parquet(
        paths=files,
        filesystem=filesystem,
        columns=columns,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        tensor_column_schema=tensor_column_schema,
        meta_provider=meta_provider,
        **arrow_parquet_args,
    )


def init_ray_on_spark(n_workers=ray.util.spark.MAX_NUM_WORKER_NODES,
                      n_cpus_per_node=4,
                      n_gpus_per_node=0,
                      log_path="/dbfs/path/to/ray_collected_logs"):
    """

    @param n_workers:
    @param n_cpus_per_node:
    @param n_gpus_per_node:
    @param log_path:
    @return:
    """
    setup_ray_cluster(
        num_worker_nodes=n_workers,
        num_cpus_per_node=n_cpus_per_node,
        num_gpus_per_node=n_gpus_per_node,
        collect_log_to_path=log_path
    )
    ray.init()


def init_ray_locally(address="auto"):
    """

    @param address:
    @return:
    """
    ray.init(address=address)


def init_ray(run_on_spark=True,
             n_workers=None,
             n_cpus_per_node=4,
             n_gpus_per_node=0,
             log_path="/dbfs/path/to/ray_collected_logs",
             address="auto"):
    """

    @param run_on_spark:
    @param n_workers:
    @param n_cpus_per_node:
    @param n_gpus_per_node:
    @param log_path:
    @param address:
    """
    if run_on_spark is True:
        if n_workers is None:
            n_workers = ray.util.spark.MAX_NUM_WORKER_NODES
        init_ray_on_spark(n_workers=n_workers,
                          n_cpus_per_node=n_cpus_per_node,
                          n_gpus_per_node=n_gpus_per_node,
                          log_path=log_path)
    else:
        init_ray_locally(address=address)
