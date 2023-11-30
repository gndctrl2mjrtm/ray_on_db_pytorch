from pyspark.sql import SparkSession
from pyspark.context import SparkContext

def get_dbutils(spark):
    try:
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
    except ImportError:
        import IPython
        dbutils = IPython.get_ipython().user_ns["dbutils"]
    return dbutils


def file_exists(path):
    spark = SparkSession.builder.getOrCreate()
    dbutils = get_dbutils(spark)
    try:
        dbutils.fs.ls(path)
        return True
    except Exception as e:
        return False


def get_n_cpus_per_node():
    spark = SparkSession.builder.getOrCreate()
    sc = SparkContext()
    return int(int(sc.defaultParallelism)/int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))

