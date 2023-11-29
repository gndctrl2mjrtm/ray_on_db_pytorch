username = "soffer"

data_name = "cola"
catalog_prefix = f"{username}"
schema = "raydemo"
train_table_default_name = f"{data_name}_training_data"
val_table_default_name = f"{data_name}_validation_data"

env_vars = {
    "dev": {
        "uc_catalog": f"{catalog_prefix}_dev",
        "schema": schema,
        "training_data": train_table_default_name,
        "validation_data": val_table_default_name
    },
    "tst": {
        "uc_catalog": f"{catalog_prefix}_tst",
        "schema": schema,
        "training_data": train_table_default_name,
        "validation_data": val_table_default_name
    },
    "stg": {
        "uc_catalog": f"{catalog_prefix}_stg",
        "schema": schema,
        "training_data": train_table_default_name,
        "validation_data": val_table_default_name
    },
    "prd": {
        "uc_catalog": f"{catalog_prefix}_prd",
        "schema": schema,
        "training_data": train_table_default_name,
        "validation_data": val_table_default_name
    }
}
