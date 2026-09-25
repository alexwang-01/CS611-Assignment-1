def process_model_dataset_gold_table(gold_feature_store_directory, gold_label_store_directory, gold_model_dataset_directory, spark):
    # connect to gold feature and label tables
    feature_filepath = gold_feature_store_directory + "gold_feature_store_*.parquet"
    label_filepath = gold_label_store_directory + "gold_label_store_*.parquet"

    feature_df = spark.read.parquet(feature_filepath)
    label_df = spark.read.parquet(label_filepath).select("loan_id", "label")

    # combine features and label for each loan
    df = feature_df.join(label_df, on="loan_id", how="inner")

    # save gold model dataset
    partition_name = "gold_model_dataset.parquet"
    filepath = gold_model_dataset_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df
