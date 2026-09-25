import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

### Feature and model dataset imports
import utils.data_processing_bronze_features
import utils.data_processing_silver_features
import utils.data_processing_gold_features
import utils.data_processing_gold_model_dataset


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
### Loan data through November 2025 for MOB 6 labels
end_date_str = "2025-11-01"
### Feature data ends in January 2025; clickstream ends in December 2024
feature_end_date_str = "2025-01-01"
clickstream_end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

### Months available for feature tables
feature_dates_str_lst = generate_first_of_month_dates(start_date_str, feature_end_date_str)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_lms_directory, spark)

### Bronze feature tables
bronze_attributes_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)

for date_str in feature_dates_str_lst:
    utils.data_processing_bronze_features.process_bronze_attributes(
        date_str, bronze_attributes_directory, spark
    )
    utils.data_processing_bronze_features.process_bronze_financials(
        date_str, bronze_financials_directory, spark
    )
    if date_str <= clickstream_end_date_str:
        utils.data_processing_bronze_features.process_bronze_clickstream(
            date_str, bronze_clickstream_directory, spark
        )


# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

### Silver feature tables
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"

if not os.path.exists(silver_attributes_directory):
    os.makedirs(silver_attributes_directory)

if not os.path.exists(silver_financials_directory):
    os.makedirs(silver_financials_directory)

if not os.path.exists(silver_clickstream_directory):
    os.makedirs(silver_clickstream_directory)

for date_str in feature_dates_str_lst:
    utils.data_processing_silver_features.process_silver_attributes(
        date_str, bronze_attributes_directory, silver_attributes_directory, spark
    )
    utils.data_processing_silver_features.process_silver_financials(
        date_str, bronze_financials_directory, silver_financials_directory, spark
    )
    if date_str <= clickstream_end_date_str:
        utils.data_processing_silver_features.process_silver_clickstream(
            date_str, bronze_clickstream_directory, silver_clickstream_directory, spark
        )

### Align clickstream with each loan application
silver_application_clickstream_directory = "datamart/silver/application_clickstream/"

if not os.path.exists(silver_application_clickstream_directory):
    os.makedirs(silver_application_clickstream_directory)

for date_str in feature_dates_str_lst:
    utils.data_processing_silver_features.process_silver_application_clickstream(
        date_str,
        silver_loan_daily_directory,
        silver_clickstream_directory,
        silver_application_clickstream_directory,
        spark,
    )


# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

### Gold feature store
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

for date_str in feature_dates_str_lst:
    utils.data_processing_gold_features.process_features_gold_table(
        date_str,
        silver_attributes_directory,
        silver_financials_directory,
        silver_application_clickstream_directory,
        gold_feature_store_directory,
        spark,
    )

### Gold model dataset
gold_model_dataset_directory = "datamart/gold/model_dataset/"

if not os.path.exists(gold_model_dataset_directory):
    os.makedirs(gold_model_dataset_directory)

utils.data_processing_gold_model_dataset.process_model_dataset_gold_table(
    gold_feature_store_directory,
    gold_label_store_directory,
    gold_model_dataset_directory,
    spark,
)



    
