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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window


def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Remove source formatting such as 40_ before casting Age to IntegerType.
    df = df.withColumn("Age", F.regexp_replace(col("Age").cast(StringType()), "_", ""))

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Keep positive ages, including the observed 14-17 range; remove implausible outliers.
    df = df.withColumn("Age", F.when(col("Age").between(1, 120), col("Age")).otherwise(F.lit(None)))
    df = df.withColumn("Occupation", F.when(F.trim(col("Occupation")).rlike(r"^_+$"), F.lit(None)).otherwise(F.trim(col("Occupation"))))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Remove underscore formatting before casting numeric fields; double-underscore values are placeholders.
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
    ]
    for column in numeric_columns:
        raw_value = F.trim(col(column).cast(StringType()))
        df = df.withColumn(column, F.when(raw_value.rlike(r"^__.*__$"), F.lit(None)).otherwise(F.regexp_replace(raw_value, "_", "")))

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Counts, income, debt, and payment amounts cannot be negative.
    # Monthly_Balance is excluded because an overdrawn balance can be negative.
    non_negative_columns = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
        "Credit_Utilization_Ratio",
        "Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
        "Total_EMI_per_month", "Amount_invested_monthly",
    ]
    for column in non_negative_columns:
        df = df.withColumn(column, F.when(col(column) >= 0, col(column)).otherwise(F.lit(None)))

    # A payment before the due date has no days of delay.
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date") < 0, F.lit(0)).otherwise(col("Delay_from_due_date")))

    # Source placeholders in categorical columns become null.
    df = df.withColumn("Credit_Mix", F.when(F.trim(col("Credit_Mix")) == "_", F.lit(None)).otherwise(F.trim(col("Credit_Mix"))))
    df = df.withColumn("Payment_of_Min_Amount", F.when(F.trim(col("Payment_of_Min_Amount")) == "NM", F.lit(None)).otherwise(F.trim(col("Payment_of_Min_Amount"))))
    df = df.withColumn("Payment_Behaviour", F.when(F.trim(col("Payment_Behaviour")) == "!@9#%8", F.lit(None)).otherwise(F.trim(col("Payment_Behaviour"))))

    # augment data: convert credit history age to a numeric month count
    credit_years = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType())
    credit_months = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Age_Months", credit_years * F.lit(12) + credit_months)

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name

    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
        "fe_1": FloatType(),
        "fe_2": FloatType(),
        "fe_3": FloatType(),
        "fe_4": FloatType(),
        "fe_5": FloatType(),
        "fe_6": FloatType(),
        "fe_7": FloatType(),
        "fe_8": FloatType(),
        "fe_9": FloatType(),
        "fe_10": FloatType(),
        "fe_11": FloatType(),
        "fe_12": FloatType(),
        "fe_13": FloatType(),
        "fe_14": FloatType(),
        "fe_15": FloatType(),
        "fe_16": FloatType(),
        "fe_17": FloatType(),
        "fe_18": FloatType(),
        "fe_19": FloatType(),
        "fe_20": FloatType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_silver_application_clickstream(snapshot_date_str, silver_loan_daily_directory, silver_clickstream_directory, silver_application_clickstream_directory, spark):
    # connect to Silver loan table and keep the application row
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    loan_df = spark.read.parquet(filepath)
    loan_df = loan_df.filter(col("mob") == 0).select("loan_id", "Customer_ID", "snapshot_date")
    print('loaded from:', filepath, 'application row count:', loan_df.count())

    # read all Silver clickstream months; the join below excludes future rows
    filepath = silver_clickstream_directory + 'silver_clickstream_*.parquet'
    clickstream_df = spark.read.parquet(filepath)
    clickstream_df = clickstream_df.withColumnRenamed("Customer_ID", "clickstream_customer_id")
    clickstream_df = clickstream_df.withColumnRenamed("snapshot_date", "clickstream_snapshot_date")

    # keep all loans, then choose the latest eligible clickstream row per loan
    df = loan_df.join(
        clickstream_df,
        (loan_df["Customer_ID"] == clickstream_df["clickstream_customer_id"])
        & (clickstream_df["clickstream_snapshot_date"] <= loan_df["snapshot_date"]),
        how="left",
    ).drop("clickstream_customer_id")
    latest_first = Window.partitionBy("loan_id").orderBy(col("clickstream_snapshot_date").desc_nulls_last())
    df = df.withColumn("row_number", F.row_number().over(latest_first))
    df = df.filter(col("row_number") == 1).drop("row_number")

    # save the Silver application-time table; Gold will join it later
    partition_name = "silver_application_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_application_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())

    return df
