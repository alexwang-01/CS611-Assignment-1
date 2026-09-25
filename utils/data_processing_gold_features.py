from datetime import datetime

import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType


def process_features_gold_table(
    snapshot_date_str,
    silver_attributes_directory,
    silver_financials_directory,
    silver_application_clickstream_directory,
    gold_feature_store_directory,
    spark,
):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver tables
    partition_date = snapshot_date_str.replace('-', '_')
    attributes_filepath = silver_attributes_directory + 'silver_attributes_' + partition_date + '.parquet'
    financials_filepath = silver_financials_directory + 'silver_financials_' + partition_date + '.parquet'
    application_filepath = silver_application_clickstream_directory + 'silver_application_clickstream_' + partition_date + '.parquet'

    attributes_df = spark.read.parquet(attributes_filepath).select('Customer_ID', 'snapshot_date', 'Age')
    financials_df = spark.read.parquet(financials_filepath)
    application_df = spark.read.parquet(application_filepath)
    print('loaded from:', attributes_filepath, 'row count:', attributes_df.count())
    print('loaded from:', financials_filepath, 'row count:', financials_df.count())
    print('loaded from:', application_filepath, 'row count:', application_df.count())

    # join the Silver application row with same-day attributes and financials
    df = application_df.join(attributes_df, on=['Customer_ID', 'snapshot_date'], how='left')
    df = df.join(financials_df, on=['Customer_ID', 'snapshot_date'], how='left')
    df = df.withColumnRenamed('snapshot_date', 'feature_snapshot_date')

    # add features known at application time
    df = df.withColumn(
        'debt_to_income_ratio',
        F.when(col('Annual_Income') > 0, col('Outstanding_Debt') / col('Annual_Income')).cast(FloatType()),
    )
    df = df.withColumn(
        'emi_to_income_ratio',
        F.when(col('Monthly_Inhand_Salary') > 0, col('Total_EMI_per_month') / col('Monthly_Inhand_Salary')).cast(FloatType()),
    )
    df = df.withColumn(
        'monthly_balance_ratio',
        F.when(col('Monthly_Inhand_Salary') > 0, col('Monthly_Balance') / col('Monthly_Inhand_Salary')).cast(FloatType()),
    )
    df = df.withColumn(
        'delayed_payment_flag',
        F.when(col('Num_of_Delayed_Payment').isNull(), F.lit(None))
        .when(col('Num_of_Delayed_Payment') > 0, 1)
        .otherwise(0)
        .cast(IntegerType()),
    )

    # select columns to save; exclude Name, SSN, and later loan outcomes
    feature_columns = [
        'loan_id', 'Customer_ID', 'feature_snapshot_date', 'clickstream_snapshot_date',
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance',
        'debt_to_income_ratio', 'emi_to_income_ratio',
        'monthly_balance_ratio', 'delayed_payment_flag',
        'fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5',
        'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10',
        'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15',
        'fe_16', 'fe_17', 'fe_18', 'fe_19', 'fe_20',
    ]
    df = df.select(*feature_columns)

    # save gold table - IRL connect to database to write
    partition_name = 'gold_feature_store_' + partition_date + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath)

    return df
