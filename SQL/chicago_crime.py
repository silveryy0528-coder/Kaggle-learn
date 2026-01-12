#%%
from google.cloud import bigquery
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


client = bigquery.Client()

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
table_ref = dataset_ref.table("crime")
table = client.get_table(table_ref)

#%%
client.list_rows(table, max_results=5).to_dataframe()
table.schema
#%%
# What categories of crime exhibited the greatest year-over-year increase between 2015 and 2016?
# Consider only crimes that resulted in an arrest, and limit your results to categories that saw
# at least 50 more incidents in 2016 than in 2015.
query_crime_increase = """
WITH crime_counts_2015 AS (
    SELECT
        primary_type,
        description,
        COUNT(*) AS num_crimes
    FROM `bigquery-public-data.chicago_crime.crime`
    WHERE
        year = 2015
        AND arrest = TRUE
    GROUP BY primary_type, description
),
crime_counts_2016 AS (
    SELECT
        primary_type,
        description,
        COUNT(*) AS num_crimes
    FROM `bigquery-public-data.chicago_crime.crime`
    WHERE
        year = 2016
        AND arrest = TRUE
    GROUP BY primary_type, description
)
SELECT
    c2015.primary_type,
    c2015.description,
    c2015.num_crimes AS num_crimes_2015,
    c2016.num_crimes AS num_crimes_2016,
    SAFE_DIVIDE(c2016.num_crimes - c2015.num_crimes, c2015.num_crimes) * 100 AS crime_increase_percent
FROM crime_counts_2015 AS c2015
INNER JOIN crime_counts_2016 AS c2016
    ON c2015.primary_type = c2016.primary_type
    AND c2015.description = c2016.description
WHERE c2016.num_crimes - c2015.num_crimes > 50
ORDER BY crime_increase_percent DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_crime_increase, job_config=safe_config)

results = query_job.result().to_dataframe()
results.head(10)

#%%
# For each year on record, which month saw the highest number of reported motor vehicle thefts?
query_motor_theft = """
WITH motor_vehicle_thefts AS (
    SELECT
        year,
        EXTRACT(MONTH FROM date) AS month,
        COUNT(*) AS num_thefts
    FROM `bigquery-public-data.chicago_crime.crime`
    WHERE primary_type = 'MOTOR VEHICLE THEFT'
    GROUP BY year, month
),
ranked_months AS (
    SELECT
        year,
        month,
        num_thefts,
        ROW_NUMBER() OVER (
            PARTITION BY year
            ORDER BY num_thefts DESC
        ) AS month_rank
    FROM motor_vehicle_thefts
)
SELECT
    year,
    month,
    num_thefts
FROM ranked_months
WHERE month_rank = 1
ORDER BY year DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_motor_theft, job_config=safe_config)

results = query_job.result().to_dataframe()
results.head(15)

#%%
query_violent = """
SELECT
    year,
    EXTRACT (MONTH FROM date) AS month,
    COUNT(*) AS num_violent_crimes
FROM `bigquery-public-data.chicago_crime.crime`
WHERE
    primary_type IN ('ASSAULT', 'BATTERY')
    AND year < 2026
GROUP BY year, month
ORDER BY year ASC, month ASC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_violent, job_config=safe_config)

results = query_job.result().to_dataframe()

#%%
violent_by_month = results.pivot(
    index='month', columns='year', values='num_violent_crimes')
violent_month_average = (
    results
    .groupby('month')['num_violent_crimes'].mean()
    .reset_index(name='average')
)
plt.figure(figsize=(12, 6))
sns.lineplot(data=violent_by_month)
sns.lineplot(
    x=violent_month_average['month'], y=violent_month_average['average'],
    color='black', label='Average', linewidth=2.5)