#%%
from google.cloud import bigquery


client = bigquery.Client()

dataset_ref = client.dataset("google_analytics_sample", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
# tables = list(client.list_tables(dataset))
# for table in tables:
#     print(table.table_id)
table_ref = dataset_ref.table("ga_sessions_20170701")
table = client.get_table(table_ref)

#%%
client.list_rows(table, max_results=5).to_dataframe()
table.schema

#%%
# What is the total number of transactions generated per device browser
# in July 2017
query_transactions_per_device = """
SELECT
    device.browser AS device_browser,
    SUM(totals.transactions) AS num_transactions
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
GROUP BY device.browser
ORDER BY num_transactions DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_transactions_per_device, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_bounce_rate = """
WITH visits_vs_bounces AS (
    SELECT
        trafficSource.source AS traffic_source,
        COUNT(trafficSource.source) AS total_visits,
        SUM(totals.bounces) AS total_bounces
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
    GROUP BY traffic_source
)
SELECT
    traffic_source,
    total_visits,
    total_bounces,
    SAFE_DIVIDE(total_bounces, total_visits) * 100 AS bounce_rate
FROM visits_vs_bounces
ORDER BY total_visits DESC
"""
query_job = client.query(query=query_bounce_rate, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head()

#%%
# What was the average number of product pageviews for users who made a
# purchase in July 2017
query_pageviews = """
WITH session_hits AS (
    SELECT
        fullVisitorID,
        visitId,
        COUNTIF(hits.eCommerceAction.action_type = '2') AS product_pageviews,
        MAX(totals.transactions) AS transactions
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
        UNNEST(hits) AS hits
    WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
    GROUP BY fullVisitorID, visitId
),
pageviews AS (
    SELECT
        fullVisitorID,
        SUM(product_pageviews) AS product_pageviews,
        SUM(transactions) AS total_transactions
    FROM session_hits
    GROUP BY fullVisitorID
)
SELECT
    SAFE_DIVIDE(SUM(product_pageviews), COUNT(1)) AS average_product_views
FROM pageviews
WHERE total_transactions >= 1   # if total_transactions IS NULL, then it considers non-purchase users
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_pageviews, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
# What was the average total transactions per user that made a purchase in July 2017
query_transactions = """
WITH transactions AS (
    SELECT
        fullVisitorID,
        COUNT(DISTINCT visitId) AS num_visits,
        SUM(totals.transactions) AS transaction_per_user
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
    GROUP BY fullVisitorID
)
SELECT AVG(transaction_per_user) AS average_total_transaction
FROM transactions
WHERE transaction_per_user >= 1
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_transactions, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

# %%
# What is the average amount of money spent per session in July 2017?
query_money_per_session = """
SELECT
    SUM(totals.totalTransactionRevenue) / 1e6 AS total_revenue,
    COUNT(1) AS num_sessions,
    SAFE_DIVIDE(SUM(totals.totalTransactionRevenue), COUNT(1)) / 1e6 AS average_spend
    # Since some TransactionRevenue is NULL, so we cannot directly use AVG function.
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_money_per_session, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(5)

# %%
# What is the sequence of pages viewed?
query_sequence = """
SELECT
    fullVisitorID,
    visitId,
    ARRAY_TO_STRING(
        ARRAY_AGG(
            hits.page.pagePath
            ORDER BY hits.hitNumber),
        '->') AS page_path
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
    UNNEST(hits) AS hits
WHERE
    _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
    AND hits.type = 'PAGE'
GROUP BY fullVisitorID, visitId
ORDER BY fullVisitorID
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_sequence, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(5)