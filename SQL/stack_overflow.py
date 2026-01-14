#%%
from google.cloud import bigquery


client = bigquery.Client()
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)

#%%
table_ref = dataset_ref.table("posts_questions")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()

#%%
# What is the percentage of questions that have been answered over the years
query_questions_answered = """
WITH question_yearly AS (
    SELECT
        EXTRACT(YEAR FROM creation_date) AS year,
        COUNT(1) AS total_questions_per_year,
        COUNTIF(answer_count > 0) AS answered_questions_per_year
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    GROUP BY year
)
SELECT
    *,
    SAFE_DIVIDE(answered_questions_per_year, total_questions_per_year) * 100 AS percent
FROM question_yearly
ORDER BY year
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_questions_answered, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
# What is the reputation and badge count of users across different tenures on StackOverflow?
query_users = """
WITH user_profiles AS (
    SELECT
        u.id AS user_id,
        DATE(u.creation_date) AS creation_date,
        FLOOR(
            DATE_DIFF(CURRENT_DATE(), DATE(u.creation_date), DAY) / 365.25
        ) AS tenure_years,
        u.reputation AS reputation,
        COUNT(DISTINCT b.id) AS num_badges
    FROM `bigquery-public-data.stackoverflow.users` AS u
    LEFT JOIN `bigquery-public-data.stackoverflow.badges` AS b
        ON u.id = b.user_id
    GROUP BY user_id, creation_date, tenure_years, reputation
)
SELECT
    tenure_years,
    COUNT(user_id) AS num_users,
    AVG(reputation) AS avg_reputation,  # AVG ignores NULL, so use COALESCE(variable, 0) first
    AVG(num_badges) AS avg_badges
FROM user_profiles
GROUP BY tenure_years
ORDER BY tenure_years
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_users, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
# What are 10 of the “easier” gold badges to earn?
query_gold_badges = """
SELECT
    name AS badge_name,
    class,
    COUNT(1) AS num_grants,
    COUNT(DISTINCT user_id) AS num_users_earned
FROM `bigquery-public-data.stackoverflow.badges`
WHERE class = 1
GROUP BY name, class
ORDER BY num_grants DESC
LIMIT 10
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_gold_badges, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
# Which day of the week has most questions answered within an hour?
query_answers = """
WITH all_questions_and_answers AS (
    SELECT
        q.id AS question_id,
        EXTRACT(DAYOFWEEK FROM q.creation_date) AS day_of_week,
        TIMESTAMP_DIFF(a.creation_date, q.creation_date, MINUTE) AS answered_in_minutes,
        ROW_NUMBER() OVER (
            PARTITION BY q.id
            ORDER BY a.creation_date
        ) AS answer_rank
    FROM `bigquery-public-data.stackoverflow.posts_questions` as q
    LEFT JOIN `bigquery-public-data.stackoverflow.posts_answers` as a
        ON q.id = a.parent_id
),
first_answer AS (
    SELECT
        question_id,
        day_of_week,
        answered_in_minutes,
        IF(answered_in_minutes BETWEEN 1 AND 60, TRUE, FALSE) AS answered_in_one_hour
    FROM all_questions_and_answers
    WHERE answer_rank = 1
)
SELECT
    day_of_week,
    COUNT(DISTINCT question_id) AS total_num_questions,
    COUNTIF(answered_in_one_hour) AS questions_answered_in_one_hour,
    COUNTIF(answered_in_one_hour) / COUNT(DISTINCT question_id) * 100 AS percent
FROM first_answer
GROUP BY day_of_week
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_answers, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)
