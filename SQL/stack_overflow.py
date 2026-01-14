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
query_answers = """
WITH questions_with_answers AS (
    SELECT
        q.id AS question_id,
        q.creation_date AS asked_date,
        a.id AS answer_id,
        a.creation_date AS answered_date,
        TIMESTAMP_DIFF(a.creation_date, q.creation_date, MINUTE) AS elapsed_minutes,
        ROW_NUMBER() OVER (
            PARTITION BY q.id
            ORDER BY a.creation_date
        ) AS answer_rank
    FROM `bigquery-public-data.stackoverflow.posts_questions` as q
    INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` as a
        ON q.id = a.parent_id
),
answered_in_one_hour AS (
    SELECT
        question_id,
        EXTRACT(DAYOFWEEK FROM asked_date) AS day_of_week,
        CASE EXTRACT (DAYOFWEEK FROM asked_date)
            WHEN 1 THEN 'Sunday'
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
        END AS day_name,
        answer_id AS first_answer_id,
        elapsed_minutes
    FROM questions_with_answers
    WHERE
        answer_rank = 1
        AND elapsed_minutes > 0
        AND elapsed_minutes <= 60
)
SELECT
    day_of_week,
    day_name,
    COUNT(DISTINCT question_id) AS num_questions
FROM answered_in_one_hour
GROUP BY day_of_week, day_name
ORDER BY num_questions DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_answers, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)
#%%
"""
SELECT
    parent_id AS question_id,
    COUNT(DISTINCT id) AS num_answers
FROM `bigquery-public-data.stackoverflow.posts_answers`
GROUP BY parent_id
ORDER BY num_answers DESC
LIMIT 10"""


"""
SELECT
    id AS answer_id,
    creation_date AS answered_date
FROM `bigquery-public-data.stackoverflow.posts_answers`
WHERE parent_id = 184618
ORDER BY answered_date
LIMIT 10
"""