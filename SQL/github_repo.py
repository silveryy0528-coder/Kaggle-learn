#%%
from google.cloud import bigquery


client = bigquery.Client()
dataset_ref = client.dataset("github_repos", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:
    print(table.table_id)

#%%
table_ref = dataset_ref.table("contents")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
# table.schema

#%%
query_languages = """
WITH repo_languages AS (
    SELECT
        repo_name,
        l.name AS language_name
    FROM `bigquery-public-data.github_repos.languages`,
        UNNEST(language) AS l
)
SELECT
    language_name,
    COUNT(DISTINCT repo_name) AS num_repos
FROM repo_languages
GROUP BY language_name
ORDER BY num_repos DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_languages, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_primary_language = """
WITH repo_languages AS (
    SELECT
        repo_name,
        l.name AS language_name,
        l.bytes AS language_bytes,
        ROW_NUMBER() OVER(
            PARTITION BY repo_name
            ORDER BY l.bytes DESC, l.name
    ) AS row_num
    FROM `bigquery-public-data.github_repos.languages`
    CROSS JOIN UNNEST(language) AS l
)
SELECT
    repo_name,
    language_name AS primary_language,
    language_bytes AS bytes_used
FROM repo_languages
WHERE row_num = 1
LIMIT 100
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_primary_language, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_licenses = """
SELECT license, COUNT(dISTINCT repo_name) AS num_repos
FROM `bigquery-public-data.github_repos.licenses`
GROUP BY license
ORDER BY num_repos DESC
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_licenses, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_commits = """
SELECT
    repo_name,
    COUNT(DISTINCT commit) AS num_commits,
    COUNT(DISTINCT author.name) AS num_authors
FROM `bigquery-public-data.github_repos.sample_commits`
GROUP BY repo_name
ORDER BY num_commits DESC
LIMIT 10
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_commits, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_correlation = """
WITH commit_stats AS (
    SELECT
        repo_name,
        COUNT(DISTINCT commit) AS num_commits,
        COUNT(DISTINCT author.name) AS num_authors
    FROM `bigquery-public-data.github_repos.sample_commits`
    GROUP BY repo_name
)
SELECT
    CORR(num_authors, num_commits) AS author_commit_correlation
FROM commit_stats
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_correlation, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)

#%%
query_committer = """
WITH commit_years AS (
    SELECT
        repo_name,
        commit AS commit_id,
        EXTRACT(YEAR FROM author.date) AS commit_year
    FROM `bigquery-public-data.github_repos.sample_commits`
)
SELECT
    repo_name,
    commit_year,
    COUNT(DISTINCT commit_id) AS num_commits
FROM commit_years
GROUP BY repo_name, commit_year
ORDER BY repo_name, commit_year
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_committer, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(50)

#%%
query_file_size = """
WITH file_sizes AS (
    SELECT
        files.repo_name,
        files.id AS file_id,
        contents.size AS file_size_bytes
    FROM `bigquery-public-data.github_repos.sample_files` AS files
    INNER JOIN `bigquery-public-data.github_repos.sample_contents` AS contents
        ON files.id = contents.id
)
SELECT
    repo_name,
    SUM(file_size_bytes) AS total_file_size_bytes,
    COUNT(file_id) AS num_files,
    AVG(file_size_bytes) AS avg_file_size_bytes,
    MAX(file_size_bytes) AS max_file_size_bytes,
    MIN(file_size_bytes) AS min_file_size_bytes
FROM file_sizes
GROUP BY repo_name
ORDER BY total_file_size_bytes DESC
LIMIT 10
"""
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query=query_file_size, job_config=safe_config)
results = query_job.result().to_dataframe()
results.head(10)