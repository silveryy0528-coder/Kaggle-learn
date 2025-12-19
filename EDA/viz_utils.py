import pandas as pd
import os


def path_join(dir_path, file_name):
    return os.path.join(dir_path, file_name)


def summarize_dataframe(df, dataset_name=None):
    summary = pd.DataFrame({
        "feature_name": df.columns,
        "amount_null": df.isna().sum().values,
        "percent_null": (df.isna().mean() * 100).values,
        "dtype": df.dtypes.astype(str).values,
        "total_cat_entries": [
            df[col].nunique() if df[col].dtype == "object" else 0
            for col in df.columns
        ]
    })
    if dataset_name is not None:
        summary.insert(0, 'dataset_name', dataset_name)

    return summary


def summarize_datasets(datasets):
    summary_lists = []
    for name, df in datasets.items():
        summary_lists.append(summarize_dataframe(df, name))

    final_summary = pd.concat(summary_lists, ignore_index=True)
    return final_summary
