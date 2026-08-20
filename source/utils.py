"""
Utility functions for data loading, memory optimization, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage decreased from {start_mem:.2f}MB to {end_mem:.2f}MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def evaluate_roc_auc(y_true, y_pred):
    """
    Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    """
    return roc_auc_score(y_true, y_pred)


def save_submission(ids, preds, filepath='submission.csv'):
    """
    Save predictions in the required Kaggle format:
    id,addicted_label
    """
    sub = pd.DataFrame({
        'id': ids,
        'addicted_label': np.clip(preds, 0.0, 1.0)
    })
    sub.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath} with {len(sub)} rows.")
    return sub
