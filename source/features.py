"""
Feature Engineering module for Smartphone Addiction Tabular Playground Series (s6e8).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class FeatureEngineer:
    def __init__(self, target_col='addicted_label', id_col='id'):
        self.target_col = target_col
        self.id_col = id_col
        self.categorical_cols = []
        self.numerical_cols = []
        self.freq_enc_maps = {}
        self.target_enc_maps = {}

    def fit_transform(self, train_df, test_df=None):
        """
        Fit feature engineering transformations on train_df and apply to test_df.
        Uses out-of-fold target encoding to prevent data leakage.
        """
        train = train_df.copy()
        test = test_df.copy() if test_df is not None else None

        # Identify features
        feature_cols = [col for col in train.columns if col not in [self.id_col, self.target_col]]
        
        self.categorical_cols = [
            col for col in feature_cols 
            if train[col].dtype == object or str(train[col].dtype) == 'category' or train[col].nunique() < 15
        ]
        self.numerical_cols = [
            col for col in feature_cols 
            if col not in self.categorical_cols
        ]

        print(f"Detected {len(self.categorical_cols)} categorical/low-cardinality features and {len(self.numerical_cols)} numerical features.")

        # 1. Row-level statistical aggregations across numerical features
        if len(self.numerical_cols) >= 3:
            print("Creating row-level numerical aggregations...")
            for df in [train, test]:
                if df is None:
                    continue
                num_sub = df[self.numerical_cols].fillna(0)
                df['row_num_mean'] = num_sub.mean(axis=1)
                df['row_num_std'] = num_sub.std(axis=1)
                df['row_num_max'] = num_sub.max(axis=1)
                df['row_num_min'] = num_sub.min(axis=1)
                df['row_num_sum'] = num_sub.sum(axis=1)
                df['row_num_skew'] = num_sub.skew(axis=1)

        # 2. Key Interaction Features for continuous usage columns (ratios and products)
        # We automatically create ratios between top numerical features with high variance
        if len(self.numerical_cols) >= 2:
            print("Creating numerical interaction ratios...")
            top_num_cols = self.numerical_cols[:8]  # pick top numerical columns
            for i in range(len(top_num_cols)):
                for j in range(i + 1, len(top_num_cols)):
                    col1, col2 = top_num_cols[i], top_num_cols[j]
                    for df in [train, test]:
                        if df is None:
                            continue
                        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, np.nan) + 1e-5)
                        df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]

        # 3. Frequency Encoding for Categorical columns
        print("Applying frequency encoding...")
        for col in self.categorical_cols:
            freq_map = train[col].value_counts(normalize=True).to_dict()
            self.freq_enc_maps[col] = freq_map
            for df in [train, test]:
                if df is None:
                    continue
                df[f'{col}_freq_enc'] = df[col].map(freq_map).fillna(0)

        # 4. Out-of-fold Target Encoding for Categorical columns
        if self.categorical_cols and self.target_col in train.columns:
            print("Applying out-of-fold target encoding...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for col in self.categorical_cols:
                train[f'{col}_target_enc'] = np.nan
                for train_idx, val_idx in skf.split(train, train[self.target_col]):
                    tr_fold = train.iloc[train_idx]
                    val_fold = train.iloc[val_idx]
                    te_map = tr_fold.groupby(col)[self.target_col].mean()
                    train.iloc[val_idx, train.columns.get_loc(f'{col}_target_enc')] = val_fold[col].map(te_map)
                
                # Global mean fallback for missing categories
                global_mean = train[self.target_col].mean()
                train[f'{col}_target_enc'] = train[f'{col}_target_enc'].fillna(global_mean)
                
                # Fit map on full train for test set
                self.target_enc_maps[col] = train.groupby(col)[self.target_col].mean().to_dict()
                if test is not None:
                    test[f'{col}_target_enc'] = test[col].map(self.target_enc_maps[col]).fillna(global_mean)

        # Convert categorical columns to 'category' dtype for LightGBM/CatBoost compatibility
        for col in self.categorical_cols:
            train[col] = train[col].astype('category')
            if test is not None:
                test[col] = test[col].astype('category')

        print(f"Feature engineering complete. Total features: {train.shape[1] - 2}")
        return train, test
