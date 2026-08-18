"""
Model definitions and Cross-Validation training routines for Kaggle Tabular Playground s6e8.
Supports LightGBM, XGBoost, CatBoost (if installed) as well as Scikit-learn native models
(HistGradientBoosting, Random Forest, Extra Trees, Logistic Regression).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Optional GBDT libraries (installed via requirements.txt)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_lgbm_default_params():
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 3000,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }


def get_xgb_default_params():
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 5,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 3000,
        'random_state': 42,
        'enable_categorical': True,
        'n_jobs': -1
    }


def get_catboost_default_params():
    return {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 3000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3.0,
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1
    }


def get_hist_gb_default_params():
    return {
        'learning_rate': 0.03,
        'max_iter': 500,
        'max_leaf_nodes': 31,
        'min_samples_leaf': 20,
        'l2_regularization': 1.0,
        'random_state': 42
    }


def train_cv(model_type, train_df, test_df=None, target_col='addicted_label', 
             id_col='id', n_splits=5, params=None, random_state=42):
    """
    Train a specified classification model using Stratified K-Fold Cross Validation.
    Returns Out-Of-Fold probabilities, test set probabilities, and overall ROC-AUC score.
    """
    feature_cols = [col for col in train_df.columns if col not in [id_col, target_col]]
    
    # Identify categorical columns
    cat_cols = [col for col in feature_cols if str(train_df[col].dtype) == 'category' or train_df[col].dtype == object]
    cat_indices = [feature_cols.index(col) for col in cat_cols]

    X = train_df[feature_cols].copy()
    y = train_df[target_col].values
    X_test = test_df[feature_cols].copy() if test_df is not None else None

    # Handle category formatting depending on model requirements
    if model_type == 'catboost' and HAS_CATBOOST:
        for col in cat_cols:
            X[col] = X[col].astype(str).fillna('missing')
            if X_test is not None:
                X_test[col] = X_test[col].astype(str).fillna('missing')
    elif model_type in ['rf', 'extra_trees', 'lr']:
        # Keep only numeric features (since target encoding and frequency encoding already capture categorical info)
        num_cols = [c for c in feature_cols if c not in cat_cols]
        X = X[num_cols]
        if X_test is not None:
            X_test = X_test[num_cols]
    elif model_type == 'hist_gb':
        # HistGradientBoosting supports categorical variables via cat.codes
        for col in cat_cols:
            X[col] = X[col].astype('category').cat.codes
            if X_test is not None:
                X_test[col] = X_test[col].astype('category').cat.codes

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df)) if test_df is not None else None
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    
    print(f"\n=================== Training {model_type.upper()} ({n_splits}-Fold CV) ===================")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        if model_type == 'lgbm':
            if not HAS_LGBM:
                raise ImportError("LightGBM is not installed. Please run `pip install lightgbm` or use `hist_gb`.")
            model_params = params or get_lgbm_default_params()
            model = lgb.LGBMClassifier(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]

        elif model_type == 'xgb':
            if not HAS_XGB:
                raise ImportError("XGBoost is not installed. Please run `pip install xgboost` or use `hist_gb`.")
            model_params = params or get_xgb_default_params()
            model = xgb.XGBClassifier(**model_params, early_stopping_rounds=100)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]

        elif model_type == 'catboost':
            if not HAS_CATBOOST:
                raise ImportError("CatBoost is not installed. Please run `pip install catboost` or use `hist_gb`.")
            model_params = params or get_catboost_default_params()
            model = CatBoostClassifier(**model_params, early_stopping_rounds=100)
            train_pool = Pool(X_train, y_train, cat_features=cat_indices)
            val_pool = Pool(X_val, y_val, cat_features=cat_indices)
            model.fit(train_pool, eval_set=val_pool, verbose=False)
            val_pred = model.predict_proba(val_pool)[:, 1]
            if X_test is not None:
                test_pool = Pool(X_test, cat_features=cat_indices)
                fold_test_pred = model.predict_proba(test_pool)[:, 1]

        elif model_type == 'hist_gb':
            model_params = params or get_hist_gb_default_params()
            model = HistGradientBoostingClassifier(**model_params)
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]

        elif model_type == 'rf':
            model = make_pipeline(
                SimpleImputer(strategy='median'),
                RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
            )
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]

        elif model_type == 'extra_trees':
            model = make_pipeline(
                SimpleImputer(strategy='median'),
                ExtraTreesClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
            )
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]

        elif model_type == 'lr':
            model = make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            )
            model.fit(X_train, y_train)
            val_pred = model.predict_proba(X_val)[:, 1]
            if X_test is not None:
                fold_test_pred = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        oof_preds[val_idx] = val_pred
        if test_preds is not None:
            test_preds += fold_test_pred / n_splits
            
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    overall_auc = roc_auc_score(y, oof_preds)
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    print(f"[{model_type.upper()}] Overall OOF ROC-AUC: {overall_auc:.5f} | Fold Mean: {mean_auc:.5f} (+/- {std_auc:.5f})")
    
    return oof_preds, test_preds, overall_auc
