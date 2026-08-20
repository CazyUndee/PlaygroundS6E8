"""
Ensemble methods for Kaggle Tabular Playground s6e8:
- Optimized Linear Blending (maximizing OOF ROC AUC)
- Rank Averaging
- Stacking with Logistic Regression
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def optimize_weights(oof_dict, y_true):
    """
    Find optimal linear blend weights to maximize Out-Of-Fold ROC-AUC score.
    `oof_dict`: dict of {model_name: oof_predictions_array}
    """
    model_names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[name] for name in model_names])
    n_models = len(model_names)

    def loss_func(weights):
        # Normalize weights to sum to 1
        w = np.array(weights)
        w = w / np.sum(w)
        blend_pred = np.dot(oof_matrix, w)
        # Minimize negative ROC AUC
        return -roc_auc_score(y_true, blend_pred)

    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

    res = minimize(
        loss_func,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    best_weights = res.x / np.sum(res.x)
    best_auc = -res.fun
    
    print("\n================= Optimized Blending Weights =================")
    for name, w in zip(model_names, best_weights):
        print(f"  {name:15s}: {w:.4f}")
    print(f"Optimized Blend OOF ROC-AUC: {best_auc:.5f}")

    return {name: w for name, w in zip(model_names, best_weights)}, best_auc


def rank_average_ensemble(pred_dict):
    """
    Perform rank averaging across model predictions.
    Since ROC-AUC is rank-based, rank averaging is highly robust against scale differences.
    """
    model_names = list(pred_dict.keys())
    n_samples = len(pred_dict[model_names[0]])
    ranked_sum = np.zeros(n_samples)
    
    for name, preds in pred_dict.items():
        ranked_sum += rankdata(preds) / n_samples
        
    return ranked_sum / len(model_names)


def stack_models(oof_dict, test_dict, y_true, n_splits=5, random_state=42):
    """
    Stack models using Logistic Regression with StratifiedKFold OOF prediction.
    """
    model_names = list(oof_dict.keys())
    X_oof = np.column_stack([oof_dict[name] for name in model_names])
    X_test = np.column_stack([test_dict[name] for name in model_names])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stacked_oof = np.zeros(len(y_true))
    stacked_test = np.zeros(len(X_test))
    
    print("\n================= Meta-Model Stacking (Logistic Regression) =================")
    for train_idx, val_idx in skf.split(X_oof, y_true):
        meta_model = LogisticRegression(C=1.0, max_iter=1000)
        meta_model.fit(X_oof[train_idx], y_true[train_idx])
        stacked_oof[val_idx] = meta_model.predict_proba(X_oof[val_idx])[:, 1]
        stacked_test += meta_model.predict_proba(X_test)[:, 1] / n_splits

    stack_auc = roc_auc_score(y_true, stacked_oof)
    print(f"Stacked OOF ROC-AUC: {stack_auc:.5f}")
    
    return stacked_oof, stacked_test, stack_auc
