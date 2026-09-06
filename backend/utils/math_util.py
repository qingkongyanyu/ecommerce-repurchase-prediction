# TODO: 数据统计计算工具函数
# backend/utils/math_util.py
"""
数学和统计辅助函数：计算分类分布、加权统计、复购率等。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

def calc_distribution(series: pd.Series, normalize: bool = True) -> Dict[Any, float]:
    """
    计算分类变量的分布（频数或频率）
    :param series: 分类系列
    :param normalize: True返回频率(0~1)，False返回计数
    :return: 字典 {类别: 比例/计数}
    """
    counts = series.value_counts()
    if normalize:
        return (counts / counts.sum()).to_dict()
    return counts.to_dict()

def calc_repeat_rate(df: pd.DataFrame, group_col: str, label_col: str = 'repurchase_label') -> Dict[Any, float]:
    """
    按分组计算复购率
    :param df: DataFrame
    :param group_col: 分组列名（如'member_level'）
    :param label_col: 标签列名，假设1为复购
    :return: 字典 {组名: 复购率}
    """
    return df.groupby(group_col)[label_col].mean().to_dict()

def safe_divide(a: Union[float, int, np.ndarray], b: Union[float, int, np.ndarray], fill_value: float = 0.0) -> np.ndarray:
    """安全除法，避免除零错误"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result = np.where(np.isfinite(result), result, fill_value)
    return result

def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    归一化数组
    :param arr: 输入数组
    :param method: 'minmax' (0~1) 或 'zscore'
    """
    arr = np.asarray(arr, dtype=float)
    if method == 'minmax':
        min_val = arr.min()
        max_val = arr.max()
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")

def compute_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    计算二分类评估指标（准确率、精确率、召回率、F1、AUC）
    :param y_true: 真实标签 (1D)
    :param y_pred: 预测标签 (1D)
    :param y_prob: 预测概率（正类概率），若提供则计算AUC
    :return: 指标字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = np.nan
    return metrics