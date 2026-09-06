# backend/api/screen_api.py
"""
大屏专用实时数据接口
提供轻量化统计指标，用于前端可视化大屏的实时刷新
特点：快速响应（<200ms），只聚合必要统计，不执行耗时计算
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from backend.config import SCREEN_PREFIX, DEBUG, DATASET_FILE
from backend.utils.file_util import load_csv
from backend.utils.math_util import calc_repeat_rate, calc_distribution, safe_divide

router = APIRouter(prefix=SCREEN_PREFIX, tags=["大屏实时数据"])

# 缓存大屏展示数据（避免重复加载）
_screen_df_cache = None
_last_cache_time = None


def get_cleaned_data() -> pd.DataFrame:
    """
    获取大屏展示数据（带缓存）
    说明：大屏展示的是业务统计口径，因此使用原始数据集、仅填充缺失值，
    不进行 z-score 标准化（否则用户分层、消费金额等统计将失去业务含义）。
    """
    global _screen_df_cache, _last_cache_time
    if _screen_df_cache is None:
        df = load_csv(DATASET_FILE)
        df = df.fillna(0)
        _screen_df_cache = df
        _last_cache_time = datetime.now()
    return _screen_df_cache


@router.get("/global")
async def get_global_metrics():
    """
    全局核心指标
    返回：总用户数、复购用户数、流失用户数、整体复购率、静默用户占比、VIP复购率等
    """
    try:
        df = get_cleaned_data()
        total = len(df)
        # 复购标签：1=复购，0=流失
        repurchase_count = df['repurchase_label'].sum()
        churn_count = total - repurchase_count
        repurchase_rate = repurchase_count / total if total > 0 else 0

        # 静默用户：silence_days > 30（假设原始数据中有该列）
        if 'silence_days' in df.columns:
            silence_count = (df['silence_days'] > 30).sum()
            silence_rate = silence_count / total if total > 0 else 0
        else:
            silence_count = 0
            silence_rate = 0

        # VIP复购率（is_vip=1）
        if 'is_vip' in df.columns and 'repurchase_label' in df.columns:
            vip_df = df[df['is_vip'] == 1]
            vip_repurchase_rate = vip_df['repurchase_label'].mean() if len(vip_df) > 0 else 0
            vip_count = len(vip_df)
        else:
            vip_repurchase_rate = 0
            vip_count = 0

        return {
            "total_users": total,
            "repurchase_users": int(repurchase_count),
            "churn_users": int(churn_count),
            "repurchase_rate": round(repurchase_rate, 4),
            "silence_users": int(silence_count),
            "silence_rate": round(silence_rate, 4),
            "vip_users": vip_count,
            "vip_repurchase_rate": round(vip_repurchase_rate, 4),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取全局指标失败: {str(e)}")


@router.get("/user-layer")
async def get_user_layer_distribution(
    group_by: str = Query("member_level", description="分组维度：member_level / city_tier / device_type")
):
    """
    用户分层复购率分布
    按指定分类维度计算各组的用户数和复购率
    """
    try:
        df = get_cleaned_data()
        if group_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"分组列 {group_by} 不存在")

        # 分组统计
        grouped = df.groupby(group_by).agg(
            total_users=('user_id', 'count'),
            repurchase_users=('repurchase_label', 'sum')
        )
        grouped['repurchase_rate'] = safe_divide(grouped['repurchase_users'], grouped['total_users'])

        # 转换为列表格式
        result = []
        for idx, row in grouped.iterrows():
            result.append({
                "group_value": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "total_users": int(row['total_users']),
                "repurchase_users": int(row['repurchase_users']),
                "repurchase_rate": round(row['repurchase_rate'], 4)
            })

        return {
            "group_by": group_by,
            "distribution": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户分层失败: {str(e)}")


@router.get("/consumption")
async def get_consumption_metrics():
    """
    消费指标统计
    返回：平均消费、优惠券使用率、高价商品占比、运费支出等
    """
    try:
        df = get_cleaned_data()
        metrics = {}

        # 平均消费（total_consume）
        if 'total_consume' in df.columns:
            metrics['avg_consume'] = round(df['total_consume'].mean(), 2)
            metrics['median_consume'] = round(df['total_consume'].median(), 2)

        # 优惠券使用率（有优惠券使用记录的用户占比）
        if 'coupon_use_cnt' in df.columns:
            coupon_users = (df['coupon_use_cnt'] > 0).sum()
            metrics['coupon_usage_rate'] = round(coupon_users / len(df), 4)

        # 高价商品占比均值
        if 'high_price_goods_ratio' in df.columns:
            metrics['avg_high_price_ratio'] = round(df['high_price_goods_ratio'].mean(), 4)

        # 运费支出均值
        if 'logistics_extra_cost' in df.columns:
            metrics['avg_logistics_cost'] = round(df['logistics_extra_cost'].mean(), 2)

        # 静默天数中位数
        if 'silence_days' in df.columns:
            metrics['median_silence_days'] = int(df['silence_days'].median())

        # 复购与非复购用户的平均消费对比
        if 'total_consume' in df.columns and 'repurchase_label' in df.columns:
            avg_repurchase = df[df['repurchase_label'] == 1]['total_consume'].mean()
            avg_non_repurchase = df[df['repurchase_label'] == 0]['total_consume'].mean()
            metrics['avg_consume_repurchase'] = round(avg_repurchase, 2) if not pd.isna(avg_repurchase) else 0
            metrics['avg_consume_non_repurchase'] = round(avg_non_repurchase, 2) if not pd.isna(avg_non_repurchase) else 0

        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消费指标失败: {str(e)}")


@router.get("/model-metrics")
async def get_model_metrics():
    """
    模型实时指标（用于大屏展示最新训练结果）
    如果有已训练的模型，返回指标；否则返回占位数据
    """
    try:
        # 尝试加载最新训练结果（假设保存在 output/report 目录）
        import os
        import json
        from backend.config import REPORT_DIR

        latest_metrics = {
            "svm": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "random_forest": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "svm_roc": None,
            "rf_roc": None,
            "trained": False,
            "last_update": None
        }

        # 尝试读取保存的模型指标文件
        metrics_file = os.path.join(REPORT_DIR, "latest_model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                latest_metrics['svm'] = data.get('svm', latest_metrics['svm'])
                latest_metrics['random_forest'] = data.get('random_forest', latest_metrics['random_forest'])
                latest_metrics['svm_roc'] = data.get('svm_roc')
                latest_metrics['rf_roc'] = data.get('rf_roc')
                latest_metrics['trained'] = True
                latest_metrics['last_update'] = data.get('timestamp', None)

        return {
            "svm": latest_metrics['svm'],
            "random_forest": latest_metrics['random_forest'],
            "svm_roc": latest_metrics['svm_roc'],
            "rf_roc": latest_metrics['rf_roc'],
            "trained": latest_metrics['trained'],
            "last_update": latest_metrics['last_update'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # 返回默认空数据
        return {
            "svm": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "random_forest": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0},
            "svm_roc": None,
            "rf_roc": None,
            "trained": False,
            "last_update": None,
            "timestamp": datetime.now().isoformat()
        }


@router.get("/feature-importance")
async def get_feature_importance(top_n: int = Query(10, description="返回前N个重要特征")):
    """
    获取特征重要性Top N（用于大屏柱状图展示）
    优先加载随机森林模型的特征重要性，若无则返回空
    """
    try:
        import os
        import joblib
        import pandas as pd
        from backend.config import MODEL_DIR

        importance_data = {"features": [], "importance": []}
        df = get_cleaned_data()
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]

        model = None
        # 优先使用内存中基于原始特征训练的随机森林模型
        try:
            from backend.api.train_api import _model_cache
            if _model_cache['rf'] is not None and _model_cache.get('fused') is False:
                model = _model_cache['rf']
        except Exception:
            pass

        # 否则尝试从磁盘加载随机森林模型
        if model is None:
            rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
            if os.path.exists(rf_model_path):
                model = joblib.load(rf_model_path)

        if model is not None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # 确保长度与原始特征数匹配（融合特征模型维度不一致时跳过）
            if len(importances) == len(feature_cols):
                imp_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                importance_data['features'] = imp_df['feature'].tolist()
                importance_data['importance'] = imp_df['importance'].round(4).tolist()

        return {
            "top_n": top_n,
            "importance": importance_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "top_n": top_n,
            "importance": {"features": [], "importance": []},
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/all")
async def get_all_screen_data():
    """
    一键获取大屏所有数据（聚合调用，减少前端请求数）
    返回 global、user_layer、consumption、model_metrics、feature_importance 的合集
    """
    try:
        from fastapi import Request
        # 分别调用各个接口
        global_data = await get_global_metrics()
        user_layer = await get_user_layer_distribution("member_level")
        consumption = await get_consumption_metrics()
        model_metrics = await get_model_metrics()
        importance = await get_feature_importance(10)

        return {
            "global": global_data,
            "user_layer": user_layer,
            "consumption": consumption,
            "model_metrics": model_metrics,
            "feature_importance": importance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取大屏全部数据失败: {str(e)}")
