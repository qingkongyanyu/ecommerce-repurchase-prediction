# backend/api/train_api.py
"""
模型训练与对比接口路由
提供 SVM、随机森林的训练、对比预测、报告导出功能
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime

from backend.config import MODEL_PREFIX, RANDOM_SEED, MODEL_DIR, REPORT_DIR, DEBUG
from backend.schemas.model_schema import (
    TrainRequest,
    TrainResponse,
    CompareResponse,
    PredictRequest,
    PredictResponse,
    ModelReportResponse
)
from backend.api.data_api import get_cleaned_df, get_cleaned_scaler  # 复用数据API的清洗缓存
from backend.core.model_train import ModelTrainer
from backend.core.visual_draw import (
    draw_roc_curves,
    draw_feature_importance,
    draw_metrics_comparison
)
from backend.utils.file_util import save_csv, ensure_dir

router = APIRouter(prefix=MODEL_PREFIX, tags=["模型训练与对比"])

# 缓存训练好的模型和评估结果（避免重复训练）
_model_cache = {
    'svm': None,
    'rf': None,
    'metrics': None,
    'fused': False  # 是否使用融合特征
}


def _get_data(use_fusion: bool = True):
    """获取训练数据（原始特征或融合特征）"""
    df = get_cleaned_df()
    X_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]

    if use_fusion:
        # 尝试从降维模块获取融合特征
        try:
            from backend.api.dim_api import get_fusion_features
            X = get_fusion_features()
            feature_names = [f'fusion_{i}' for i in range(X.shape[1])]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取融合特征失败: {str(e)}")
    else:
        X = df[X_cols].values
        feature_names = X_cols

    y = df['repurchase_label'].values
    return X, y, feature_names


def _compute_roc(model, X_test, y_test):
    """计算模型在测试集上的 ROC 曲线数据（供大屏展示）"""
    try:
        from sklearn.metrics import roc_curve
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        return {'fpr': [float(v) for v in fpr], 'tpr': [float(v) for v in tpr]}
    except Exception:
        return None


def _write_latest_metrics(svm_metrics=None, rf_metrics=None, svm_roc=None, rf_roc=None):
    """将最新训练指标写入 output/report/latest_model_metrics.json，供大屏接口读取"""
    if svm_metrics is None and rf_metrics is None:
        return
    data = {}
    if svm_metrics is not None:
        data['svm'] = {k: v for k, v in svm_metrics.items() if k != 'confusion_matrix'}
    if rf_metrics is not None:
        data['random_forest'] = {k: v for k, v in rf_metrics.items() if k != 'confusion_matrix'}
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if svm_roc:
        data['svm_roc'] = svm_roc
    if rf_roc:
        data['rf_roc'] = rf_roc
    ensure_dir(REPORT_DIR)
    with open(os.path.join(REPORT_DIR, "latest_model_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _train_models(X, y, feature_names, svm_kwargs=None, rf_kwargs=None) -> ModelTrainer:
    """用 ModelTrainer 训练双模型并更新缓存"""
    trainer = ModelTrainer(random_state=RANDOM_SEED)
    trainer.feature_names = feature_names
    trainer.split_data(X, y)
    trainer.train_svm(**(svm_kwargs or {}))
    trainer.train_random_forest(**(rf_kwargs or {}))
    _model_cache['svm'] = trainer.svm_model
    _model_cache['rf'] = trainer.rf_model
    _model_cache['metrics'] = {
        'svm': trainer.svm_metrics,
        'rf': trainer.rf_metrics,
    }
    _model_cache['fused'] = None  # 由调用方设置

    # 写入实时指标文件（供大屏 model-metrics 接口读取）
    _write_latest_metrics(
        trainer.svm_metrics, trainer.rf_metrics,
        _compute_roc(trainer.svm_model, trainer.X_test, trainer.y_test),
        _compute_roc(trainer.rf_model, trainer.X_test, trainer.y_test),
    )
    return trainer


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    训练指定模型（SVM 或 随机森林）
    """
    try:
        use_fusion = request.use_fusion_features
        X, y, feature_names = _get_data(use_fusion)

        start_time = time.time()
        trainer = ModelTrainer(random_state=RANDOM_SEED)
        trainer.feature_names = feature_names
        trainer.split_data(X, y)

        if request.model_type.lower() == 'svm':
            params = {}
            if request.svm_c is not None:
                params['C'] = request.svm_c
            if request.svm_gamma is not None:
                params['gamma'] = request.svm_gamma
            trainer.train_svm(**params)
            model = trainer.svm_model
            metrics = trainer.svm_metrics
            model_type = 'svm'
        elif request.model_type.lower() == 'random_forest':
            params = {}
            if request.rf_n_estimators is not None:
                params['n_estimators'] = request.rf_n_estimators
            if request.rf_max_depth is not None:
                params['max_depth'] = request.rf_max_depth
            trainer.train_random_forest(**params)
            model = trainer.rf_model
            metrics = trainer.rf_metrics
            model_type = 'random_forest'
        else:
            raise HTTPException(status_code=400, detail="model_type 必须为 'svm' 或 'random_forest'")

        train_time = time.time() - start_time

        response = TrainResponse(
            model_type=model_type,
            is_trained=True,
            train_time_seconds=train_time,
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1=metrics.get('f1', 0.0),
            auc=metrics.get('auc', None),
            confusion_matrix=metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        )

        # 随机森林特征重要性
        if model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            order = np.argsort(importances)[::-1]
            top_names = [feature_names[i] for i in order[:30]]
            top_importances = importances[order[:30]]
            response.feature_importance = [
                {'feature': name, 'importance': float(val)}
                for name, val in zip(top_names, top_importances)
            ]

        # 缓存
        _model_cache['svm' if model_type == 'svm' else 'rf'] = model
        _model_cache['fused'] = use_fusion

        # 写入实时指标文件（供大屏 model-metrics 接口读取）
        _write_latest_metrics(
            trainer.svm_metrics if model_type == 'svm' else None,
            trainer.rf_metrics if model_type == 'random_forest' else None,
            _compute_roc(trainer.svm_model, trainer.X_test, trainer.y_test) if model_type == 'svm' else None,
            _compute_roc(trainer.rf_model, trainer.X_test, trainer.y_test) if model_type == 'random_forest' else None,
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")


@router.get("/compare", response_model=CompareResponse)
async def compare_models(
        use_fusion: bool = Query(True, description="是否使用融合特征进行对比")
):
    """
    对比 SVM 和 随机森林的性能
    先检查是否已有训练好的模型，否则自动训练
    """
    try:
        X, y, feature_names = _get_data(use_fusion)

        if _model_cache['svm'] is not None and _model_cache['rf'] is not None and _model_cache['fused'] == use_fusion:
            svm_model = _model_cache['svm']
            rf_model = _model_cache['rf']
        else:
            trainer = _train_models(X, y, feature_names)
            svm_model = _model_cache['svm']
            rf_model = _model_cache['rf']
            _model_cache['fused'] = use_fusion

        # 重新评估指标（测试集划分）
        trainer = ModelTrainer(random_state=RANDOM_SEED)
        trainer.feature_names = feature_names
        trainer.split_data(X, y)
        trainer.svm_model = svm_model
        trainer.rf_model = rf_model
        trainer.svm_metrics = trainer._eval_model(svm_model, 'svm') if hasattr(trainer, '_eval_model') else None
        trainer.rf_metrics = trainer._eval_model(rf_model, 'rf') if hasattr(trainer, '_eval_model') else None

        # 若类没有私有评估方法，直接用手动评估
        if trainer.svm_metrics is None or trainer.rf_metrics is None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            svm_metrics, rf_metrics = {}, {}
            for name, model in [('svm', svm_model), ('rf', rf_model)]:
                y_pred = model.predict(trainer.X_test)
                y_prob = model.predict_proba(trainer.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                m = {
                    'accuracy': accuracy_score(trainer.y_test, y_pred),
                    'precision': precision_score(trainer.y_test, y_pred, zero_division=0),
                    'recall': recall_score(trainer.y_test, y_pred, zero_division=0),
                    'f1': f1_score(trainer.y_test, y_pred, zero_division=0),
                    'auc': roc_auc_score(trainer.y_test, y_prob) if y_prob is not None else 0.0,
                    'confusion_matrix': confusion_matrix(trainer.y_test, y_pred).tolist()
                }
                if name == 'svm':
                    svm_metrics = m
                else:
                    rf_metrics = m

        svm_metrics_dict = {k: v for k, v in svm_metrics.items() if k != 'confusion_matrix'}
        rf_metrics_dict = {k: v for k, v in rf_metrics.items() if k != 'confusion_matrix'}

        # ROC 曲线
        from sklearn.metrics import roc_curve, auc
        roc_base64 = ""
        try:
            svm_proba = svm_model.predict_proba(trainer.X_test)[:, 1] if hasattr(svm_model, 'predict_proba') else None
            rf_proba = rf_model.predict_proba(trainer.X_test)[:, 1]
            if rf_proba is not None:
                fpr_rf, tpr_rf, _ = roc_curve(trainer.y_test, rf_proba)
                auc_rf = auc(fpr_rf, tpr_rf)
                if svm_proba is not None:
                    fpr_svm, tpr_svm, _ = roc_curve(trainer.y_test, svm_proba)
                    auc_svm = auc(fpr_svm, tpr_svm)
                elif hasattr(svm_model, 'decision_function'):
                    svm_scores = svm_model.decision_function(trainer.X_test)
                    fpr_svm, tpr_svm, _ = roc_curve(trainer.y_test, svm_scores)
                    auc_svm = auc(fpr_svm, tpr_svm)
                else:
                    fpr_svm, tpr_svm, auc_svm = [0, 1], [0, 1], 0.5
                roc_base64 = draw_roc_curves(
                    fpr_svm, tpr_svm, auc_svm,
                    fpr_rf, tpr_rf, auc_rf,
                    return_base64=True
                )
        except Exception:
            pass

        # 指标对比柱状图
        metrics_bar_base64 = draw_metrics_comparison(
            svm_metrics_dict, rf_metrics_dict,
            metrics_list=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            return_base64=True
        )

        # 对比摘要
        svm_auc = svm_metrics_dict.get('auc', 0)
        rf_auc = rf_metrics_dict.get('auc', 0)
        if svm_auc > rf_auc:
            summary = f"SVM 整体略优于随机森林，AUC高出 {svm_auc - rf_auc:.3f}"
        else:
            summary = f"随机森林整体略优于SVM，AUC高出 {rf_auc - svm_auc:.3f}"
        if abs(svm_metrics_dict.get('accuracy', 0) - rf_metrics_dict.get('accuracy', 0)) < 0.02:
            summary += "，两者准确率接近"

        return CompareResponse(
            svm_metrics=svm_metrics_dict,
            rf_metrics=rf_metrics_dict,
            comparison_summary=summary,
            roc_curve_base64=roc_base64,
            metrics_bar_base64=metrics_bar_base64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比失败: {str(e)}")


def _scale_input(features: list) -> np.ndarray:
    """
    将用户输入的原始特征向量转换为与训练一致的标准化特征向量。
    与训练口径一致：连续数值列用 scaler 标准化，分类编码列保持原样。
    """
    df_cleaned = get_cleaned_df()
    scaler = get_cleaned_scaler()
    feature_cols = [c for c in df_cleaned.columns if c not in ['user_id', 'repurchase_label']]
    row = pd.DataFrame([list(features)], columns=feature_cols).astype(float)
    # 通过 scaler 记录的 feature_names_in_ 定位被标准化的连续数值列
    scaled_cols = list(getattr(scaler, 'feature_names_in_', []))
    if scaled_cols:
        row[scaled_cols] = scaler.transform(row[scaled_cols])
    return row[feature_cols].values


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    对单用户特征进行复购预测
    演示用：默认基于原始 36 维特征训练/加载随机森林模型，输入为用户原始特征向量。
    """
    try:
        # 若无可用（基于原始特征）模型，快速训练一个随机森林（演示用）
        if _model_cache['rf'] is None or _model_cache.get('fused') is not False:
            X, y, feature_names = _get_data(use_fusion=False)
            trainer = ModelTrainer(random_state=RANDOM_SEED)
            trainer.feature_names = feature_names
            trainer.split_data(X, y)
            trainer.train_random_forest()
            _model_cache['rf'] = trainer.rf_model
            _model_cache['metrics'] = {'rf': trainer.rf_metrics}
            _model_cache['fused'] = False
            _write_latest_metrics(
                rf_metrics=trainer.rf_metrics,
                rf_roc=_compute_roc(trainer.rf_model, trainer.X_test, trainer.y_test),
            )

        model = _model_cache['rf']
        model_name = "RandomForest_Original"

        if len(request.features) != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"特征维度不匹配：模型期望 {model.n_features_in_} 维（36个原始特征），收到 {len(request.features)} 维",
            )

        start_time = time.time()
        features = _scale_input(request.features)
        label = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0][1]) if hasattr(model, 'predict_proba') else None
        if prob is None and hasattr(model, 'decision_function'):
            prob = float(1 / (1 + np.exp(-model.decision_function(features)[0])))
        elapsed_ms = (time.time() - start_time) * 1000

        if prob is not None:
            if prob >= 0.7:
                risk = "高"
            elif prob >= 0.4:
                risk = "中"
            else:
                risk = "低"
        else:
            risk = "未知"

        return PredictResponse(
            model_used=model_name,
            repurchase_probability=prob,
            prediction_label=label,
            risk_level=risk,
            predict_time_ms=elapsed_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/report", response_model=ModelReportResponse)
async def generate_model_report(
        use_fusion: bool = Query(True, description="是否使用融合特征")
):
    """
    生成模型对比报告（CSV + JSON 摘要）
    """
    try:
        compare_result = await compare_models(use_fusion)

        metrics_summary = {
            "SVM": compare_result.svm_metrics,
            "RandomForest": compare_result.rf_metrics
        }
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_df = pd.DataFrame([
            {"Model": "SVM", **compare_result.svm_metrics},
            {"Model": "RandomForest", **compare_result.rf_metrics}
        ])
        report_name = f"model_comparison_{int(time.time())}.csv"
        report_path = os.path.join(REPORT_DIR, report_name)
        save_csv(report_df, report_path)

        json_path = report_path.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "generated_at": report_time,
                "use_fusion": use_fusion,
                "summary": compare_result.comparison_summary,
                "metrics": metrics_summary
            }, f, ensure_ascii=False, indent=2)

        return ModelReportResponse(
            report_path=report_path,
            report_name=report_name,
            generated_at=report_time,
            metrics_summary=metrics_summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")


@router.get("/importance")
async def get_top_features(
        top_n: int = Query(10, description="返回前N个重要特征"),
        use_fusion: bool = Query(True, description="是否使用融合特征模型")
):
    """
    获取随机森林的特征重要性（Top N）
    """
    try:
        if _model_cache['rf'] is None:
            X, y, feature_names = _get_data(use_fusion)
            _train_models(X, y, feature_names)
            _model_cache['fused'] = use_fusion
        else:
            _, _, feature_names = _get_data(use_fusion)

        importances = _model_cache['rf'].feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in idx]
        top_scores = importances[idx].tolist()

        return {
            "features": top_features,
            "importance": top_scores,
            "top_n": top_n,
            "model": "RandomForest" + ("_Fusion" if use_fusion else "_Original")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取特征重要性失败: {str(e)}")
