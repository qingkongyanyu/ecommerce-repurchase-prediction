# backend/schemas/model_schema.py
"""
建模与降维模块的请求/响应模型
涵盖：降维、训练、预测、评估对比
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


# ---------- 降维相关 ----------
class PCAResultResponse(BaseModel):
    """PCA结果响应"""
    n_components_selected: int = Field(..., description="选择的主成分数量")
    explained_variance_ratio: List[float] = Field(..., description="各主成分方差贡献率")
    cumulative_variance_ratio: List[float] = Field(..., description="累计方差贡献率")
    components_matrix: Optional[List[List[float]]] = Field(None, description="主成分载荷矩阵（前5个成分）")
    scree_plot_base64: str = Field(..., description="碎石图Base64编码")


class FactorResultResponse(BaseModel):
    """因子分析结果响应"""
    n_factors: int = Field(..., description="公因子数量")
    communality: List[float] = Field(..., description="公因子方差（共同度）")
    loadings_matrix: Optional[List[List[float]]] = Field(None, description="因子载荷矩阵（前5个因子）")
    factor_score: Optional[List[List[float]]] = Field(None, description="因子得分矩阵（前10行示例）")
    loading_plot_base64: Optional[str] = Field(None, description="因子载荷散点图Base64")


class FusionResultResponse(BaseModel):
    """PCA + 因子分析融合结果响应"""
    fused_feature_shape: List[int] = Field(..., description="融合后特征矩阵形状 [n_samples, n_features]")
    pca_components: int = Field(..., description="PCA主成分数量")
    factor_components: int = Field(..., description="因子数量")
    sample_fused_features: List[List[float]] = Field(..., description="融合特征前5行示例")
    fusion_plot_base64: Optional[str] = Field(None, description="融合特征分布图（可选）")


# ---------- 模型训练与评估 ----------
class TrainRequest(BaseModel):
    """模型训练请求（可覆盖默认超参）"""
    model_type: str = Field(..., description="模型类型，'svm' 或 'random_forest'")
    use_fusion_features: bool = Field(True, description="是否使用融合降维特征，否则使用原始特征")
    # 随机森林超参覆盖
    rf_n_estimators: Optional[int] = Field(None, description="树的数量")
    rf_max_depth: Optional[int] = Field(None, description="最大深度")
    # SVM超参覆盖
    svm_c: Optional[float] = Field(None, description="正则化参数C")
    svm_gamma: Optional[str] = Field(None, description="核函数系数，'scale' 或 'auto'")


class TrainResponse(BaseModel):
    """单模型训练响应"""
    model_type: str = Field(..., description="模型类型")
    is_trained: bool = Field(..., description="是否训练成功")
    train_time_seconds: float = Field(..., description="训练耗时（秒）")
    accuracy: float = Field(..., description="准确率")
    precision: float = Field(..., description="精确率")
    recall: float = Field(..., description="召回率")
    f1: float = Field(..., description="F1分数")
    auc: Optional[float] = Field(None, description="AUC值")
    feature_importance: Optional[List[Dict[str, float]]] = Field(None, description="特征重要性（仅随机森林）")
    confusion_matrix: List[List[int]] = Field(..., description="混淆矩阵 [[TN, FP], [FN, TP]]")


class CompareResponse(BaseModel):
    """双模型对比响应"""
    svm_metrics: Dict[str, Union[float, str]] = Field(..., description="SVM指标字典")
    rf_metrics: Dict[str, Union[float, str]] = Field(..., description="随机森林指标字典")
    comparison_summary: str = Field(..., description="对比结论文字摘要，如 '随机森林整体优于SVM，AUC高出0.03'")
    roc_curve_base64: str = Field(..., description="ROC对比曲线Base64")
    metrics_bar_base64: Optional[str] = Field(None, description="指标对比柱状图Base64")


class PredictRequest(BaseModel):
    """单用户复购预测请求"""
    features: List[float] = Field(..., description="用户特征向量，长度必须与训练特征数一致（原始特征36维或降维后特征）")
    use_fusion_features: bool = Field(True, description="是否使用融合特征模型（需与训练时一致）")


class PredictResponse(BaseModel):
    """单用户预测响应"""
    model_used: str = Field(..., description="使用的模型名称，如 'RandomForest_Fusion'")
    repurchase_probability: float = Field(..., description="复购概率（0~1）")
    prediction_label: int = Field(..., description="预测标签 0 或 1")
    risk_level: str = Field(..., description="风险等级，'低' / '中' / '高'，基于概率划分")
    predict_time_ms: float = Field(..., description="预测耗时（毫秒）")


class ModelReportResponse(BaseModel):
    """建模报告导出响应"""
    report_path: str = Field(..., description="报告文件路径")
    report_name: str = Field(..., description="报告文件名")
    generated_at: str = Field(..., description="生成时间")
    metrics_summary: Dict[str, Dict[str, float]] = Field(..., description="完整指标汇总表")
