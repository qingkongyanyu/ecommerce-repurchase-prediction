# backend/api/dim_api.py
"""
降维分析接口路由
提供 PCA、因子分析、融合降维的接口，以及对应的可视化图表
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import json

from backend.config import DIM_PREFIX, RANDOM_SEED, DEBUG
from backend.schemas.data_schema import DataOverviewResponse
from backend.schemas.model_schema import (
    PCAResultResponse,
    FactorResultResponse,
    FusionResultResponse
)
from backend.core.data_clean import clean_dataset
from backend.core.visual_draw import (
    draw_correlation_heatmap,
    draw_pca_scree,
    draw_factor_loadings
)
from backend.utils.file_util import load_csv
from backend.utils.math_util import normalize_array

# 假设 dim_reduce 模块已存在（后续补全核心算法时实现）
# 这里先定义接口，实际调用时再导入
try:
    from backend.core.dim_reduce import (
        perform_pca,
        perform_factor_analysis,
        perform_fusion,
        get_feature_names
    )
except ImportError as e:
    # 临时方案：如果 dim_reduce 未实现，在接口中返回提示
    print(f"[警告] dim_reduce 模块尚未实现: {e}")
    # 定义占位函数，避免路由无法加载
    def perform_pca(*args, **kwargs):
        raise NotImplementedError("PCA 核心算法尚未实现")
    def perform_factor_analysis(*args, **kwargs):
        raise NotImplementedError("因子分析核心算法尚未实现")
    def perform_fusion(*args, **kwargs):
        raise NotImplementedError("融合降维核心算法尚未实现")
    def get_feature_names(*args, **kwargs):
        return []


router = APIRouter(prefix=DIM_PREFIX, tags=["降维分析"])

# 缓存清洗后的数据，避免重复加载
_cleaned_df_cache = None
_pca_model_cache = None
_factor_model_cache = None


def get_cleaned_data() -> pd.DataFrame:
    """获取清洗后的数据（缓存）"""
    global _cleaned_df_cache
    if _cleaned_df_cache is None:
        result = clean_dataset()
        _cleaned_df_cache = result['df_cleaned']
    return _cleaned_df_cache


@router.get("/pca", response_model=PCAResultResponse)
async def pca_analysis(
    n_components: Optional[int] = Query(None, description="指定主成分数量，默认自动选择（累计方差>=95%）"),
    return_components: bool = Query(False, description="是否返回载荷矩阵（可能较大）")
):
    """
    执行 PCA 主成分分析
    """
    try:
        df = get_cleaned_data()
        # 排除 user_id 和标签列
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
        X = df[feature_cols].values

        # 调用核心算法
        result = perform_pca(
            X,
            n_components=n_components,
            random_state=RANDOM_SEED,
            return_components=return_components
        )

        # 生成碎石图
        scree_base64 = draw_pca_scree(result['pca_model'], return_base64=True)

        # 构建响应
        response = PCAResultResponse(
            n_components_selected=result['n_components'],
            explained_variance_ratio=result['explained_variance_ratio'].tolist(),
            cumulative_variance_ratio=result['cumulative_variance_ratio'].tolist(),
            scree_plot_base64=scree_base64
        )

        # 如果请求返回载荷矩阵
        if return_components and 'components' in result:
            # 只返回前5个成分的载荷（避免数据过大）
            components = result['components'][:5, :].tolist()
            response.components_matrix = components

        return response
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"PCA 功能开发中: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA 分析失败: {str(e)}")


@router.get("/factor", response_model=FactorResultResponse)
async def factor_analysis(
    n_factors: Optional[int] = Query(None, description="公因子数量，默认10"),
    return_scores: bool = Query(False, description="是否返回因子得分（可能较大）")
):
    """
    执行因子分析
    """
    try:
        df = get_cleaned_data()
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
        feature_names = feature_cols
        X = df[feature_cols].values

        # 调用核心算法
        result = perform_factor_analysis(
            X,
            n_factors=n_factors,
            random_state=RANDOM_SEED,
            return_scores=return_scores
        )

        # 生成因子载荷散点图（需要至少2个因子）
        loadings_plot = None
        if result['loadings'].shape[1] >= 2:
            loadings_plot = draw_factor_loadings(
                result['loadings'],
                feature_names,
                return_base64=True
            )

        response = FactorResultResponse(
            n_factors=result['n_factors'],
            communality=result['communality'].tolist() if result['communality'] is not None else [],
            loading_plot_base64=loadings_plot
        )

        # 返回载荷矩阵（前5行示例）
        if result['loadings'] is not None:
            loadings_sample = result['loadings'][:5, :].tolist()
            response.loadings_matrix = loadings_sample

        # 返回因子得分示例（前10行）
        if return_scores and result['factor_scores'] is not None:
            scores_sample = result['factor_scores'][:10, :].tolist()
            response.factor_score = scores_sample

        return response
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"因子分析功能开发中: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"因子分析失败: {str(e)}")


@router.get("/fusion", response_model=FusionResultResponse)
async def fusion_analysis(
    pca_components: Optional[int] = Query(None, description="PCA主成分数量，自动选择"),
    factor_components: int = Query(10, description="因子分析公因子数量"),
    return_sample: bool = Query(True, description="是否返回融合特征示例")
):
    """
    执行 PCA + 因子分析融合降维
    """
    try:
        df = get_cleaned_data()
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
        X = df[feature_cols].values

        # 调用核心算法
        result = perform_fusion(
            X,
            pca_components=pca_components,
            factor_components=factor_components,
            random_state=RANDOM_SEED,
            scale_fusion=True
        )

        response = FusionResultResponse(
            fused_feature_shape=result['fused_shape'],
            pca_components=result['pca_components'],
            factor_components=result['factor_components'],
            sample_fused_features=[]
        )

        # 返回融合特征示例（前5行）
        if return_sample and result['fused_features'] is not None:
            sample = result['fused_features'][:5, :].tolist()
            response.sample_fused_features = sample

        return response
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=f"融合降维功能开发中: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"融合降维失败: {str(e)}")


@router.get("/heatmap")
async def get_correlation_heatmap(
    max_features: int = Query(30, description="最多显示的特征数量，避免热力图过于拥挤")
):
    """
    获取特征相关性数据（供前端 ECharts heatmap 原生绘制）
    同时保留 matplotlib 生成的 Base64 图片（用于报告/导出）
    """
    try:
        df = get_cleaned_data()
        # 排除 user_id 和标签列
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
        # 如果特征过多，选取方差最大的前N个
        if len(feature_cols) > max_features:
            # 按方差排序选取
            variances = df[feature_cols].var()
            top_features = variances.nlargest(max_features).index.tolist()
        else:
            top_features = feature_cols

        # 相关系数矩阵（按 top_features 顺序）
        corr = df[top_features].corr()
        correlation = [[round(float(v), 4) for v in row] for row in corr.values]

        heatmap_base64 = draw_correlation_heatmap(
            df,
            features=top_features,
            title=f'特征相关性热力图 (Top {len(top_features)})',
            return_base64=True
        )

        return {
            "heatmap_base64": heatmap_base64,
            "features": top_features,
            "correlation": correlation,
            "features_count": len(top_features),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成热力图失败: {str(e)}")


@router.get("/summary")
async def get_dimension_summary():
    """
    获取降维分析的综合摘要（供大屏使用）
    返回 PCA、因子分析的关键统计数据
    """
    try:
        # 这里返回一些预计算的统计信息，不触发耗时的降维计算
        df = get_cleaned_data()
        feature_count = len([col for col in df.columns if col not in ['user_id', 'repurchase_label']])

        return {
            "original_features": feature_count,
            "pca_available": True,
            "factor_available": True,
            "fusion_available": True,
            "sample_size": len(df),
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取摘要失败: {str(e)}")


# ---------- 辅助函数：供其他模块调用 ----------
def get_fusion_features() -> np.ndarray:
    """
    供模型训练模块调用，获取融合降维特征
    """
    try:
        df = get_cleaned_data()
        feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
        X = df[feature_cols].values
        result = perform_fusion(X, scale_fusion=True)
        return result['fused_features']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取融合特征失败: {str(e)}")