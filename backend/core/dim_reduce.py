# backend/core/dim_reduce.py
"""
降维核心算法模块
实现：
1. PCA 主成分分析（自动选择成分数量）
2. 因子分析（公因子提取）
3. PCA + 因子分析融合降维
所有算法基于 sklearn 实现，与 config 配置联动
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from backend.config import (
    PCA_N_COMPONENTS,
    PCA_VARIANCE_THRESHOLD,
    FACTOR_N_COMPONENTS,
    FUSION_SCALE,
    RANDOM_SEED,
    DEBUG
)


def _auto_select_pca_components(explained_variance_ratio: np.ndarray, threshold: float = 0.95) -> int:
    """
    根据累计方差贡献率自动选择主成分数量
    :param explained_variance_ratio: 各主成分方差贡献率数组
    :param threshold: 累计方差阈值（默认95%）
    :return: 满足阈值的最小主成分数量
    """
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative >= threshold) + 1
    # 至少保留2个主成分（便于可视化）
    if n_components < 2:
        n_components = min(2, len(explained_variance_ratio))
    return n_components


def perform_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    random_state: int = RANDOM_SEED,
    return_components: bool = False
) -> Dict[str, Any]:
    """
    执行 PCA 主成分分析
    :param X: 特征矩阵 (n_samples, n_features)
    :param n_components: 指定主成分数量，若为None则自动选择（累计方差>=95%）
    :param random_state: 随机种子
    :param return_components: 是否返回完整载荷矩阵
    :return: 字典包含 pca_model, n_components, explained_variance_ratio,
             cumulative_variance_ratio, 可选的 components
    """
    if X is None or len(X) == 0:
        raise ValueError("输入特征矩阵 X 不能为空")

    n_features = X.shape[1]

    # 如果未指定 n_components，先计算全量 PCA 获取方差贡献率
    if n_components is None:
        pca_full = PCA(n_components=min(n_features, X.shape[0]))
        pca_full.fit(X)
        var_ratio = pca_full.explained_variance_ratio_
        n_components = _auto_select_pca_components(var_ratio, PCA_VARIANCE_THRESHOLD)

    # 确保 n_components 不超过特征数和样本数
    max_components = min(n_features, X.shape[0])
    if n_components > max_components:
        n_components = max_components
        if DEBUG:
            print(f"[PCA] 调整成分数为 {n_components}（不超过特征数和样本数）")

    # 执行 PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)

    # 计算累计方差贡献率
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    result = {
        'pca_model': pca,
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance_ratio,
    }

    # 返回载荷矩阵（前5个成分作为示例，完整矩阵可能很大）
    if return_components:
        result['components'] = pca.components_

    if DEBUG:
        print(f"[PCA] 完成: 选择 {n_components} 个主成分，累计方差 {cumulative_variance_ratio[-1]:.2%}")

    return result


def perform_factor_analysis(
    X: np.ndarray,
    n_factors: Optional[int] = None,
    random_state: int = RANDOM_SEED,
    return_scores: bool = False
) -> Dict[str, Any]:
    """
    执行因子分析（探索性因子分析）
    :param X: 特征矩阵 (n_samples, n_features)
    :param n_factors: 公因子数量，若为None则使用 config 中的默认值
    :param random_state: 随机种子
    :param return_scores: 是否返回因子得分矩阵
    :return: 字典包含 n_factors, loadings, communality, 可选的 factor_scores
    """
    if X is None or len(X) == 0:
        raise ValueError("输入特征矩阵 X 不能为空")

    n_features = X.shape[1]
    n_samples = X.shape[0]

    # 确定因子数量
    if n_factors is None:
        n_factors = FACTOR_N_COMPONENTS

    # 因子数量不能超过特征数
    if n_factors > n_features:
        n_factors = n_features
        if DEBUG:
            print(f"[FactorAnalysis] 调整因子数为 {n_factors}（不超过特征数）")

    # 因子数量不能超过样本数
    if n_factors > n_samples:
        n_factors = max(1, n_samples - 1)
        if DEBUG:
            print(f"[FactorAnalysis] 调整因子数为 {n_factors}（不超过样本数-1）")

    # 执行因子分析
    fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
    fa.fit(X)

    # 获取载荷矩阵 (n_features, n_factors)
    loadings = fa.components_.T  # 转置为 (n_factors, n_features) -> 通常载荷定义为 (n_features, n_factors)
    # 更标准的表示：载荷矩阵为 (n_features, n_factors)
    loadings = fa.components_.T

    # 计算公因子方差 (communality)：每个特征在各因子上的载荷平方和
    communality = np.sum(loadings ** 2, axis=1)

    # 计算因子得分（如果请求）
    factor_scores = None
    if return_scores:
        factor_scores = fa.transform(X)

    result = {
        'n_factors': n_factors,
        'loadings': loadings,
        'communality': communality,
        'fa_model': fa,
        'factor_scores': factor_scores,
    }

    if DEBUG:
        print(f"[FactorAnalysis] 完成: 提取 {n_factors} 个公因子")
        print(f"[FactorAnalysis] 平均公因子方差: {communality.mean():.3f}")

    return result


def perform_fusion(
    X: np.ndarray,
    pca_components: Optional[int] = None,
    factor_components: int = 10,
    random_state: int = RANDOM_SEED,
    scale_fusion: bool = FUSION_SCALE
) -> Dict[str, Any]:
    """
    执行 PCA + 因子分析融合降维
    将 PCA 主成分得分与因子分析公因子得分拼接，形成融合特征
    :param X: 特征矩阵 (n_samples, n_features)
    :param pca_components: PCA主成分数量（自动选择）
    :param factor_components: 因子分析公因子数量
    :param random_state: 随机种子
    :param scale_fusion: 融合后是否再次标准化
    :return: 字典包含 fused_features, fused_shape, pca_components, factor_components
    """
    if X is None or len(X) == 0:
        raise ValueError("输入特征矩阵 X 不能为空")

    n_samples, n_features = X.shape

    # 1. 执行 PCA（自动选择成分数量）
    pca_result = perform_pca(
        X,
        n_components=pca_components,
        random_state=random_state,
        return_components=False
    )
    pca_model = pca_result['pca_model']
    n_pca = pca_result['n_components']
    pca_scores = pca_model.transform(X)

    if DEBUG:
        print(f"[Fusion] PCA 完成: {n_pca} 个主成分")

    # 2. 执行因子分析
    fa_result = perform_factor_analysis(
        X,
        n_factors=factor_components,
        random_state=random_state,
        return_scores=True
    )
    n_factor = fa_result['n_factors']
    factor_scores = fa_result['factor_scores']

    if DEBUG:
        print(f"[Fusion] 因子分析完成: {n_factor} 个公因子")

    # 3. 融合特征：横向拼接 PCA 得分 + 因子得分
    fused_features = np.hstack([pca_scores, factor_scores])

    # 4. 可选：融合后标准化
    if scale_fusion:
        scaler = StandardScaler()
        fused_features = scaler.fit_transform(fused_features)
        if DEBUG:
            print("[Fusion] 融合特征已标准化")

    result = {
        'fused_features': fused_features,
        'fused_shape': fused_features.shape,
        'pca_components': n_pca,
        'factor_components': n_factor,
        'pca_scores': pca_scores,
        'factor_scores': factor_scores,
        'pca_model': pca_model,
        'fa_model': fa_result['fa_model'],
    }

    if DEBUG:
        print(f"[Fusion] 完成: 融合特征维度 {fused_features.shape}")

    return result


def get_feature_names(exclude_cols: List[str] = None) -> List[str]:
    """
    获取特征名称列表（供可视化使用）
    这是一个辅助函数，用于在绘图时标记特征名
    :param exclude_cols: 要排除的列名（如 user_id, repurchase_label）
    :return: 特征名称列表
    """
    from backend.config import DATASET_FILE
    from backend.utils.file_util import load_csv

    df = load_csv(DATASET_FILE)
    if exclude_cols is None:
        exclude_cols = ['user_id', 'repurchase_label']
    feature_names = [col for col in df.columns if col not in exclude_cols]
    return feature_names


def get_pca_transformed(
    X: np.ndarray,
    n_components: Optional[int] = None,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, PCA]:
    """
    快捷函数：获取 PCA 转换后的数据（用于模型训练）
    :param X: 特征矩阵
    :param n_components: 主成分数量（自动选择）
    :param random_state: 随机种子
    :return: (转换后数据, PCA模型)
    """
    result = perform_pca(X, n_components=n_components, random_state=random_state)
    pca = result['pca_model']
    X_transformed = pca.transform(X)
    return X_transformed, pca


def get_factor_transformed(
    X: np.ndarray,
    n_factors: Optional[int] = None,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, FactorAnalysis]:
    """
    快捷函数：获取因子分析转换后的数据（用于模型训练）
    :param X: 特征矩阵
    :param n_factors: 公因子数量
    :param random_state: 随机种子
    :return: (转换后数据, 因子分析模型)
    """
    result = perform_factor_analysis(X, n_factors=n_factors, random_state=random_state, return_scores=True)
    fa = result['fa_model']
    X_transformed = fa.transform(X)
    return X_transformed, fa


def get_fusion_transformed(
    X: np.ndarray,
    pca_components: Optional[int] = None,
    factor_components: int = 10,
    random_state: int = RANDOM_SEED,
    scale_fusion: bool = FUSION_SCALE
) -> Tuple[np.ndarray, Dict]:
    """
    快捷函数：获取融合降维后的数据（用于模型训练）
    :param X: 特征矩阵
    :param pca_components: PCA主成分数量
    :param factor_components: 因子数量
    :param random_state: 随机种子
    :param scale_fusion: 融合后标准化
    :return: (融合后数据, 包含模型的字典)
    """
    result = perform_fusion(
        X,
        pca_components=pca_components,
        factor_components=factor_components,
        random_state=random_state,
        scale_fusion=scale_fusion
    )
    return result['fused_features'], {
        'pca_model': result['pca_model'],
        'fa_model': result['fa_model'],
        'pca_components': result['pca_components'],
        'factor_components': result['factor_components']
    }


if __name__ == "__main__":
    # 简单测试（需要清洗后的数据）
    from backend.core.data_clean import clean_dataset

    print("=" * 60)
    print("测试降维模块")
    print("=" * 60)

    # 1. 加载清洗数据
    result = clean_dataset()
    df = result['df_cleaned']
    feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
    X = df[feature_cols].values
    print(f"数据形状: {X.shape}")

    # 2. 测试 PCA
    print("\n--- PCA 测试 ---")
    pca_result = perform_pca(X, return_components=False)
    print(f"主成分数: {pca_result['n_components']}")
    print(f"累计方差: {pca_result['cumulative_variance_ratio'][-1]:.2%}")

    # 3. 测试因子分析
    print("\n--- 因子分析测试 ---")
    fa_result = perform_factor_analysis(X, n_factors=8, return_scores=False)
    print(f"公因子数: {fa_result['n_factors']}")
    print(f"载荷矩阵形状: {fa_result['loadings'].shape}")
    print(f"平均公因子方差: {fa_result['communality'].mean():.3f}")

    # 4. 测试融合降维
    print("\n--- 融合降维测试 ---")
    fusion_result = perform_fusion(X, factor_components=8, scale_fusion=True)
    print(f"融合特征形状: {fusion_result['fused_shape']}")
    print(f"PCA成分数: {fusion_result['pca_components']}")
    print(f"因子数: {fusion_result['factor_components']}")

    print("\n[OK]  所有测试通过！")