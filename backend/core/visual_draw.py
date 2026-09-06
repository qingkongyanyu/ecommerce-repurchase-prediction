# TODO: seaborn/matplotlib绘图，生成图表Base64
# backend/core/visual_draw.py
"""
可视化绘图模块：使用 matplotlib + seaborn 生成全套分析图表。
支持输出 Base64 字符串（供前端直接渲染）或保存为本地图片。
涵盖图表清单：
1. 特征相关性热力图
2. PCA 方差贡献率碎石图
3. 因子载荷散点图（二维，展示前两个因子）
4. SVM & 随机森林 ROC 对比曲线
5. 随机森林 Top10 特征重要性柱状图
6. 用户消费金额分布箱线图（或小提琴图）
7. 模型评估指标横向对比柱状图（准确率、F1、召回率、AUC）
"""

import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，适合服务器
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from backend.config import (
    FIGURE_DPI, FIGURE_FIGSIZE, COLOR_PALETTE, DEBUG,
    STATIC_DIR, IMAGE_DIR
)
from backend.utils.file_util import save_fig_to_base64, save_fig_to_file, ensure_dir

# 设置全局 seaborn 样式
sns.set_theme(style='whitegrid', palette=COLOR_PALETTE)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False


def draw_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None,
                             title: str = '特征相关性热力图',
                             figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                             return_base64: bool = True) -> str:
    """
    绘制特征相关性热力图（使用 Pearson 相关系数）
    :param df: 包含特征的 DataFrame（已标准化）
    :param features: 指定要绘制的特征列，默认使用全部数值列
    :param title: 图表标题
    :param figsize: 图片尺寸
    :param return_base64: True 返回 base64，False 保存到本地
    :return: base64 字符串或空字符串（保存时）
    """
    if features is None:
        # 排除 user_id 和标签列
        exclude = ['user_id', 'repurchase_label']
        features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]

    # 若特征过多，限制显示数量（避免热力图过于拥挤）
    if len(features) > 30:
        features = features[:30]

    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=figsize)
    # 使用掩码显示上半部分（可选）
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=.5, cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/correlation_heatmap.png", dpi=FIGURE_DPI)
        return path


def draw_pca_scree(pca_model, title: str = 'PCA 主成分方差贡献率碎石图',
                   figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                   return_base64: bool = True) -> str:
    """
    绘制 PCA 碎石图（方差贡献率 + 累计贡献率）
    :param pca_model: 已拟合的 sklearn.decomposition.PCA 对象
    :param title: 标题
    :param figsize: 尺寸
    :param return_base64: True 返回 base64
    """
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance_ratio)
    n_components = len(explained_variance_ratio)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, n_components + 1)

    # 柱状图显示各成分方差贡献率
    ax.bar(x, explained_variance_ratio, alpha=0.7, label='个体方差贡献率', color='steelblue')
    # 折线图显示累计贡献率
    ax.plot(x, cumulative, marker='o', linestyle='--', color='darkred', label='累计贡献率', linewidth=2)

    ax.set_xlabel('主成分序号')
    ax.set_ylabel('方差贡献率')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 添加阈值线（例如 95%）
    threshold = 0.95
    ax.axhline(y=threshold, color='green', linestyle=':', alpha=0.7, label=f'{threshold*100:.0f}% 阈值')
    # 标记达到阈值的位置
    idx = np.argmax(cumulative >= threshold) + 1
    if idx < n_components:
        ax.axvline(x=idx, color='orange', linestyle=':', alpha=0.7, label=f'选择 {idx} 个主成分')

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/pca_scree.png", dpi=FIGURE_DPI)
        return path


def draw_factor_loadings(loadings_matrix: np.ndarray, feature_names: List[str],
                         title: str = '因子载荷散点图（前两个因子）',
                         figsize: Tuple[int, int] = (10, 8),
                         return_base64: bool = True) -> str:
    """
    绘制因子载荷散点图（仅展示前两个公因子）
    :param loadings_matrix: 载荷矩阵，形状 (n_features, n_factors)
    :param feature_names: 特征名称列表
    :param title: 标题
    :param figsize: 尺寸
    :param return_base64: True 返回 base64
    """
    if loadings_matrix.shape[1] < 2:
        raise ValueError("因子数量必须 >= 2 才能绘制二维散点图")

    factor1 = loadings_matrix[:, 0]
    factor2 = loadings_matrix[:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(factor1, factor2, color='royalblue', alpha=0.7, s=60)

    # 添加特征名称标签
    for i, name in enumerate(feature_names):
        ax.annotate(name, (factor1[i], factor2[i]), fontsize=9, alpha=0.8,
                    xytext=(3, 3), textcoords='offset points')

    # 添加参考线
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('因子1 载荷')
    ax.set_ylabel('因子2 载荷')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/factor_loadings.png", dpi=FIGURE_DPI)
        return path


def draw_roc_curves(fpr_svm: np.ndarray, tpr_svm: np.ndarray, auc_svm: float,
                    fpr_rf: np.ndarray, tpr_rf: np.ndarray, auc_rf: float,
                    title: str = 'SVM 与 随机森林 ROC 曲线对比',
                    figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                    return_base64: bool = True) -> str:
    """
    绘制双模型 ROC 曲线对比图
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f})', color='blue', linewidth=2)
    ax.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {auc_rf:.3f})', color='red', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='随机猜想', linewidth=1, alpha=0.6)

    ax.set_xlabel('假阳性率 (FPR)')
    ax.set_ylabel('真阳性率 (TPR)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/roc_curves.png", dpi=FIGURE_DPI)
        return path


def draw_feature_importance(importance_df: pd.DataFrame, top_n: int = 10,
                            title: str = '随机森林 Top-{} 特征重要性',
                            figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                            return_base64: bool = True) -> str:
    """
    绘制特征重要性水平柱状图（从高到低排序）
    :param importance_df: 包含 'feature' 和 'importance' 两列的 DataFrame
    :param top_n: 显示前 N 个特征
    :param title: 标题（可包含占位符 {} 用于 top_n）
    :param figsize: 尺寸
    :param return_base64: True 返回 base64
    """
    df_sorted = importance_df.sort_values('importance', ascending=True).tail(top_n)
    features = df_sorted['feature'].tolist()
    importances = df_sorted['importance'].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(features, importances, color='steelblue', alpha=0.8)
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=9)

    ax.set_xlabel('重要性系数')
    ax.set_title(title.format(top_n))
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/feature_importance.png", dpi=FIGURE_DPI)
        return path


def draw_consumption_boxplot(df: pd.DataFrame, column: str = 'total_consume',
                             title: str = '用户累计消费金额分布箱线图',
                             figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                             return_base64: bool = True) -> str:
    """
    绘制指定消费列的分组箱线图（可按复购标签分组）
    :param df: 包含目标列和 'repurchase_label' 的 DataFrame
    :param column: 要绘制的列名
    :param title: 标题
    """
    if column not in df.columns:
        raise ValueError(f"列 {column} 不存在于 DataFrame")
    if 'repurchase_label' not in df.columns:
        # 若没有标签，只画单组箱线图
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(y=df[column], ax=ax, color='skyblue')
        ax.set_ylabel(column)
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        # 按复购分组绘制箱线图
        sns.boxplot(x='repurchase_label', y=column, data=df, ax=ax, palette='Set2')
        ax.set_xlabel('复购标签 (0=不复购, 1=复购)')
        ax.set_ylabel(column)
        ax.set_title(title)

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/consumption_boxplot.png", dpi=FIGURE_DPI)
        return path


def draw_metrics_comparison(metrics_dict_svm: Dict[str, float],
                            metrics_dict_rf: Dict[str, float],
                            metrics_list: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'auc'],
                            title: str = '双模型评估指标对比',
                            figsize: Tuple[int, int] = FIGURE_FIGSIZE,
                            return_base64: bool = True) -> str:
    """
    绘制模型指标横向对比柱状图（分组柱状图）
    :param metrics_dict_svm: SVM 指标字典
    :param metrics_dict_rf: 随机森林指标字典
    :param metrics_list: 要显示的指标列表
    :param title: 标题
    """
    x = np.arange(len(metrics_list))
    width = 0.35

    svm_values = [metrics_dict_svm.get(m, 0) for m in metrics_list]
    rf_values = [metrics_dict_rf.get(m, 0) for m in metrics_list]

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, svm_values, width, label='SVM', color='royalblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_values, width, label='随机森林', color='coral', alpha=0.8)

    ax.set_ylabel('分数')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 添加数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if return_base64:
        return save_fig_to_base64(fig, dpi=FIGURE_DPI)
    else:
        path = save_fig_to_file(fig, f"{STATIC_DIR}/metrics_comparison.png", dpi=FIGURE_DPI)
        return path


# ---------- 便捷批量生成函数 ----------
def generate_all_plots(df_cleaned: pd.DataFrame,
                       pca_model=None,
                       loadings_matrix=None,
                       feature_names=None,
                       fpr_svm=None, tpr_svm=None, auc_svm=None,
                       fpr_rf=None, tpr_rf=None, auc_rf=None,
                       importance_df=None,
                       metrics_svm=None, metrics_rf=None) -> Dict[str, str]:
    """
    一键生成所有常用图表，返回 {图名: base64} 字典
    :param df_cleaned: 清洗后标准化的 DataFrame
    :param pca_model: 拟合好的 PCA 对象
    :param loadings_matrix: 因子载荷矩阵 (n_features, n_factors)
    :param feature_names: 特征名列表
    :param 其他: ROC 曲线所需数据
    :param importance_df: 特征重要性 DataFrame
    :param metrics_svm, metrics_rf: 评估指标字典
    :return: 字典 { 'heatmap': 'base64...', 'scree': 'base64...', ... }
    """
    results = {}

    # 1. 相关性热力图
    if df_cleaned is not None:
        results['heatmap'] = draw_correlation_heatmap(df_cleaned, return_base64=True)

    # 2. PCA 碎石图
    if pca_model is not None:
        results['scree'] = draw_pca_scree(pca_model, return_base64=True)

    # 3. 因子载荷图
    if loadings_matrix is not None and feature_names is not None:
        results['loadings'] = draw_factor_loadings(loadings_matrix, feature_names, return_base64=True)

    # 4. ROC 曲线
    if all(x is not None for x in [fpr_svm, tpr_svm, auc_svm, fpr_rf, tpr_rf, auc_rf]):
        results['roc'] = draw_roc_curves(fpr_svm, tpr_svm, auc_svm, fpr_rf, tpr_rf, auc_rf, return_base64=True)

    # 5. 特征重要性
    if importance_df is not None:
        results['importance'] = draw_feature_importance(importance_df, top_n=10, return_base64=True)

    # 6. 消费分布箱线图
    if df_cleaned is not None and 'total_consume' in df_cleaned.columns:
        results['boxplot'] = draw_consumption_boxplot(df_cleaned, column='total_consume', return_base64=True)

    # 7. 指标对比柱状图
    if metrics_svm is not None and metrics_rf is not None:
        results['metrics_compare'] = draw_metrics_comparison(metrics_svm, metrics_rf, return_base64=True)

    return results


if __name__ == "__main__":
    # 简单测试（需要清洗后的数据）
    from backend.core.data_clean import clean_dataset
    result_clean = clean_dataset()
    df = result_clean['df_cleaned']

    # 测试相关性热力图
    heat = draw_correlation_heatmap(df, return_base64=False)
    print("热力图已保存")

    # 测试消费箱线图
    box = draw_consumption_boxplot(df, return_base64=False)
    print("箱线图已保存")