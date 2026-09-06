# backend/core/data_clean.py
"""
数据清洗模块：缺失值填充、异常值检测与剔除、特征标准化。
所有操作均基于 config.py 中的配置，并返回详细的清洗统计报告。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from backend.config import (
    NUMERIC_FILL_STRATEGY, CATEGORICAL_FILL_STRATEGY,
    OUTLIER_SIGMA, SCALER_METHOD, RANDOM_SEED, DEBUG
)
from backend.utils.file_util import load_csv, save_csv
from backend.utils.math_util import safe_divide


class DataCleaner:
    """
    数据清洗器
    功能：
    1. 加载原始数据集
    2. 缺失值填充（数值均值/中位数，分类众数）
    3. 异常值剔除（3σ原则）
    4. 特征标准化（StandardScaler）
    5. 返回清洗后数据及统计报告
    """

    def __init__(self, file_path: str):
        """
        :param file_path: CSV数据集路径
        """
        self.file_path = file_path
        self.df_raw = None
        self.df_cleaned = None
        self.scaler = None
        self.cleaning_report = {
            'original_rows': 0,
            'cleaned_rows': 0,
            'removed_rows': 0,
            'missing_filled': {},
            'outlier_removed': {},
            'status': 'pending'
        }
        # 记录数值列和分类列（用于区分填充策略）
        self.numeric_cols = []
        self.categorical_cols = []
        self.label_col = 'repurchase_label'
        self.id_col = 'user_id'

    def load_data(self) -> pd.DataFrame:
        """加载原始数据集"""
        self.df_raw = load_csv(self.file_path)
        self.cleaning_report['original_rows'] = len(self.df_raw)
        if DEBUG:
            print(f"[DataCleaner] 成功加载数据集: {len(self.df_raw)} 行, {len(self.df_raw.columns)} 列")
        return self.df_raw

    def _classify_columns(self, df: pd.DataFrame) -> None:
        """
        自动分类数值列和分类列（排除ID列和标签列）
        策略：数值列中唯一值数量 <= 10 视为分类编码列（如会员等级、设备类型等）。
        注意：数据集存在缺失值，注入 NaN 后 int 列会被 pandas 自动转为 float64，
        因此不能依赖 dtype 判断，必须以唯一值数量为准。
        """
        self.numeric_cols = []
        self.categorical_cols = []

        for col in df.columns:
            if col in [self.id_col, self.label_col]:
                continue

            # 判断是否为数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                # 唯一值很少的数值列视为分类编码特征
                if df[col].nunique() <= 10:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                # 非数值类型视为分类
                self.categorical_cols.append(col)

        if DEBUG:
            print(f"[DataCleaner] 数值列: {len(self.numeric_cols)} 个")
            print(f"[DataCleaner] 分类列: {len(self.categorical_cols)} 个")

    def fill_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        缺失值填充
        - 数值列：均值（或中位数，由config控制）
        - 分类列：众数
        """
        filled_counts = {}
        df_filled = df.copy()

        # 填充数值列
        for col in self.numeric_cols:
            if df_filled[col].isnull().any():
                if NUMERIC_FILL_STRATEGY == 'mean':
                    fill_val = df_filled[col].mean()
                elif NUMERIC_FILL_STRATEGY == 'median':
                    fill_val = df_filled[col].median()
                else:
                    fill_val = df_filled[col].mean()
                missing_count = df_filled[col].isnull().sum()
                df_filled[col].fillna(fill_val, inplace=True)
                filled_counts[col] = missing_count

        # 填充分类列
        for col in self.categorical_cols:
            if df_filled[col].isnull().any():
                mode_val = df_filled[col].mode()[0] if not df_filled[col].mode().empty else 0
                missing_count = df_filled[col].isnull().sum()
                df_filled[col].fillna(mode_val, inplace=True)
                filled_counts[col] = missing_count

        # 标签列如果有缺失，填充为0（不复购）
        if self.label_col in df_filled.columns and df_filled[self.label_col].isnull().any():
            missing_count = df_filled[self.label_col].isnull().sum()
            df_filled[self.label_col].fillna(0, inplace=True)
            filled_counts[self.label_col] = missing_count

        return df_filled, filled_counts

    def remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        使用3σ原则剔除异常值（仅针对数值列）
        """
        outlier_counts = {}
        df_cleaned = df.copy()
        total_removed = 0

        for col in self.numeric_cols:
            if col in df_cleaned.columns and df_cleaned[col].dtype in ['float64', 'int64']:
                mean_val = df_cleaned[col].mean()
                std_val = df_cleaned[col].std()
                if std_val == 0:
                    continue
                lower_bound = mean_val - OUTLIER_SIGMA * std_val
                upper_bound = mean_val + OUTLIER_SIGMA * std_val
                # 标记异常值位置
                mask_outlier = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                count_outlier = mask_outlier.sum()
                if count_outlier > 0:
                    outlier_counts[col] = count_outlier
                    # 将异常值替换为边界值（而非直接删除整行，保留样本量）
                    df_cleaned.loc[mask_outlier, col] = np.clip(
                        df_cleaned.loc[mask_outlier, col], lower_bound, upper_bound
                    )
                    total_removed += count_outlier

        # 同时剔除包含NaN的行（如果有残余缺失）
        before_drop = len(df_cleaned)
        df_cleaned.dropna(inplace=True)
        after_drop = len(df_cleaned)
        if before_drop - after_drop > 0:
            outlier_counts['_dropna'] = before_drop - after_drop

        return df_cleaned, outlier_counts

    def standardize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        标准化特征（排除ID列和标签列）
        仅对连续数值列做 z-score 标准化；分类编码列（会员等级、设备等）保留原始编码，
        以便大屏按业务口径做用户分层统计。
        返回标准化后的DataFrame和训练好的Scaler
        """
        df_scaled = df.copy()

        if self.numeric_cols:
            # 传入 DataFrame（而非 ndarray），使 scaler 记录 feature_names_in_，便于预测时对齐
            X = df_scaled[self.numeric_cols]
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 替换连续数值特征为标准化值
            for i, col in enumerate(self.numeric_cols):
                df_scaled[col] = X_scaled[:, i]
        else:
            # 兜底：没有连续列时仍提供一个可用的 scaler
            self.scaler = StandardScaler()
            self.scaler.fit(df_scaled[df_scaled.columns[:1]].astype(float).values)

        return df_scaled, self.scaler

    def run_full_clean(self) -> Dict:
        """
        执行完整清洗流水线：加载 → 分类列 → 填充缺失 → 异常值处理 → 标准化
        :return: 包含清洗后DataFrame和统计信息的字典
        """
        if self.df_raw is None:
            self.load_data()

        # 1. 列分类
        self._classify_columns(self.df_raw)

        # 2. 缺失值填充
        df_filled, filled_counts = self.fill_missing(self.df_raw)
        self.cleaning_report['missing_filled'] = filled_counts

        # 3. 异常值剔除
        df_no_outliers, outlier_counts = self.remove_outliers(df_filled)
        self.cleaning_report['outlier_removed'] = outlier_counts

        # 4. 标准化
        df_standardized, scaler = self.standardize(df_no_outliers)
        self.scaler = scaler
        self.df_cleaned = df_standardized

        # 5. 更新报告
        self.cleaning_report['cleaned_rows'] = len(df_standardized)
        self.cleaning_report['removed_rows'] = self.cleaning_report['original_rows'] - len(df_standardized)
        self.cleaning_report['status'] = 'success'

        if DEBUG:
            print(f"[DataCleaner] 清洗完成: 原始 {self.cleaning_report['original_rows']} 行 → 清洗后 {self.cleaning_report['cleaned_rows']} 行")
            print(f"[DataCleaner] 填充缺失值: {sum(filled_counts.values())} 个")
            print(f"[DataCleaner] 处理异常值: {sum(outlier_counts.values())} 个")

        return {
            'df_cleaned': self.df_cleaned,
            'scaler': self.scaler,
            'report': self.cleaning_report
        }

    def get_cleaned_data(self) -> pd.DataFrame:
        """获取清洗后的DataFrame"""
        if self.df_cleaned is None:
            raise ValueError("请先运行 run_full_clean() 方法")
        return self.df_cleaned

    def get_scaler(self) -> StandardScaler:
        """获取训练好的标准化器"""
        if self.scaler is None:
            raise ValueError("请先运行 run_full_clean() 方法")
        return self.scaler

    def get_report(self) -> Dict:
        """获取清洗统计报告"""
        return self.cleaning_report


# ---------- 便捷函数（供API调用） ----------
def clean_dataset(file_path: str = None) -> Dict:
    """
    一键清洗数据集
    :param file_path: CSV路径，默认使用config中的路径
    :return: 清洗结果字典，包含 df_cleaned, scaler, report
    """
    from backend.config import DATASET_FILE
    if file_path is None:
        file_path = DATASET_FILE

    cleaner = DataCleaner(file_path)
    result = cleaner.run_full_clean()

    # 额外补充：清洗后数据统计
    df = result['df_cleaned']
    result['stats'] = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'label_distribution': df['repurchase_label'].value_counts().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    }
    return result


if __name__ == "__main__":
    # 简单测试
    from backend.config import DATASET_FILE
    result = clean_dataset(DATASET_FILE)
    print(f"清洗后形状: {result['df_cleaned'].shape}")
    print(f"标签分布: {result['stats']['label_distribution']}")