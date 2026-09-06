# backend/api/data_api.py
"""
数据管理接口路由
提供数据集概览、预览（分页）、清洗、导出功能
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import os
import pandas as pd
import numpy as np
import json

from backend.config import DATASET_FILE, DATA_PREFIX, DEBUG, OUTPUT_DIR
from backend.schemas.data_schema import (
    DataOverviewResponse,
    DataPreviewRequest,
    DataPreviewResponse,
    CleanStatisticsResponse,
    ExportRequest,
    ExportResponse
)
from backend.core.data_clean import clean_dataset, DataCleaner
from backend.utils.file_util import save_csv, ensure_dir, load_csv
from backend.utils.math_util import calc_distribution

router = APIRouter(prefix=DATA_PREFIX, tags=["数据管理"])

# 全局缓存清洗后的DataFrame（避免重复清洗）
_cleaned_df_cache = None
_scaler_cache = None
_cleaner_cache = None


def get_cleaner() -> DataCleaner:
    """获取或创建DataCleaner实例（单例模式）"""
    global _cleaner_cache
    if _cleaner_cache is None:
        _cleaner_cache = DataCleaner(DATASET_FILE)
        _cleaner_cache.load_data()
    return _cleaner_cache


def get_cleaned_df() -> pd.DataFrame:
    """获取清洗后的DataFrame（缓存）"""
    global _cleaned_df_cache, _scaler_cache
    if _cleaned_df_cache is None:
        cleaner = get_cleaner()
        result = cleaner.run_full_clean()
        _cleaned_df_cache = result['df_cleaned']
        _scaler_cache = result['scaler']
    return _cleaned_df_cache


def get_cleaned_scaler():
    """获取与清洗数据对应的StandardScaler（用于单用户预测时对齐输入）"""
    get_cleaned_df()
    return _scaler_cache


@router.get("/load", response_model=DataOverviewResponse)
async def load_dataset():
    """
    加载数据集，返回概览统计
    """
    try:
        df = load_csv(DATASET_FILE)
        total_samples = len(df)
        total_columns = len(df.columns)
        total_features = total_columns - 1  # 排除标签列
        missing_count = df.isnull().sum().sum()
        missing_rate = missing_count / (total_samples * total_columns) if total_samples > 0 else 0

        label_dist = df['repurchase_label'].value_counts().to_dict() if 'repurchase_label' in df.columns else {}

        # 检查重复行
        duplicate_count = df.duplicated().sum()

        memory_usage = f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"

        return DataOverviewResponse(
            total_samples=total_samples,
            total_features=total_features,
            total_columns=total_columns,
            missing_count=missing_count,
            missing_rate=missing_rate,
            label_distribution=label_dist,
            duplicate_count=duplicate_count,
            memory_usage=memory_usage,
            column_names=df.columns.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载数据集失败: {str(e)}")


@router.post("/preview", response_model=DataPreviewResponse)
async def preview_data(request: DataPreviewRequest):
    """
    分页预览原始数据
    """
    try:
        df = load_csv(DATASET_FILE)
        total_rows = len(df)
        page = request.page
        page_size = request.page_size
        total_pages = (total_rows + page_size - 1) // page_size

        if page > total_pages:
            page = total_pages

        start = (page - 1) * page_size
        end = start + page_size
        subset = df.iloc[start:end]

        # 将NaN转换为None以便JSON序列化
        data = subset.replace({pd.NA: None, np.nan: None}).to_dict(orient='records')

        return DataPreviewResponse(
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预览数据失败: {str(e)}")


@router.post("/clean", response_model=CleanStatisticsResponse)
async def clean_data():
    """
    执行数据清洗，返回清洗统计报告
    """
    try:
        result = clean_dataset(DATASET_FILE)
        report = result['report']
        filled = report.get('missing_filled', {})
        outlier = report.get('outlier_removed', {})

        return CleanStatisticsResponse(
            original_rows=report['original_rows'],
            cleaned_rows=report['cleaned_rows'],
            removed_rows=report['removed_rows'],
            missing_filled=filled,
            outlier_removed=outlier,
            clean_status=report.get('status', 'success')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据清洗失败: {str(e)}")


@router.post("/export", response_model=ExportResponse)
async def export_data(request: ExportRequest):
    """
    导出清洗后的数据集（CSV或Parquet）
    """
    try:
        if request.use_cleaned:
            df = get_cleaned_df()
        else:
            df = load_csv(DATASET_FILE)

        # 确定文件名
        if request.filename:
            base_name = request.filename
        else:
            import time
            timestamp = int(time.time())
            base_name = f"ecommerce_export_{timestamp}"

        ext = request.format.lower()
        if ext not in ['csv', 'parquet']:
            raise HTTPException(status_code=400, detail="不支持的导出格式，仅支持 csv 或 parquet")

        # 保存到 output/report 目录
        from backend.config import REPORT_DIR
        file_name = f"{base_name}.{ext}"
        file_path = os.path.join(REPORT_DIR, file_name)
        ensure_dir(REPORT_DIR)

        if ext == 'csv':
            save_csv(df, file_path)
        else:
            df.to_parquet(file_path, index=False)

        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / 1024 / 1024:.2f} MB"

        # 构造下载URL（FastAPI挂载的静态路由，需在main中配置）
        download_url = f"/api/data/download?path={file_path}"

        return ExportResponse(
            file_path=file_path,
            file_name=file_name,
            file_size=size_str,
            download_url=download_url
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出数据失败: {str(e)}")


@router.get("/download")
async def download_file(path: str = Query(..., description="文件绝对路径")):
    """
    下载导出文件（直接返回文件）
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    # 安全检查：只允许下载 output 目录下的文件（与启动目录无关）
    output_root = os.path.abspath(OUTPUT_DIR)
    abs_path = os.path.abspath(path)
    if not (abs_path == output_root or abs_path.startswith(output_root + os.sep)):
        raise HTTPException(status_code=403, detail="禁止访问该路径")
    return FileResponse(abs_path, filename=os.path.basename(abs_path))
