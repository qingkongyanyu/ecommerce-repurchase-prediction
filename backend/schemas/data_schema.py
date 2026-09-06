# backend/schemas/data_schema.py
"""
数据管理模块的请求/响应模型
涵盖：数据集加载预览、清洗、导出
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class DataOverviewResponse(BaseModel):
    """数据集概览响应"""
    total_samples: int = Field(..., description="总样本数")
    total_features: int = Field(..., description="特征数量（不含标签）")
    total_columns: int = Field(..., description="总列数（含标签）")
    missing_count: int = Field(..., description="缺失值总数")
    missing_rate: float = Field(..., description="缺失值比例（0~1）")
    label_distribution: Dict[int, int] = Field(..., description="复购标签分布，如 {0: 4500, 1: 4000}")
    duplicate_count: int = Field(0, description="重复行数")
    memory_usage: str = Field(..., description="DataFrame内存占用（如 '2.3 MB'）")
    column_names: List[str] = Field(..., description="所有列名列表")


class DataPreviewRequest(BaseModel):
    """数据预览请求（分页）"""
    page: int = Field(1, ge=1, description="页码，从1开始")
    page_size: int = Field(10, ge=1, le=100, description="每页条数")


class DataPreviewResponse(BaseModel):
    """数据预览响应（分页）"""
    total_rows: int = Field(..., description="总行数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页条数")
    total_pages: int = Field(..., description="总页数")
    data: List[Dict[str, Any]] = Field(..., description="当前页数据，键为列名，值为对应数值（NaN转为null）")


class CleanStatisticsResponse(BaseModel):
    """清洗统计信息"""
    original_rows: int = Field(..., description="原始行数")
    cleaned_rows: int = Field(..., description="清洗后行数")
    removed_rows: int = Field(..., description="移除的异常/重复行数")
    missing_filled: Dict[str, int] = Field(..., description="各列缺失值填充数量，如 {'user_age': 5}")
    outlier_removed: Dict[str, int] = Field(..., description="各列异常值剔除数量")
    clean_status: str = Field(..., description="清洗状态，如 'success' 或 'failed'")


class ExportRequest(BaseModel):
    """数据导出请求"""
    format: str = Field("csv", description="导出格式，支持 'csv' 或 'parquet'")
    filename: Optional[str] = Field(None, description="自定义文件名（不含扩展名），默认自动生成时间戳")
    use_cleaned: bool = Field(True, description="是否导出清洗后的数据，否则导出原始数据")


class ExportResponse(BaseModel):
    """数据导出响应"""
    file_path: str = Field(..., description="导出文件的绝对路径")
    file_name: str = Field(..., description="文件名")
    file_size: str = Field(..., description="文件大小（如 '2.1 MB'）")
    download_url: str = Field(..., description="下载接口URL（/api/data/download?path=...）")