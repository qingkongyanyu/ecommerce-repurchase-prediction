# TODO: 文件读写、csv/parquet导出工具函数
# backend/utils/file_util.py
"""
通用文件工具：CSV读写、Parquet读写、数据集导出、目录管理、图片保存等。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import json
import base64
from io import BytesIO
from backend.config import DATASET_DIR, OUTPUT_DIR, STATIC_DIR, DEBUG

# 确保目录存在
def ensure_dir(path: str) -> None:
    """创建目录（如果不存在）"""
    Path(path).mkdir(parents=True, exist_ok=True)

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载CSV文件，支持常见参数。
    :param file_path: csv文件路径
    :param kwargs: pandas.read_csv的其他参数
    :return: DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return pd.read_csv(file_path, encoding='utf-8-sig', **kwargs)

def save_csv(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """保存DataFrame为CSV（UTF-8 with BOM）"""
    ensure_dir(os.path.dirname(file_path))
    df.to_csv(file_path, index=index, encoding='utf-8-sig')
    if DEBUG:
        print(f"CSV已保存: {file_path}")

def load_parquet(file_path: str) -> pd.DataFrame:
    """加载Parquet文件"""
    return pd.read_parquet(file_path)

def save_parquet(df: pd.DataFrame, file_path: str) -> None:
    """保存DataFrame为Parquet"""
    ensure_dir(os.path.dirname(file_path))
    df.to_parquet(file_path, index=False)
    if DEBUG:
        print(f"Parquet已保存: {file_path}")

def save_json(data: dict, file_path: str) -> None:
    """保存字典为JSON文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if DEBUG:
        print(f"JSON已保存: {file_path}")

def load_json(file_path: str) -> dict:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_fig_to_base64(fig, dpi: int = 120, close_fig: bool = True) -> str:
    """
    将matplotlib图形对象转为base64编码的PNG字符串。
    :param fig: matplotlib.figure.Figure
    :param dpi: 分辨率
    :param close_fig: 是否关闭图形释放内存
    :return: base64字符串（不含data:image/png;base64,前缀，仅编码）
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    img_png = buffer.getvalue()
    buffer.close()
    if close_fig:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return base64.b64encode(img_png).decode('utf-8')

def save_fig_to_file(fig, file_path: str, dpi: int = 120) -> None:
    """保存图片到文件（PNG）"""
    ensure_dir(os.path.dirname(file_path))
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    if DEBUG:
        print(f"图片已保存: {file_path}")
    import matplotlib.pyplot as plt
    plt.close(fig)

def export_dataframe(df: pd.DataFrame, format: str = 'csv', path: Optional[str] = None) -> str:
    """
    导出DataFrame到指定格式，返回文件路径。
    默认保存在output/report目录下，文件名自动生成时间戳。
    """
    if path is None:
        import time
        timestamp = int(time.time())
        filename = f"export_{timestamp}.{format}"
        path = os.path.join(REPORT_DIR, filename) if format == 'csv' else os.path.join(OUTPUT_DIR, filename)
    else:
        ensure_dir(os.path.dirname(path))

    if format.lower() == 'csv':
        save_csv(df, path)
    elif format.lower() == 'parquet':
        save_parquet(df, path)
    else:
        raise ValueError(f"不支持的导出格式: {format}")
    return path