# backend/main.py
"""
FastAPI 服务入口文件
负责：
1. 创建 FastAPI 应用
2. 配置 CORS 跨域
3. 挂载静态文件路由（用于访问后端生成的图表）
4. 挂载所有 API 路由（data、dim、train、screen）
5. 全局异常捕获
6. 健康检查接口
7. uvicorn 启动入口
"""

import os
import sys
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn

# 添加项目根目录到 Python 路径（便于直接运行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from backend.config import (
    API_PREFIX,
    CORS_ORIGINS,
    STATIC_DIR,
    DEBUG,
    DATASET_DIR,
    OUTPUT_DIR,
    IMAGE_DIR,
    REPORT_DIR,
    MODEL_DIR,
    SCREEN_REFRESH_INTERVAL
)

# 导入路由
from backend.api.data_api import router as data_router
from backend.api.dim_api import router as dim_router
from backend.api.train_api import router as train_router
from backend.api.screen_api import router as screen_router

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO if DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------- 生命周期管理 ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动/关闭时的钩子函数
    """
    # 启动时执行
    logger.info("=" * 60)
    logger.info("[OK]  融合 PCA 与因子分析的电商用户复购预测系统")
    logger.info(f" 项目启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f" 调试模式: {DEBUG}")
    logger.info(f" 数据集目录: {DATASET_DIR}")
    logger.info(f" 输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # 检查数据集是否存在
    from backend.config import DATASET_FILE
    if not os.path.exists(DATASET_FILE):
        logger.warning(f"[WARN]  数据集文件不存在: {DATASET_FILE}")
        logger.warning("请先运行 data_generate/generate_dataset.py 生成数据集")
    else:
        logger.info(f"[OK]  数据集已就绪: {DATASET_FILE}")

    yield  # 应用运行中

    # 关闭时执行
    logger.info(" 系统正在关闭...")


# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="融合 PCA 与因子分析的电商用户复购预测系统",
    description="""
    ## 项目概述
    基于 8500 条电商用户数据，通过 PCA + 因子分析融合降维，
    使用 SVM 和随机森林进行复购预测，并提供可视化大屏接口。

    ## 核心技术
    - 数据清洗：缺失值填充、异常值剔除、特征标准化
    - 降维算法：PCA、因子分析、融合降维
    - 机器学习：SVM、随机森林
    - 可视化：matplotlib、seaborn

    ## 接口分组
    - `/api/data`：数据管理（加载、预览、清洗、导出）
    - `/api/dim`：降维分析（PCA、因子分析、融合）
    - `/api/model`：模型训练（SVM、随机森林、对比、预测）
    - `/api/screen`：大屏专用（实时统计、用户分层、模型指标）
    """,
    version="1.0.0",
    contact={
        "name": "电商复购预测系统开发团队",
        "email": "dev@example.com"
    },
    license_info={
        "name": "MIT License",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# ---------- CORS 跨域配置 ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 静态文件服务 ----------
# 挂载静态文件目录（用于访问后端生成的图表）
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- 挂载路由 ----------
app.include_router(data_router)
app.include_router(dim_router)
app.include_router(train_router)
app.include_router(screen_router)


# ---------- 根路径 ----------
@app.get("/")
async def root():
    """
    根路径：返回项目基本信息
    """
    return {
        "project": "融合 PCA 与因子分析的电商用户复购预测系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_prefix": API_PREFIX,
        "sample_count": 8500,
        "feature_count": 36,
        "models": ["SVM", "RandomForest"],
        "dimension_reduction": ["PCA", "FactorAnalysis", "Fusion"]
    }


@app.get("/health")
async def health_check():
    """
    健康检查接口（用于 Docker/负载均衡器）
    """
    from backend.config import DATASET_FILE
    dataset_exists = os.path.exists(DATASET_FILE)

    return {
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset_ready": dataset_exists,
        "debug_mode": DEBUG
    }


# ---------- 全局异常处理 ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    自定义 HTTP 异常处理
    """
    logger.error(f"HTTP异常: {exc.status_code} - {exc.detail} - 路径: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": exc.detail,
            "path": request.url.path,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理（捕获所有未处理的异常）
    """
    logger.error(f"未捕获异常: {str(exc)} - 路径: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": f"服务器内部错误: {str(exc)}",
            "path": request.url.path,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    )


# ---------- 开发模式：直接运行 ----------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info(" 启动 FastAPI 服务 (开发模式)")
    logger.info(f" 访问地址: http://127.0.0.1:8000")
    logger.info(f" API 文档: http://127.0.0.1:8000/docs")
    logger.info(f" ReDoc 文档: http://127.0.0.1:8000/redoc")
    logger.info("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,  # 调试模式下自动重载
        log_level="info" if DEBUG else "warning"
    )