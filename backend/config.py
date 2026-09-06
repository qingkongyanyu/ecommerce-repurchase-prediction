# backend/config.py
"""
全局配置模块：统一管理路径、模型超参数、随机种子、缺失值填充策略等。
所有模块均从该文件读取配置，避免硬编码。
"""

import os
import numpy as np

# ---------- 路径配置 ----------
# 项目根目录（backend的上级目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集目录
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_FILE = os.path.join(DATASET_DIR, "ecommerce_repurchase_8500.csv")

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")      # 保存训练好的模型
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")    # 保存评估报告CSV
IMAGE_DIR = os.path.join(OUTPUT_DIR, "image")      # 保存后端生成的图片（备用）

# 后端静态资源缓存目录（用于存放生成的图表，通过FastAPI静态路由访问）
STATIC_DIR = os.path.join(BASE_DIR, "backend", "static")

# 自动创建所有需要的目录
for dir_path in [DATASET_DIR, OUTPUT_DIR, MODEL_DIR, REPORT_DIR, IMAGE_DIR, STATIC_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ---------- 随机种子（保证可复现） ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- 数据清洗配置 ----------
# 缺失值填充策略：数值型用均值，分类型用众数（在data_clean中具体实现）
NUMERIC_FILL_STRATEGY = "mean"   # 可选 'mean', 'median'
CATEGORICAL_FILL_STRATEGY = "mode"

# 异常值检测：使用3σ原则（Z-score > 3视为异常）
OUTLIER_SIGMA = 3.0

# 标准化方法：StandardScaler (z-score)
SCALER_METHOD = "standard"       # 目前仅支持standard

# ---------- 降维配置 ----------
# PCA保留的主成分数量（若设为None，则自动选择累计方差>=0.95的成分数）
PCA_N_COMPONENTS = None
PCA_VARIANCE_THRESHOLD = 0.95

# 因子分析的公因子数量（若设为None，则根据特征值>1或累计方差决定，此处默认设为10）
FACTOR_N_COMPONENTS = 10

# 融合降维后是否进行二次标准化（通常需要，因为PCA和因子分析得分尺度不同）
FUSION_SCALE = True

# ---------- 模型训练配置 ----------
# 数据集划分比例
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# 随机森林超参数（可调）
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_RANDOM_STATE = RANDOM_SEED

# SVM超参数（使用RBF核）
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"   # 或 'auto'
SVM_RANDOM_STATE = RANDOM_SEED

# 模型持久化文件命名
RF_MODEL_FILE = os.path.join(MODEL_DIR, "random_forest_model.joblib")
SVM_MODEL_FILE = os.path.join(MODEL_DIR, "svm_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")   # 保存标准化器用于新数据预测

# ---------- API 接口配置 ----------
API_PREFIX = "/api"
DATA_PREFIX = f"{API_PREFIX}/data"
DIM_PREFIX = f"{API_PREFIX}/dim"
MODEL_PREFIX = f"{API_PREFIX}/model"
SCREEN_PREFIX = f"{API_PREFIX}/screen"

# 大屏自动刷新间隔（秒）
SCREEN_REFRESH_INTERVAL = 30

# 跨域允许来源（开发环境允许所有，生产环境可限制）
CORS_ORIGINS = ["*"]

# ---------- 绘图配置 ----------
# 图表DPI及尺寸
FIGURE_DPI = 120
FIGURE_FIGSIZE = (10, 6)

# 颜色主题
COLOR_PALETTE = "Blues_d"  # seaborn配色

# ---------- 其他 ----------
# 是否启用详细日志（便于调试）
DEBUG = True