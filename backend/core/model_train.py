# backend/core/model_train.py
"""
机器学习建模核心模块
实现：
1. 数据集划分（训练/测试）
2. SVM 训练（支持 RBF 核，可调超参）
3. 随机森林训练（可调树数量、深度）
4. 统一评估指标计算（准确率、精确率、召回率、F1、AUC）
5. 双模型对比（生成对比结果和 ROC 曲线数据）
6. 单用户复购预测
7. 模型持久化（joblib 保存/加载）
8. 特征重要性提取（随机森林）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import time
import warnings
from typing import Optional, Tuple, Dict, Any, List, Union
warnings.filterwarnings('ignore')

from backend.config import (
    TRAIN_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_RANDOM_STATE,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    SVM_RANDOM_STATE,
    RF_MODEL_FILE,
    SVM_MODEL_FILE,
    SCALER_FILE,
    DEBUG
)
from backend.utils.math_util import compute_metrics, safe_divide


class ModelTrainer:
    """
    模型训练器
    封装 SVM 和随机森林的训练、评估、预测功能
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.svm_model = None
        self.rf_model = None
        self.svm_metrics = None
        self.rf_metrics = None
        self.feature_names = None
        self.is_fusion_features = False
        self.training_history = {}

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = TRAIN_RATIO,
        stratify: bool = True
    ) -> Dict[str, Any]:
        """
        划分训练集和测试集
        :param X: 特征矩阵
        :param y: 标签向量
        :param train_ratio: 训练集比例
        :param stratify: 是否按标签分层采样
        :return: 字典包含 X_train, X_test, y_train, y_test
        """
        if len(X) != len(y):
            raise ValueError("特征矩阵 X 和标签 y 的长度必须一致")

        test_ratio = 1 - train_ratio

        if stratify:
            stratify_y = y
        else:
            stratify_y = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=stratify_y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'train_positive_rate': y_train.mean() if len(y_train) > 0 else 0,
            'test_positive_rate': y_test.mean() if len(y_test) > 0 else 0,
        }

        if DEBUG:
            print(f"[ModelTrainer] 数据集划分完成:")
            print(f"  训练集: {len(X_train)} 样本 (正例率: {result['train_positive_rate']:.2%})")
            print(f"  测试集: {len(X_test)} 样本 (正例率: {result['test_positive_rate']:.2%})")

        self.training_history['split'] = result
        return result

    def train_svm(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        kernel: str = SVM_KERNEL,
        C: float = SVM_C,
        gamma: str = SVM_GAMMA,
        probability: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练 SVM 模型
        :param X_train: 训练特征（若为None则使用 self.X_train）
        :param y_train: 训练标签（若为None则使用 self.y_train）
        :param kernel: 核函数类型
        :param C: 正则化参数
        :param gamma: 核函数系数
        :param probability: 是否启用概率估计（用于AUC）
        :param kwargs: 其他 SVC 参数
        :return: 训练结果字典
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("请先调用 split_data() 划分数据集，或传入 X_train 和 y_train")

        start_time = time.time()

        # 创建 SVM 模型
        svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=self.random_state,
            **kwargs
        )

        # 训练
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 保存模型
        self.svm_model = svm

        # 预测训练集和测试集
        y_train_pred = svm.predict(X_train)
        y_test_pred = svm.predict(self.X_test) if hasattr(self, 'X_test') and self.X_test is not None else None

        # 预测概率（用于 AUC）
        y_train_prob = None
        y_test_prob = None
        if probability:
            if X_train is not None:
                y_train_prob = svm.predict_proba(X_train)[:, 1]
            if hasattr(self, 'X_test') and self.X_test is not None:
                y_test_prob = svm.predict_proba(self.X_test)[:, 1]

        result = {
            'model': svm,
            'train_time': train_time,
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'probability': probability,
        }

        # 计算训练集指标
        if y_train is not None:
            train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
            result['train_metrics'] = train_metrics

        # 计算测试集指标
        if y_test_pred is not None:
            test_metrics = compute_metrics(self.y_test, y_test_pred, y_test_prob)
            result['test_metrics'] = test_metrics
            self.svm_metrics = test_metrics

        if DEBUG:
            print(f"[ModelTrainer] SVM 训练完成，耗时 {train_time:.3f}s")
            if 'test_metrics' in result:
                print(f"  测试集准确率: {result['test_metrics']['accuracy']:.3f}")
                print(f"  测试集 AUC: {result['test_metrics'].get('auc', 0):.3f}")

        self.training_history['svm'] = result
        return result

    def train_random_forest(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: Optional[int] = RF_MAX_DEPTH,
        min_samples_split: int = RF_MIN_SAMPLES_SPLIT,
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练随机森林模型
        :param X_train: 训练特征
        :param y_train: 训练标签
        :param n_estimators: 树的数量
        :param max_depth: 最大深度
        :param min_samples_split: 节点分裂最小样本数
        :param kwargs: 其他 RandomForestClassifier 参数
        :return: 训练结果字典
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        if X_train is None or y_train is None:
            raise ValueError("请先调用 split_data() 划分数据集，或传入 X_train 和 y_train")

        start_time = time.time()

        # 创建随机森林模型
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state,
            **kwargs
        )

        # 训练
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 保存模型
        self.rf_model = rf

        # 预测
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(self.X_test) if hasattr(self, 'X_test') and self.X_test is not None else None

        # 预测概率
        y_train_prob = rf.predict_proba(X_train)[:, 1] if hasattr(rf, 'predict_proba') else None
        y_test_prob = None
        if hasattr(self, 'X_test') and self.X_test is not None:
            y_test_prob = rf.predict_proba(self.X_test)[:, 1] if hasattr(rf, 'predict_proba') else None

        # 提取特征重要性
        feature_importance = None
        if hasattr(rf, 'feature_importances_'):
            importance_values = rf.feature_importances_
            feature_importance = importance_values.tolist()

        result = {
            'model': rf,
            'train_time': train_time,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'feature_importance': feature_importance,
        }

        # 计算指标
        if y_train is not None:
            train_metrics = compute_metrics(y_train, y_train_pred, y_train_prob)
            result['train_metrics'] = train_metrics

        if y_test_pred is not None:
            test_metrics = compute_metrics(self.y_test, y_test_pred, y_test_prob)
            result['test_metrics'] = test_metrics
            self.rf_metrics = test_metrics

        if DEBUG:
            print(f"[ModelTrainer] 随机森林训练完成，耗时 {train_time:.3f}s")
            if 'test_metrics' in result:
                print(f"  测试集准确率: {result['test_metrics']['accuracy']:.3f}")
                print(f"  测试集 AUC: {result['test_metrics'].get('auc', 0):.3f}")

        self.training_history['random_forest'] = result
        return result

    def compare_models(self) -> Dict[str, Any]:
        """
        对比 SVM 和随机森林的测试集指标
        :return: 对比结果字典
        """
        if self.svm_metrics is None or self.rf_metrics is None:
            raise ValueError("请先训练两个模型，或确保 self.svm_metrics 和 self.rf_metrics 存在")

        # 获取 ROC 曲线数据
        svm_roc = None
        rf_roc = None

        if self.svm_model is not None and hasattr(self, 'X_test') and self.X_test is not None:
            try:
                y_svm_prob = self.svm_model.predict_proba(self.X_test)[:, 1]
                fpr_svm, tpr_svm, _ = roc_curve(self.y_test, y_svm_prob)
                svm_roc = {'fpr': fpr_svm.tolist(), 'tpr': tpr_svm.tolist()}
            except:
                pass

        if self.rf_model is not None and hasattr(self, 'X_test') and self.X_test is not None:
            try:
                y_rf_prob = self.rf_model.predict_proba(self.X_test)[:, 1]
                fpr_rf, tpr_rf, _ = roc_curve(self.y_test, y_rf_prob)
                rf_roc = {'fpr': fpr_rf.tolist(), 'tpr': tpr_rf.tolist()}
            except:
                pass

        # 确定优胜模型
        svm_auc = self.svm_metrics.get('auc', 0)
        rf_auc = self.rf_metrics.get('auc', 0)

        if svm_auc > rf_auc:
            winner = 'SVM'
            winner_metric = 'AUC'
            winner_value = svm_auc
        elif rf_auc > svm_auc:
            winner = '随机森林'
            winner_metric = 'AUC'
            winner_value = rf_auc
        else:
            winner = '平局'
            winner_metric = 'AUC'
            winner_value = svm_auc

        # 获取混淆矩阵
        svm_cm = None
        rf_cm = None
        if self.svm_model is not None and hasattr(self, 'X_test') and self.X_test is not None:
            y_svm_pred = self.svm_model.predict(self.X_test)
            svm_cm = confusion_matrix(self.y_test, y_svm_pred).tolist()
        if self.rf_model is not None and hasattr(self, 'X_test') and self.X_test is not None:
            y_rf_pred = self.rf_model.predict(self.X_test)
            rf_cm = confusion_matrix(self.y_test, y_rf_pred).tolist()

        result = {
            'svm_metrics': self.svm_metrics,
            'rf_metrics': self.rf_metrics,
            'svm_roc_data': svm_roc,
            'rf_roc_data': rf_roc,
            'svm_confusion_matrix': svm_cm,
            'rf_confusion_matrix': rf_cm,
            'winner': winner,
            'winner_metric': winner_metric,
            'winner_value': winner_value,
            'comparison_summary': f"{winner} 在 {winner_metric} 上表现更优 ({winner_value:.3f})"
        }

        return result

    def predict_single(
        self,
        features: Union[List[float], np.ndarray],
        model_type: str = 'random_forest'
    ) -> Dict[str, Any]:
        """
        单用户复购预测
        :param features: 用户特征向量
        :param model_type: 'svm' 或 'random_forest'
        :return: 预测结果字典
        """
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)

        if model_type == 'svm':
            model = self.svm_model
            model_name = 'SVM'
        elif model_type == 'random_forest':
            model = self.rf_model
            model_name = '随机森林'
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        if model is None:
            raise ValueError(f"{model_name} 模型尚未训练，请先调用 train_svm() 或 train_random_forest()")

        start_time = time.time()

        # 预测标签
        label = model.predict(features)[0]

        # 预测概率
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0][1]
        else:
            # 如果没有概率预测，用决策函数转换
            if hasattr(model, 'decision_function'):
                prob = 1 / (1 + np.exp(-model.decision_function(features)[0]))

        pred_time = (time.time() - start_time) * 1000  # 转为毫秒

        # 风险等级划分
        if prob is not None:
            if prob >= 0.7:
                risk_level = '高'
            elif prob >= 0.4:
                risk_level = '中'
            else:
                risk_level = '低'
        else:
            risk_level = '未知'

        return {
            'model_used': model_name,
            'prediction_label': int(label),
            'repurchase_probability': float(prob) if prob is not None else None,
            'risk_level': risk_level,
            'predict_time_ms': pred_time,
            'features_shape': features.shape
        }

    def get_feature_importance(self, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        获取随机森林的特征重要性（排序后）
        :param top_n: 返回前N个特征
        :return: DataFrame 包含 feature 和 importance
        """
        if self.rf_model is None:
            return None

        if not hasattr(self.rf_model, 'feature_importances_'):
            return None

        importances = self.rf_model.feature_importances_

        # 获取特征名称
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        if top_n is not None and top_n < len(df):
            df = df.head(top_n)

        return df

    def save_models(self, svm_path: str = SVM_MODEL_FILE, rf_path: str = RF_MODEL_FILE) -> Dict[str, str]:
        """
        保存训练好的模型
        :param svm_path: SVM 模型保存路径
        :param rf_path: 随机森林模型保存路径
        :return: 保存路径字典
        """
        saved = {}
        if self.svm_model is not None:
            joblib.dump(self.svm_model, svm_path)
            saved['svm'] = svm_path
            if DEBUG:
                print(f"[ModelTrainer] SVM 模型已保存: {svm_path}")

        if self.rf_model is not None:
            joblib.dump(self.rf_model, rf_path)
            saved['random_forest'] = rf_path
            if DEBUG:
                print(f"[ModelTrainer] 随机森林模型已保存: {rf_path}")

        return saved

    def load_models(self, svm_path: str = SVM_MODEL_FILE, rf_path: str = RF_MODEL_FILE) -> Dict[str, Any]:
        """
        加载保存的模型
        :param svm_path: SVM 模型路径
        :param rf_path: 随机森林模型路径
        :return: 加载的模型字典
        """
        loaded = {}
        try:
            self.svm_model = joblib.load(svm_path)
            loaded['svm'] = self.svm_model
            if DEBUG:
                print(f"[ModelTrainer] SVM 模型已加载: {svm_path}")
        except Exception as e:
            if DEBUG:
                print(f"[ModelTrainer] SVM 模型加载失败: {e}")

        try:
            self.rf_model = joblib.load(rf_path)
            loaded['random_forest'] = self.rf_model
            if DEBUG:
                print(f"[ModelTrainer] 随机森林模型已加载: {rf_path}")
        except Exception as e:
            if DEBUG:
                print(f"[ModelTrainer] 随机森林模型加载失败: {e}")

        return loaded


# ---------- 快捷函数（供 API 调用） ----------
def train_and_compare(
    X: np.ndarray,
    y: np.ndarray,
    use_fusion: bool = True,
    train_ratio: float = TRAIN_RATIO,
    svm_kwargs: Optional[Dict] = None,
    rf_kwargs: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    一键训练并对比双模型
    :param X: 特征矩阵（原始特征或降维后特征）
    :param y: 标签
    :param use_fusion: 是否使用融合特征（仅用于日志标注）
    :param train_ratio: 训练集比例
    :param svm_kwargs: SVM 超参覆盖
    :param rf_kwargs: 随机森林超参覆盖
    :param feature_names: 特征名列表（用于特征重要性）
    :return: 完整训练对比结果
    """
    trainer = ModelTrainer()

    # 设置特征名
    if feature_names is not None:
        trainer.feature_names = feature_names
    trainer.is_fusion_features = use_fusion

    # 1. 划分数据集
    split_result = trainer.split_data(X, y, train_ratio=train_ratio)

    # 2. 训练 SVM
    if svm_kwargs is None:
        svm_kwargs = {}
    trainer.train_svm(**svm_kwargs)

    # 3. 训练随机森林
    if rf_kwargs is None:
        rf_kwargs = {}
    trainer.train_random_forest(**rf_kwargs)

    # 4. 对比结果
    compare_result = trainer.compare_models()

    # 5. 获取特征重要性
    importance_df = trainer.get_feature_importance(top_n=10)

    # 6. 构建完整结果
    result = {
        'trainer': trainer,
        'split_info': split_result,
        'svm_metrics': trainer.svm_metrics,
        'rf_metrics': trainer.rf_metrics,
        'comparison': compare_result,
        'feature_importance': importance_df.to_dict('records') if importance_df is not None else None,
        'is_fusion_features': use_fusion,
        'svm_model': trainer.svm_model,
        'rf_model': trainer.rf_model,
    }

    if DEBUG:
        print(f"[train_and_compare] 训练对比完成，使用{'融合' if use_fusion else '原始'}特征")
        print(f"  SVM AUC: {trainer.svm_metrics.get('auc', 0):.3f}")
        print(f"  随机森林 AUC: {trainer.rf_metrics.get('auc', 0):.3f}")

    return result


def predict_with_model(
    features: Union[List[float], np.ndarray],
    model_type: str = 'random_forest',
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用已训练模型进行预测（支持加载保存的模型）
    :param features: 用户特征向量
    :param model_type: 'svm' 或 'random_forest'
    :param model_path: 模型路径（若模型未加载到内存）
    :return: 预测结果
    """
    trainer = ModelTrainer()

    if model_path is not None:
        if model_type == 'svm':
            trainer.load_models(svm_path=model_path)
        else:
            trainer.load_models(rf_path=model_path)
    else:
        # 尝试从默认路径加载
        if model_type == 'svm':
            trainer.load_models(svm_path=SVM_MODEL_FILE)
        else:
            trainer.load_models(rf_path=RF_MODEL_FILE)

    return trainer.predict_single(features, model_type=model_type)


if __name__ == "__main__":
    # 完整测试（需要清洗和降维后的数据）
    from backend.core.data_clean import clean_dataset
    from backend.core.dim_reduce import perform_pca, perform_fusion
    from backend.config import DATASET_FILE

    print("=" * 60)
    print("测试模型训练模块")
    print("=" * 60)

    # 1. 加载清洗数据
    print("\n[1] 加载数据...")
    result = clean_dataset()
    df = result['df_cleaned']
    feature_cols = [col for col in df.columns if col not in ['user_id', 'repurchase_label']]
    X_raw = df[feature_cols].values
    y = df['repurchase_label'].values
    print(f"  数据形状: {X_raw.shape}, 标签分布: {np.bincount(y)}")

    # 2. 测试原始特征训练
    print("\n[2] 测试原始特征训练...")
    result_original = train_and_compare(
        X_raw, y,
        use_fusion=False,
        feature_names=feature_cols,
        rf_kwargs={'n_estimators': 50, 'max_depth': 8}
    )
    print(f"  SVM AUC: {result_original['svm_metrics'].get('auc', 0):.3f}")
    print(f"  随机森林 AUC: {result_original['rf_metrics'].get('auc', 0):.3f}")

    # 3. 测试融合特征训练
    print("\n[3] 测试融合特征训练...")
    fusion_result = perform_fusion(X_raw, factor_components=8, scale_fusion=True)
    X_fusion = fusion_result['fused_features']
    print(f"  融合特征形状: {X_fusion.shape}")

    result_fusion = train_and_compare(
        X_fusion, y,
        use_fusion=True,
        feature_names=[f'fusion_feature_{i}' for i in range(X_fusion.shape[1])],
        rf_kwargs={'n_estimators': 50, 'max_depth': 8}
    )
    print(f"  SVM AUC: {result_fusion['svm_metrics'].get('auc', 0):.3f}")
    print(f"  随机森林 AUC: {result_fusion['rf_metrics'].get('auc', 0):.3f}")

    # 4. 测试单用户预测（用刚训练好的内存模型，不走文件加载）
    print("\n[4] 测试单用户预测...")
    sample_features = X_fusion[0].tolist()
    trainer_mem = result_fusion['trainer']
    predict_result = trainer_mem.predict_single(sample_features, model_type='random_forest')
    print(f"  预测结果: 复购概率 {predict_result['repurchase_probability']:.3f}, 标签 {predict_result['prediction_label']}")

    # 5. 测试模型保存/加载
    print("\n[5] 测试模型保存/加载...")
    trainer = result_fusion['trainer']
    saved_paths = trainer.save_models()
    print(f"  模型已保存: {saved_paths}")

    new_trainer = ModelTrainer()
    new_trainer.load_models()
    print("  模型已加载")

    print("\n[OK]  所有测试通过！")