"""
从 Qlib 借鉴: 模型抽象层 (BaseModel)
从 FreqTrade 借鉴: 策略接口

用法:
    from quant_agent.ml.models import MLSignalModel, TrainTestSplit

    model = MLSignalModel(
        model_type="random_forest",
        features=["sma20","sma50","rsi14","volume"],
        label_col="target",
    )
    metrics = model.walk_forward(df, n_splits=5)
    print(f"平均准确率: {metrics['accuracy_mean']:.1f}%")
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit


class TrainTestSplit:
    """时间序列训练/测试集划分 — 防数据泄露"""

    @staticmethod
    def split(df, train_pct=0.8):
        """按时间顺序切分，不打乱"""
        split_idx = int(len(df) * train_pct)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

    @staticmethod
    def walk_forward(df, n_splits=5):
        """滚动时间序列交叉验证"""
        return TimeSeriesSplit(n_splits=n_splits).split(df)


class MLSignalModel:
    """
    从 Qlib BaseModel 借鉴的模型抽象层

    支持:
    - 随机森林 / GBDT / 逻辑回归
    - 时间序列交叉验证
    - 特征重要性输出
    - Walk-forward 验证
    """

    MODEL_REGISTRY = {
        "random_forest": RandomForestClassifier,
        "gbdt": GradientBoostingClassifier,
    }

    def __init__(
        self,
        model_type: str = "random_forest",
        features: Optional[List[str]] = None,
        label_col: str = "target",
        model_params: Optional[Dict] = None,
    ):
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_type}. Choose from {list(self.MODEL_REGISTRY.keys())}")

        default_params = {
            "random_forest": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "gbdt": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        }

        params = {**default_params.get(model_type, {}), **(model_params or {})}
        self.model = self.MODEL_REGISTRY[model_type](**params)
        self.features = features or []
        self.label_col = label_col
        self.fitted = False

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 FreqTrade 借鉴: 特征工程自动化"""
        result = df.copy()
        c = result["Close"] if "Close" in result else result.get("close", result.iloc[:, 3])

        # 技术指标特征
        result["sma10"] = c.rolling(10).mean()
        result["sma20"] = c.rolling(20).mean()
        result["sma50"] = c.rolling(50).mean()
        result["sma200"] = c.rolling(200).mean()

        # 动量特征
        result["ret1"] = c.pct_change(1)
        result["ret5"] = c.pct_change(5)
        result["ret20"] = c.pct_change(20)

        # 波动率特征
        result["volatility"] = c.pct_change().rolling(20).std()

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        result["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # 成交量特征
        if "Volume" in result:
            result["volume_ma"] = result["Volume"].rolling(20).mean()
            result["volume_ratio"] = result["Volume"] / result["volume_ma"]

        # 生成标签: 未来 N 天涨跌
        result["target"] = (c.shift(-5) > c).astype(int)

        return result

    def prepare_data(self, df: pd.DataFrame):
        """准备训练数据"""
        df = self.engineer_features(df)

        if not self.features:
            self.features = [c for c in ["sma10","sma20","sma50","sma200","ret1","ret5","ret20","rsi14","volatility"]
                           if c in df.columns]

        data = df[self.features + [self.label_col]].dropna()
        return data[self.features].values, data[self.label_col].values

    def fit(self, df: pd.DataFrame):
        """训练模型"""
        X, y = self.prepare_data(df)
        self.model.fit(X, y)
        self.fitted = True

        # 输出特征重要性
        if hasattr(self.model, "feature_importances_"):
            print("\n特征重要性:")
            for name, imp in sorted(zip(self.features, self.model.feature_importances_),
                                    key=lambda x: -x[1]):
                print(f"  {name:<10} {imp*100:.1f}%")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call .fit() first.")
        X, _ = self.prepare_data(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """预测概率（从 FreqTrade 借鉴）"""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call .fit() first.")
        X, _ = self.prepare_data(df)
        return self.model.predict_proba(X)

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """在测试集上评估"""
        train, test = TrainTestSplit.split(df, train_pct=0.7)
        self.fit(train)

        X_test, y_test = self.prepare_data(test)
        pred = self.model.predict(X_test)

        return {
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "total_samples": len(y_test),
            "positive_pct": round(y_test.mean() * 100, 1),
        }

    def walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> Dict:
        """从 Qlib 借鉴: Walk-forward 验证"""
        df = self.engineer_features(df)
        data = df[self.features + [self.label_col]].dropna()
        X, y = data[self.features].values, data[self.label_col].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        accuracies = []
        precisions = []
        recalls = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self.MODEL_REGISTRY[list(self.MODEL_REGISTRY.keys())[0]](**self.model.get_params())
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            accuracies.append(accuracy_score(y_test, pred))
            precisions.append(precision_score(y_test, pred, zero_division=0))
            recalls.append(recall_score(y_test, pred, zero_division=0))

        return {
            "accuracy_mean": round(np.mean(accuracies) * 100, 1),
            "accuracy_std": round(np.std(accuracies), 3),
            "precision_mean": round(np.mean(precisions), 3),
            "recall_mean": round(np.mean(recalls), 3),
            "n_splits": n_splits,
            "per_split": [round(a * 100, 1) for a in accuracies],
        }


def walk_forward_validation(df, strategy_fn, n_splits=5):
    """从 FreqTrade 借鉴: 策略滚动验证"""
    results = []
    split_size = len(df) // (n_splits + 1)

    for i in range(n_splits):
        train_end = (i + 1) * split_size
        test_end = train_end + split_size if i < n_splits - 1 else len(df)

        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        train_signal = strategy_fn(train)
        test_signal = strategy_fn(test)

        results.append({
            "fold": i + 1,
            "train": f"{train.index[0].date()} ~ {train.index[-1].date()}",
            "test": f"{test.index[0].date()} ~ {test.index[-1].date()}",
        })

    return results
