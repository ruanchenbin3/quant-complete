"""
从 Qlib 借鉴: ML Pipeline (数据处理 → 特征 → 模型 → 回测)

完整流程:
    pipeline = MLPipeline()
    pipeline.run("BTC-USD", model_type="random_forest")
    pipeline.report()
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from .models import MLSignalModel


class MLPipeline:
    """
    ML 量化完整流程

    1. 获取数据
    2. 特征工程
    3. 训练/验证
    4. 回测
    5. 报告
    """

    def __init__(self):
        self.df = None
        self.model = None
        self.results = {}
        self.equity_curve = None

    def fetch_data(self, symbol: str = "BTC-USD", period: str = "3y"):
        """获取数据"""
        self.df = yf.download(symbol, period=period, interval="1d")
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.droplevel(1)
        self.symbol = symbol
        return self

    def run(self, symbol: str = "BTC-USD", model_type: str = "random_forest",
            features: Optional[list] = None, period: str = "3y"):
        """运行完整 ML Pipeline"""
        self.fetch_data(symbol, period)

        # 从 Qlib 借鉴: 用不同特征集做对比实验
        all_features = features or [
            ["sma20", "sma50", "rsi14"],
            ["sma20", "sma50", "sma200", "rsi14", "volatility"],
            ["sma20", "sma50", "sma200", "rsi14", "volatility", "ret5", "ret20"],
        ]

        print(f"ML Pipeline: {symbol} | 模型: {model_type}")
        print(f"{'='*55}")
        print(f"{'特征集':<15} {'准确率':<8} {'精度':<8} {'召回率':<8} {'样本数':<8}")
        print(f"{'-'*55}")

        best_acc = 0
        best_features = None

        for i, feats in enumerate(all_features):
            model = MLSignalModel(model_type=model_type, features=feats)
            try:
                metrics = model.evaluate(self.df)
                acc = metrics["accuracy"] * 100
                print(f"{'特征组'+str(i+1):<15} {acc:<8.1f} {metrics['precision']:<8.3f} {metrics['recall']:<8.3f} {metrics['total_samples']:<8}")

                if acc > best_acc:
                    best_acc = acc
                    best_features = feats
                    self.model = model
                    self.results["best_metrics"] = metrics
            except Exception as e:
                print(f"{'特征组'+str(i+1):<15} ERROR: {str(e)[:30]}")

        print(f"{'-'*55}")
        print(f"最佳特征集: {best_features}")
        print(f"最佳准确率: {best_acc:.1f}%")

        # 回测
        self._backtest()
        return self.results

    def _backtest(self):
        """用 ML 信号做回测"""
        if not self.model or not self.model.fitted:
            return

        # 生成信号
        df = self.model.engineer_features(self.df.copy())
        data = df[self.model.features + [self.model.label_col]].dropna()
        X = data[self.model.features].values
        probas = self.model.model.predict_proba(X)[:, 1]  # 上涨概率

        # 信号: 上涨概率 > 60% 买入
        df_signal = df.loc[data.index].copy()
        df_signal["signal"] = (probas > 0.6).astype(int)

        # 回测
        df_signal["position"] = df_signal["signal"].shift(1)
        df_signal["returns"] = df_signal["Close"].pct_change()
        df_signal["strategy"] = df_signal["position"] * df_signal["returns"]
        df_signal["strategy"] -= 0.001 * df_signal["position"].diff().abs()

        initial = 10000
        self.equity_curve = initial * (1 + df_signal["strategy"]).cumprod()

        eq = self.equity_curve
        total_ret = (float(eq.iloc[-1]) / initial - 1) * 100
        rolling_max = eq.cummax()
        max_dd = float(((eq - rolling_max) / rolling_max * 100).min())
        strat_ret = df_signal["strategy"].dropna()
        sharpe = np.sqrt(252) * strat_ret.mean() / strat_ret.std() if strat_ret.std() > 0 else 0

        # 对比买入持有
        bh_ret = (float(df_signal["Close"].iloc[-1]) / float(df_signal["Close"].iloc[0]) - 1) * 100
        bh_rm = df_signal["Close"].cummax()
        bh_mdd = float(((df_signal["Close"] - bh_rm) / bh_rm * 100).min())

        self.results["backtest"] = {
            "ml_return": round(total_ret, 2),
            "ml_max_drawdown": round(max_dd, 2),
            "ml_sharpe": round(sharpe, 2),
            "buy_hold_return": round(bh_ret, 2),
            "buy_hold_drawdown": round(bh_mdd, 2),
        }

    def report(self):
        """输出报告"""
        metrics = self.results.get("best_metrics", {})
        bt = self.results.get("backtest", {})

        print(f"\n{'='*55}")
        print(f"  ML 量化报告")
        print(f"{'='*55}")
        print(f"  ML 准确率:     {metrics.get('accuracy', 0)*100:.1f}%")
        print(f"  ML 夏普比率:   {bt.get('ml_sharpe', 0):.2f}")
        print(f"  ML 收益率:     {bt.get('ml_return', 0):+.2f}%")
        print(f"  ML 最大回撤:   {bt.get('ml_max_drawdown', 0):.2f}%")
        print(f"  买入持有收益:  {bt.get('buy_hold_return', 0):+.2f}%")
        print(f"  买入持有回撤:  {bt.get('buy_hold_drawdown', 0):.2f}%")
        print(f"{'='*55}")

        return self.results
