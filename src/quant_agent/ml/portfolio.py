"""
从 PyPortfolioOpt 借鉴: 投资组合优化

用法:
    from quant_agent.ml.portfolio import PortfolioOptimizer

    optimizer = PortfolioOptimizer()
    weights = optimizer.equal_weight(["BTC","ETH","SOL"])
    print(weights)  # {"BTC": 0.33, "ETH": 0.33, "SOL": 0.33}
"""
import numpy as np
import pandas as pd


class PortfolioOptimizer:
    """投资组合优化 — 不预测涨跌，只分配资金"""

    @staticmethod
    def equal_weight(tickers: list) -> dict:
        """等权重分配"""
        w = 1.0 / len(tickers)
        return {t: round(w, 4) for t in tickers}

    @staticmethod
    def risk_parity(returns: pd.DataFrame) -> dict:
        """
        从 PyPortfolioOpt 借鉴: 风险平价
        每个资产贡献相同风险
        """
        cov = returns.cov()
        inv_vol = 1.0 / np.sqrt(np.diag(cov))
        weights = inv_vol / inv_vol.sum()
        return {returns.columns[i]: round(w, 4) for i, w in enumerate(weights)}

    @staticmethod
    def min_variance(returns: pd.DataFrame) -> dict:
        """
        从 PyPortfolioOpt 借鉴: 最小方差组合
        追求波动最小的权重分配
        """
        cov = returns.cov().values
        n = len(cov)
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(n)
        weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
        return {returns.columns[i]: round(float(w), 4) for i, w in enumerate(weights)}

    @staticmethod
    def kelly_allocation(win_rates: dict, avg_wins: dict, avg_losses: dict, half=True) -> dict:
        """从风险管理借鉴: 凯利公式分配"""
        result = {}
        for asset in win_rates:
            p = win_rates[asset]
            b = avg_wins[asset] / abs(avg_losses[asset]) if avg_losses[asset] != 0 else 1
            kelly = (p * b - (1 - p)) / b if b > 0 else 0
            result[asset] = round(max(0, kelly * (0.5 if half else 1)), 4)
        # 归一化
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total, 4) for k, v in result.items()}
        return result

    @staticmethod
    def risk_budget(weights: dict, budgets: dict) -> dict:
        """
        风险预算: 你指定每个资产的风险比例
        risk_budget({"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.4}, ...)
        如果某个资产风险高，自动降低仓位
        """
        total_budget = sum(budgets.values())
        if total_budget == 0:
            return PortfolioOptimizer.equal_weight(list(weights.keys()))
        return {k: round(v / total_budget, 4) for k, v in budgets.items()}
