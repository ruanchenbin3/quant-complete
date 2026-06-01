"""
工具函数
"""
import json
from pathlib import Path
from typing import Any


def load_config(path: str = None) -> dict:
    """加载配置文件"""
    if path is None:
        path = Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pct_change(current: float, previous: float) -> float:
    """计算涨跌幅百分比"""
    if previous == 0:
        return 0.0
    return (current - previous) / previous * 100


def format_currency(value: float, precision: int = 2) -> str:
    """格式化货币显示"""
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.{precision}f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.{precision}f}K"
    return f"${value:.{precision}f}"
