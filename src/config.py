"""
Configuration management module for Stock Analysis Dashboard.
Loads UI and application configuration from JSON files.
"""

import json
import os
from typing import Dict, Any


def load_config(config_path: str = "config/ui_config.json") -> Dict[str, Any]:
    """
    Load UI configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
    
    Returns:
        Dictionary with configuration settings
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return get_default_config()
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration if JSON file doesn't exist.
    
    Returns:
        Dictionary with default configuration settings
    """
    return {
        "theme": {
            "primary_color": "#1f77b4",
            "secondary_color": "#ff7f0e",
            "success_color": "#2ca02c",
            "danger_color": "#d62728",
            "warning_color": "#ffc107",
            "background_color": "#ffffff",
            "text_color": "#333333"
        },
        "charts": {
            "default_height": 600,
            "candlestick_colors": {
                "increasing": "#26a69a",
                "decreasing": "#ef5350"
            },
            "volume_colors": {
                "up": "#26a69a",
                "down": "#ef5350"
            }
        },
        "indicators": {
            "sma_periods": [20, 50, 200],
            "ema_periods": [12, 26],
            "rsi_period": 14,
            "macd": {
                "fast": 12,
                "slow": 26,
                "signal": 9
            },
            "bollinger": {
                "period": 20,
                "std_dev": 2.0
            }
        },
        "benchmarks": {
            "default": ["SPY"],
            "options": ["SPY", "QQQ", "^GSPC", "^DJI", "^IXIC", "DIA", "IWM"]
        },
        "kpi_cards": {
            "format": {
                "currency": "${:,.2f}",
                "percentage": "{:.2f}%",
                "number": "{:,.0f}"
            }
        }
    }

