"""
Technical indicators, risk metrics, and statistical calculations.
All calculations include validation for sufficient data points.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily and cumulative returns.
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        DataFrame with 'Daily Return' and 'Cumulative Return' columns
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    result['Daily Return'] = df['Close'].pct_change()
    result['Cumulative Return'] = (1 + result['Daily Return']).cumprod() - 1
    
    return result


def calculate_rolling_stats(df: pd.DataFrame, windows: List[int] = [20, 60]) -> pd.DataFrame:
    """
    Calculate rolling volatility and returns.
    
    Args:
        df: DataFrame with 'Daily Return' column
        windows: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with rolling statistics columns
    """
    if df is None or df.empty or 'Daily Return' not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    
    for window in windows:
        if len(df) >= window:
            result[f'Rolling Volatility {window}'] = df['Daily Return'].rolling(window=window).std() * np.sqrt(252)
            result[f'Rolling Return {window}'] = df['Daily Return'].rolling(window=window).mean() * 252
    
    return result


def calculate_summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for returns.
    
    Args:
        df: DataFrame with 'Daily Return' column
    
    Returns:
        Dictionary with summary statistics
    """
    if df is None or df.empty or 'Daily Return' not in df.columns:
        return {}
    
    returns = df['Daily Return'].dropna()
    
    if len(returns) == 0:
        return {}
    
    stats = {
        'Mean Return': returns.mean() * 252,  # Annualized
        'Median Return': returns.median() * 252,
        'Std Deviation': returns.std() * np.sqrt(252),  # Annualized
        'Skewness': returns.skew(),
        'Positive Days %': (returns > 0).sum() / len(returns) * 100
    }
    
    return stats


def calculate_volatility(df: pd.DataFrame, annualized: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with 'Daily Return' column
        annualized: Whether to annualize the volatility
    
    Returns:
        Volatility value
    """
    if df is None or df.empty or 'Daily Return' not in df.columns:
        return 0.0
    
    returns = df['Daily Return'].dropna()
    
    if len(returns) == 0:
        return 0.0
    
    vol = returns.std()
    
    if annualized:
        vol = vol * np.sqrt(252)
    
    return vol


def calculate_drawdown(df: pd.DataFrame) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series.
    
    Args:
        df: DataFrame with 'Close' column
    
    Returns:
        Tuple of (max_drawdown, drawdown_series)
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return 0.0, pd.Series()
    
    cumulative = (1 + df['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_drawdown = drawdown.min()
    
    return max_drawdown, drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0)
    
    Returns:
        Sharpe ratio
    """
    if returns is None or len(returns) == 0:
        return 0.0
    
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0.0
    
    excess_returns = returns_clean - (risk_free_rate / 252)
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return sharpe


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (downside deviation only).
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 0)
    
    Returns:
        Sortino ratio
    """
    if returns is None or len(returns) == 0:
        return 0.0
    
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0.0
    
    excess_returns = returns_clean - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    return sortino


def calculate_var_cvar(returns: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
    
    Args:
        returns: Series of daily returns
        confidence: Confidence level (default 0.95)
    
    Returns:
        Tuple of (VaR, CVaR)
    """
    if returns is None or len(returns) == 0:
        return 0.0, 0.0
    
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return 0.0, 0.0
    
    var = returns_clean.quantile(1 - confidence)
    cvar = returns_clean[returns_clean <= var].mean()
    
    return var, cvar


def calculate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate beta relative to benchmark.
    
    Args:
        stock_returns: Series of stock daily returns
        benchmark_returns: Series of benchmark daily returns
    
    Returns:
        Beta value
    """
    if (stock_returns is None or len(stock_returns) == 0 or
        benchmark_returns is None or len(benchmark_returns) == 0):
        return 0.0
    
    # Align indices
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    covariance = aligned['stock'].cov(aligned['benchmark'])
    benchmark_variance = aligned['benchmark'].var()
    
    if benchmark_variance == 0:
        return 0.0
    
    beta = covariance / benchmark_variance
    
    return beta


def calculate_correlation(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate correlation with benchmark.
    
    Args:
        stock_returns: Series of stock daily returns
        benchmark_returns: Series of benchmark daily returns
    
    Returns:
        Correlation coefficient
    """
    if (stock_returns is None or len(stock_returns) == 0 or
        benchmark_returns is None or len(benchmark_returns) == 0):
        return 0.0
    
    # Align indices
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 0.0
    
    correlation = aligned['stock'].corr(aligned['benchmark'])
    
    if pd.isna(correlation):
        return 0.0
    
    return correlation


def calculate_sma(df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages.
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of periods for SMA
    
    Returns:
        DataFrame with SMA columns added
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    
    for period in periods:
        if len(df) >= period:
            result[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    return result


def calculate_ema(df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages.
    
    Args:
        df: DataFrame with 'Close' column
        periods: List of periods for EMA
    
    Returns:
        DataFrame with EMA columns added
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    
    for period in periods:
        if len(df) >= period:
            result[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'Close' column
        period: RSI period (default 14)
    
    Returns:
        Series with RSI values
    """
    if df is None or df.empty or 'Close' not in df.columns or len(df) < period + 1:
        return pd.Series()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with 'Close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        DataFrame with MACD, Signal, and Histogram columns
    """
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    
    if len(df) < slow + signal:
        return pd.DataFrame()
    
    result = df.copy()
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    result['MACD'] = ema_fast - ema_slow
    result['MACD_Signal'] = result['MACD'].ewm(span=signal, adjust=False).mean()
    result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']
    
    return result


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'Close' column
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2.0)
    
    Returns:
        DataFrame with Upper, Middle, Lower band columns
    """
    if df is None or df.empty or 'Close' not in df.columns or len(df) < period:
        return pd.DataFrame()
    
    result = df.copy()
    
    result['BB_Middle'] = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    
    result['BB_Upper'] = result['BB_Middle'] + (std * std_dev)
    result['BB_Lower'] = result['BB_Middle'] - (std * std_dev)
    
    return result


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Volume Moving Average.
    
    Args:
        df: DataFrame with 'Volume' column
        period: Moving average period (default 20)
    
    Returns:
        Series with volume moving average
    """
    if df is None or df.empty or 'Volume' not in df.columns or len(df) < period:
        return pd.Series()
    
    return df['Volume'].rolling(window=period).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
    
    Returns:
        Series with OBV values
    """
    if df is None or df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        return pd.Series()
    
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return obv


def generate_signals(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with price and indicator data
        indicators: Dictionary with indicator dataframes/series
    
    Returns:
        DataFrame with signals: Signal Name, State, Last Trigger Date
    """
    signals = []
    
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame(columns=['Signal Name', 'State', 'Last Trigger Date'])
    
    current_price = df['Close'].iloc[-1]
    current_date = df.index[-1]
    
    # Golden/Death Cross (SMA50 vs SMA200)
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        sma50 = df['SMA_50'].iloc[-1]
        sma200 = df['SMA_200'].iloc[-1]
        
        if not pd.isna(sma50) and not pd.isna(sma200):
            # Find last cross
            cross_dates = []
            for i in range(1, len(df)):
                prev_sma50 = df['SMA_50'].iloc[i-1]
                prev_sma200 = df['SMA_200'].iloc[i-1]
                curr_sma50 = df['SMA_50'].iloc[i]
                curr_sma200 = df['SMA_200'].iloc[i]
                
                if (not pd.isna(prev_sma50) and not pd.isna(prev_sma200) and
                    not pd.isna(curr_sma50) and not pd.isna(curr_sma200)):
                    if prev_sma50 <= prev_sma200 and curr_sma50 > curr_sma200:
                        cross_dates.append(('Golden Cross', df.index[i]))
                    elif prev_sma50 >= prev_sma200 and curr_sma50 < curr_sma200:
                        cross_dates.append(('Death Cross', df.index[i]))
            
            if sma50 > sma200:
                last_cross = cross_dates[-1][1] if cross_dates and cross_dates[-1][0] == 'Golden Cross' else None
                signals.append({
                    'Signal Name': 'Golden Cross (SMA50 > SMA200)',
                    'State': 'Bullish',
                    'Last Trigger Date': last_cross if last_cross else 'Active'
                })
            else:
                last_cross = cross_dates[-1][1] if cross_dates and cross_dates[-1][0] == 'Death Cross' else None
                signals.append({
                    'Signal Name': 'Death Cross (SMA50 < SMA200)',
                    'State': 'Bearish',
                    'Last Trigger Date': last_cross if last_cross else 'Active'
                })
    
    # RSI Overbought/Oversold
    if 'RSI' in indicators and len(indicators['RSI']) > 0:
        rsi = indicators['RSI'].iloc[-1]
        if not pd.isna(rsi):
            if rsi >= 70:
                signals.append({
                    'Signal Name': 'RSI Overbought (≥70)',
                    'State': 'Bearish',
                    'Last Trigger Date': current_date
                })
            elif rsi <= 30:
                signals.append({
                    'Signal Name': 'RSI Oversold (≤30)',
                    'State': 'Bullish',
                    'Last Trigger Date': current_date
                })
            else:
                signals.append({
                    'Signal Name': 'RSI Neutral',
                    'State': 'Neutral',
                    'Last Trigger Date': current_date
                })
    
    # MACD Cross
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_Signal'].iloc[-1]
        
        if not pd.isna(macd) and not pd.isna(signal):
            # Find last cross
            macd_cross_dates = []
            for i in range(1, len(df)):
                prev_macd = df['MACD'].iloc[i-1]
                prev_signal = df['MACD_Signal'].iloc[i-1]
                curr_macd = df['MACD'].iloc[i]
                curr_signal = df['MACD_Signal'].iloc[i]
                
                if (not pd.isna(prev_macd) and not pd.isna(prev_signal) and
                    not pd.isna(curr_macd) and not pd.isna(curr_signal)):
                    if prev_macd <= prev_signal and curr_macd > curr_signal:
                        macd_cross_dates.append(('Bullish', df.index[i]))
                    elif prev_macd >= prev_signal and curr_macd < curr_signal:
                        macd_cross_dates.append(('Bearish', df.index[i]))
            
            if macd > signal:
                last_cross = macd_cross_dates[-1][1] if macd_cross_dates and macd_cross_dates[-1][0] == 'Bullish' else None
                signals.append({
                    'Signal Name': 'MACD Bullish Cross',
                    'State': 'Bullish',
                    'Last Trigger Date': last_cross if last_cross else 'Active'
                })
            else:
                last_cross = macd_cross_dates[-1][1] if macd_cross_dates and macd_cross_dates[-1][0] == 'Bearish' else None
                signals.append({
                    'Signal Name': 'MACD Bearish Cross',
                    'State': 'Bearish',
                    'Last Trigger Date': last_cross if last_cross else 'Active'
                })
    
    # Price vs SMA200
    if 'SMA_200' in df.columns:
        sma200 = df['SMA_200'].iloc[-1]
        if not pd.isna(sma200):
            if current_price > sma200:
                signals.append({
                    'Signal Name': 'Price Above SMA200',
                    'State': 'Bullish',
                    'Last Trigger Date': current_date
                })
            else:
                signals.append({
                    'Signal Name': 'Price Below SMA200',
                    'State': 'Bearish',
                    'Last Trigger Date': current_date
                })
    
    if not signals:
        return pd.DataFrame(columns=['Signal Name', 'State', 'Last Trigger Date'])
    
    return pd.DataFrame(signals)

