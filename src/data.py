"""
Data fetching and caching module for Stock Analysis Dashboard.
Handles all yfinance API calls with robust error handling and caching.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta


@st.cache_resource(ttl=3600)  # Cache Ticker objects (1 hour)
def get_ticker_object(ticker: str):
    """
    Get cached yfinance Ticker object.
    Uses st.cache_resource because Ticker objects are not serializable.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        yfinance Ticker object
    """
    return yf.Ticker(ticker)


@st.cache_data(ttl=300)  # 5 minutes cache for price data
def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch stock price data (OHLCV) from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        auto_adjust: Whether to adjust for splits and dividends
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)  # Use cached Ticker object
        df = stock.history(period=period, interval=interval, auto_adjust=auto_adjust)
        
        if df.empty:
            return None
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        return df
    except Exception as e:
        st.warning(f"Failed to fetch price data for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for static info
def fetch_stock_info(ticker: str) -> Optional[Dict]:
    """
    Fetch company information from yfinance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with company info or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)  # Use cached Ticker object
        info = stock.info
        
        if not info or len(info) == 0:
            return None
        
        return info
    except Exception as e:
        st.warning(f"Failed to fetch info for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for financials
def fetch_financials(ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch financial statements from yfinance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with 'income_statement', 'balance_sheet', 'cashflow' DataFrames
    """
    result = {
        'income_statement': None,
        'balance_sheet': None,
        'cashflow': None
    }
    
    try:
        stock = get_ticker_object(ticker)  # Use cached Ticker object
        
        # Try to fetch each financial statement
        try:
            income = stock.financials
            if not income.empty:
                result['income_statement'] = income
        except:
            pass
        
        try:
            balance = stock.balance_sheet
            if not balance.empty:
                result['balance_sheet'] = balance
        except:
            pass
        
        try:
            cashflow = stock.cashflow
            if not cashflow.empty:
                result['cashflow'] = cashflow
        except:
            pass
        
    except Exception as e:
        st.warning(f"Failed to fetch financials for {ticker}: {str(e)}")
    
    return result


@st.cache_data(ttl=3600)  # 1 hour cache for actions
def fetch_actions(ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch dividends and stock splits from yfinance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with 'dividends' and 'splits' DataFrames
    """
    result = {
        'dividends': None,
        'splits': None
    }
    
    try:
        stock = get_ticker_object(ticker)  # Use cached Ticker object
        actions = stock.actions
        
        if not actions.empty:
            if 'Dividends' in actions.columns:
                dividends = actions[actions['Dividends'] > 0]['Dividends']
                if not dividends.empty:
                    result['dividends'] = dividends.to_frame()
            
            if 'Stock Splits' in actions.columns:
                splits = actions[actions['Stock Splits'] > 0]['Stock Splits']
                if not splits.empty:
                    result['splits'] = splits.to_frame()
    
    except Exception as e:
        st.warning(f"Failed to fetch actions for {ticker}: {str(e)}")
    
    return result


@st.cache_data(ttl=300)  # 5 minutes cache for benchmark data
def fetch_benchmark_data(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d"
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch data for multiple tickers (for comparison).
    
    Args:
        tickers: List of ticker symbols
        period: Period to fetch
        interval: Data interval
    
    Returns:
        Dictionary mapping ticker to DataFrame
    """
    result = {}
    
    for ticker in tickers:
        try:
            stock = get_ticker_object(ticker)  # Use cached Ticker object
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                result[ticker] = None
            else:
                result[ticker] = df
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
            result[ticker] = None
    
    return result


def get_period_from_preset(preset: str) -> Tuple[str, Optional[datetime], Optional[datetime]]:
    """
    Convert preset period to yfinance period string and date range.
    
    Args:
        preset: One of '1M', '3M', '6M', '1Y', '5Y', 'MAX'
    
    Returns:
        Tuple of (period_string, start_date, end_date)
    """
    end_date = datetime.now()
    
    if preset == '1M':
        return '1mo', end_date - timedelta(days=30), end_date
    elif preset == '3M':
        return '3mo', end_date - timedelta(days=90), end_date
    elif preset == '6M':
        return '6mo', end_date - timedelta(days=180), end_date
    elif preset == '1Y':
        return '1y', end_date - timedelta(days=365), end_date
    elif preset == '5Y':
        return '5y', end_date - timedelta(days=1825), end_date
    elif preset == 'MAX':
        return 'max', None, end_date
    else:
        return '1y', end_date - timedelta(days=365), end_date


@st.cache_data(ttl=1800)  # 30 minutes cache for news
def fetch_news(ticker: str, num_articles: int = 10) -> Optional[List[Dict]]:
    """
    Fetch news articles for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        num_articles: Number of articles to fetch
    
    Returns:
        List of news dictionaries or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        news = stock.news
        
        if news and len(news) > 0:
            return news[:num_articles]
        return None
    except Exception as e:
        st.warning(f"Failed to fetch news for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for recommendations
def fetch_recommendations(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch analyst recommendations.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        DataFrame with recommendations or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            return recommendations
        return None
    except Exception as e:
        st.warning(f"Failed to fetch recommendations for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for institutional holders
def fetch_institutional_holders(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch institutional holders data.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        DataFrame with institutional holders or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        holders = stock.institutional_holders
        
        if holders is not None and not holders.empty:
            return holders
        return None
    except Exception as e:
        st.warning(f"Failed to fetch institutional holders for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for major holders
def fetch_major_holders(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch major holders data.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        DataFrame with major holders or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        holders = stock.major_holders
        
        if holders is not None and not holders.empty:
            return holders
        return None
    except Exception as e:
        st.warning(f"Failed to fetch major holders for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for calendar
def fetch_calendar(ticker: str) -> Optional[Dict]:
    """
    Fetch calendar events (earnings, dividends, etc.).
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with calendar events or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        calendar = stock.calendar
        
        if calendar is not None and not calendar.empty:
            return calendar.to_dict()
        return None
    except Exception as e:
        st.warning(f"Failed to fetch calendar for {ticker}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache for options
def fetch_options_dates(ticker: str) -> Optional[List]:
    """
    Fetch available options expiration dates.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        List of expiration dates or None if fetch fails
    """
    try:
        stock = get_ticker_object(ticker)
        options_dates = stock.options
        
        if options_dates and len(options_dates) > 0:
            return list(options_dates)
        return None
    except Exception as e:
        st.warning(f"Failed to fetch options dates for {ticker}: {str(e)}")
        return None

