"""
UI components and layout helpers for Stock Analysis Dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
from src import config


def render_header(ticker: str, current_price: Optional[float], last_update: Optional[datetime]) -> None:
    """
    Render header with ticker, current price, and timestamp.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current/last close price
        last_update: Timestamp of last data update
    """
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown(f"### {ticker.upper()}")
    
    with col2:
        if current_price is not None:
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            st.metric("Current Price", "N/A")
    
    with col3:
        if last_update:
            st.caption(f"Last Updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.caption("Last Updated: N/A")


def render_kpi_card(label: str, value, delta=None, delta_color="normal", format_str="${:.2f}") -> None:
    """
    Render a KPI metric card.
    
    Args:
        label: Label for the metric
        value: Value to display
        delta: Delta value (optional)
        delta_color: Color for delta ("normal", "inverse", "off")
        format_str: Format string for value
    """
    if isinstance(value, (int, float)):
        if format_str.startswith("$"):
            formatted_value = format_str.format(value)
        elif "%" in format_str:
            formatted_value = format_str.format(value)
        else:
            formatted_value = format_str.format(value)
    else:
        formatted_value = str(value) if value is not None else "N/A"
    
    st.metric(label, formatted_value, delta=delta, delta_color=delta_color)


def render_sidebar_filters() -> Dict:
    """
    Render sidebar with all filter controls.
    
    Returns:
        Dictionary with filter values
    """
    st.sidebar.header("Filters & Settings")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper().strip()
    
    # Period preset selector
    period_preset = st.sidebar.selectbox(
        "Time Period",
        options=['1M', '3M', '6M', '1Y', '5Y', 'MAX'],
        index=3,  # Default to 1Y
        help="Select the time period for historical data"
    )
    
    # Interval selector
    interval = st.sidebar.selectbox(
        "Data Interval",
        options=['1d', '1wk', '1mo'],
        index=0,  # Default to 1d
        help="Select the data interval"
    )
    
    # Auto-adjust checkbox
    auto_adjust = st.sidebar.checkbox(
        "Adjust for Splits/Dividends",
        value=True,
        help="Automatically adjust prices for stock splits and dividends"
    )
    
    st.sidebar.divider()
    
    # Indicator toggles
    st.sidebar.subheader("Indicators")
    show_sma = st.sidebar.checkbox("Show SMA", value=True)
    show_ema = st.sidebar.checkbox("Show EMA", value=True)
    show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
    show_volume = st.sidebar.checkbox("Show Volume", value=True)
    
    st.sidebar.divider()
    
    # Benchmark selection (load from config)
    st.sidebar.subheader("Benchmarks")
    ui_config = config.load_config()
    benchmark_options = ui_config.get("benchmarks", {}).get("options", ['SPY', 'QQQ', '^GSPC', '^DJI', '^IXIC'])
    default_benchmarks = ui_config.get("benchmarks", {}).get("default", ['SPY'])
    selected_benchmarks = st.sidebar.multiselect(
        "Select Benchmarks",
        options=benchmark_options,
        default=default_benchmarks,
        help="Select benchmarks for comparison"
    )
    
    st.sidebar.divider()
    
    # Risk-free rate input
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        help="Annual risk-free rate for Sharpe/Sortino calculations"
    ) / 100  # Convert to decimal
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    return {
        'ticker': ticker,
        'period_preset': period_preset,
        'interval': interval,
        'auto_adjust': auto_adjust,
        'show_sma': show_sma,
        'show_ema': show_ema,
        'show_bollinger': show_bollinger,
        'show_volume': show_volume,
        'selected_benchmarks': selected_benchmarks,
        'risk_free_rate': risk_free_rate
    }


def render_signals_panel(signals_df: pd.DataFrame) -> None:
    """
    Render signals panel with color coding.
    
    Args:
        signals_df: DataFrame with columns: Signal Name, State, Last Trigger Date
    """
    if signals_df is None or signals_df.empty:
        st.info("No signals available. Ensure sufficient data and indicators are calculated.")
        return
    
    st.subheader("Trading Signals")
    
    # Create styled dataframe
    styled_df = signals_df.copy()
    
    # Display with color coding
    for idx, row in styled_df.iterrows():
        state = row['State']
        
        if state == 'Bullish':
            color = '🟢'
        elif state == 'Bearish':
            color = '🔴'
        else:
            color = '🟡'
        
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.write(f"{color} **{row['Signal Name']}**")
        with col2:
            st.write(f"**{state}**")
        with col3:
            trigger_date = row['Last Trigger Date']
            if isinstance(trigger_date, str):
                st.write(trigger_date)
            else:
                st.write(trigger_date.strftime('%Y-%m-%d') if hasattr(trigger_date, 'strftime') else str(trigger_date))
        
        st.divider()


def render_fundamentals_cards(info: Dict) -> None:
    """
    Render fundamental metrics as KPI cards.
    
    Args:
        info: Dictionary with company info from yfinance
    """
    if not info:
        st.info("Fundamentals data not available for this ticker.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = info.get('marketCap')
        if market_cap:
            render_kpi_card("Market Cap", market_cap, format_str="${:,.0f}")
        else:
            render_kpi_card("Market Cap", "N/A")
    
    with col2:
        sector = info.get('sector', 'N/A')
        st.metric("Sector", sector)
    
    with col3:
        industry = info.get('industry', 'N/A')
        st.metric("Industry", industry)
    
    with col4:
        pe_ratio = info.get('trailingPE')
        if pe_ratio:
            render_kpi_card("Trailing P/E", pe_ratio, format_str="{:.2f}")
        else:
            render_kpi_card("Trailing P/E", "N/A")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        forward_pe = info.get('forwardPE')
        if forward_pe:
            render_kpi_card("Forward P/E", forward_pe, format_str="{:.2f}")
        else:
            render_kpi_card("Forward P/E", "N/A")
    
    with col6:
        dividend_yield = info.get('dividendYield')
        if dividend_yield:
            render_kpi_card("Dividend Yield", dividend_yield * 100, format_str="{:.2f}%")
        else:
            render_kpi_card("Dividend Yield", "N/A")
    
    with col7:
        beta = info.get('beta')
        if beta:
            render_kpi_card("Beta", beta, format_str="{:.2f}")
        else:
            render_kpi_card("Beta", "N/A")
    
    with col8:
        eps = info.get('trailingEps')
        if eps:
            render_kpi_card("EPS (TTM)", eps, format_str="${:.2f}")
        else:
            render_kpi_card("EPS (TTM)", "N/A")


def render_enhanced_kpi_card(label: str, value, delta=None, 
                             card_type="default", format_str="${:.2f}") -> None:
    """
    Enhanced KPI card with better styling.
    
    Args:
        label: Label for the metric
        value: Value to display
        delta: Delta value (optional)
        card_type: Type of card ("default", "success", "danger")
        format_str: Format string for value
    """
    if isinstance(value, (int, float)):
        if format_str.startswith("$"):
            formatted_value = format_str.format(value)
        elif "%" in format_str:
            formatted_value = format_str.format(value)
        else:
            formatted_value = format_str.format(value)
    else:
        formatted_value = str(value) if value is not None else "N/A"
    
    delta_html = ""
    if delta:
        delta_value = float(str(delta).replace("%", "").replace("$", "").replace(",", ""))
        delta_color = "success" if delta_value >= 0 else "danger"
        delta_html = f'<span style="color: {"green" if delta_value >= 0 else "red"}; font-weight: bold;">{delta}</span>'
    
    card_class = f"kpi-card {card_type}"
    
    html = f"""
    <div class="{card_class}" style="padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #333;">{formatted_value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_comparison(current_value: float, previous_value: float, label: str) -> None:
    """
    Render metric with comparison to previous period.
    
    Args:
        current_value: Current period value
        previous_value: Previous period value
        label: Label for the metric
    """
    change = current_value - previous_value
    change_pct = (change / previous_value * 100) if previous_value != 0 else 0
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(label, f"${current_value:,.2f}", 
                 delta=f"{change_pct:.2f}%")
    with col2:
        st.caption(f"Previous: ${previous_value:,.2f}")


def check_price_alerts(ticker: str, current_price: float, 
                      alerts_config: Dict) -> List[str]:
    """
    Check if price triggers any alerts.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        alerts_config: Dictionary with alert configurations
    
    Returns:
        List of alert messages
    """
    alerts = []
    
    if 'target_price' in alerts_config:
        target = alerts_config['target_price']
        if current_price >= target:
            alerts.append(f"🎯 Target price reached: ${target:.2f}")
    
    if 'stop_loss' in alerts_config:
        stop_loss = alerts_config['stop_loss']
        if current_price <= stop_loss:
            alerts.append(f"⚠️ Stop loss triggered: ${stop_loss:.2f}")
    
    return alerts


def render_alerts_panel(alerts: List[str]) -> None:
    """
    Render alerts panel in sidebar.
    
    Args:
        alerts: List of alert messages
    """
    if alerts:
        st.sidebar.subheader("🔔 Alerts")
        for alert in alerts:
            st.sidebar.warning(alert)

