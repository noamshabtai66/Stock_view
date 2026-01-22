"""
Stock Analysis Dashboard - Main Application
A comprehensive stock analysis tool using Streamlit and yfinance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import json
import time
from io import BytesIO

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import data
from src import features
from src import charts
from src import ui
from src import config
from src import utils

# Setup logging
utils.setup_logging()

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
ui_config = config.load_config()
theme = ui_config.get("theme", {})

# Custom CSS for better styling
def load_custom_css():
    """Load advanced CSS styling from JSON config."""
    primary_color = theme.get('primary_color', '#1f77b4')
    secondary_color = theme.get('secondary_color', '#ff7f0e')
    success_color = theme.get('success_color', '#2ca02c')
    danger_color = theme.get('danger_color', '#d62728')
    
    css = f"""
    <style>
    /* Main Header */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, {primary_color}, {secondary_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin-bottom: 1rem;
    }}
    
    .kpi-card.success {{
        background: linear-gradient(135deg, {success_color} 0%, #38ef7d 100%);
    }}
    
    .kpi-card.danger {{
        background: linear-gradient(135deg, {danger_color} 0%, #ff6a00 100%);
    }}
    
    /* Sidebar */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }}
    
    /* Data Tables */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Charts Container */
    .chart-container {{
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    /* Loading Spinner */
    .stSpinner > div {{
        border-color: {primary_color};
    }}
    
    /* Buttons */
    .stButton > button {{
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Disclaimer */
    .disclaimer {{
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin-top: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    </style>
    """
    return css

st.markdown(load_custom_css(), unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">📈 Stock Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar filters
filters = ui.render_sidebar_filters()
ticker = filters['ticker']
period_preset = filters['period_preset']
interval = filters['interval']
auto_adjust = filters['auto_adjust']
show_sma = filters['show_sma']
show_ema = filters['show_ema']
show_bollinger = filters['show_bollinger']
show_volume = filters['show_volume']
selected_benchmarks = filters['selected_benchmarks']
risk_free_rate = filters['risk_free_rate']

# Convert period preset to yfinance period
period, _, _ = data.get_period_from_preset(period_preset)

# Fetch main stock data
if ticker:
    # Enhanced data fetching with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Fetching data...")
    progress_bar.progress(20)
    
    stock_df = data.fetch_stock_data(ticker, period=period, interval=interval, auto_adjust=auto_adjust)
    progress_bar.progress(50)
    
    stock_info = data.fetch_stock_info(ticker)
    progress_bar.progress(70)
    
    financials = data.fetch_financials(ticker)
    progress_bar.progress(85)
    
    actions = data.fetch_actions(ticker)
    progress_bar.progress(100)
    
    # Clear progress
    time.sleep(0.3)
    progress_bar.empty()
    status_text.empty()
    
    if stock_df is None or stock_df.empty:
        st.error(f"❌ No data available for ticker {ticker}. Please check the ticker symbol and try again.")
        st.stop()
    
    # Calculate returns
    stock_df = features.calculate_returns(stock_df)
    
    # Get current price and last update
    current_price = stock_df['Close'].iloc[-1] if not stock_df.empty else None
    last_update = stock_df.index[-1] if not stock_df.empty else None
    
    # Render header
    ui.render_header(ticker, current_price, last_update)
    
    # Initialize session state for watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Overview",
        "📈 Technical Analysis",
        "⚠️ Risk & Performance",
        "💼 Fundamentals",
        "📅 Events & Dividends",
        "🔀 Compare & Correlation",
        "💾 Data Explorer",
        "👁️ Watchlist",
        "📰 News & Insights"
    ])
    
    # ========== TAB 1: OVERVIEW ==========
    with tab1:
        st.header("Overview")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if current_price:
                prev_close = stock_df['Close'].iloc[-2] if len(stock_df) > 1 else current_price
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close != 0 else 0
                ui.render_kpi_card("Current Price", current_price, delta=f"{change_pct:.2f}%")
        
        with col2:
            if 'Daily Return' in stock_df.columns:
                daily_return = stock_df['Daily Return'].iloc[-1]
                if not pd.isna(daily_return):
                    ui.render_kpi_card("Daily Return", daily_return * 100, format_str="{:.2f}%")
        
        with col3:
            if 'Volume' in stock_df.columns:
                volume = stock_df['Volume'].iloc[-1]
                ui.render_kpi_card("Volume", volume, format_str="{:,.0f}")
        
        with col4:
            if stock_info and 'marketCap' in stock_info:
                market_cap = stock_info['marketCap']
                ui.render_kpi_card("Market Cap", market_cap, format_str="${:,.0f}")
            else:
                ui.render_kpi_card("Market Cap", "N/A")
        
        st.divider()
        
        # Price chart
        st.subheader("Price Chart")
        if show_volume:
            fig = charts.plot_candlestick(stock_df, volume=True)
        else:
            fig = charts.plot_candlestick(stock_df, volume=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume Profile (if enabled)
        if show_volume and len(stock_df) > 20:
            st.subheader("Volume Profile")
            fig_vol_profile = charts.plot_volume_profile(stock_df)
            if fig_vol_profile:
                st.plotly_chart(fig_vol_profile, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        stats = features.calculate_summary_stats(stock_df)
        if stats:
            stats_df = pd.DataFrame([stats]).T
            stats_df.columns = ['Value']
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("Insufficient data for summary statistics.")
        
        # Recent price action
        st.subheader("Recent Price Action")
        recent_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Daily Return' in stock_df.columns:
            recent_cols.append('Daily Return')
        recent_df = stock_df[recent_cols].tail(10)
        st.dataframe(recent_df, use_container_width=True)
    
    # ========== TAB 2: TECHNICAL ANALYSIS ==========
    with tab2:
        st.header("Technical Analysis")
        
        # Calculate indicators (load periods from config)
        indicators_dict = {}
        indicator_config = ui_config.get("indicators", {})
        
        if show_sma:
            sma_periods = indicator_config.get("sma_periods", [20, 50, 200])
            sma_df = features.calculate_sma(stock_df, periods=sma_periods)
            if not sma_df.empty:
                indicators_dict['SMA'] = sma_df
        
        if show_ema:
            ema_periods = indicator_config.get("ema_periods", [12, 26])
            ema_df = features.calculate_ema(stock_df, periods=ema_periods)
            if not ema_df.empty:
                indicators_dict['EMA'] = ema_df
        
        if show_bollinger:
            bb_config = indicator_config.get("bollinger", {})
            bb_period = bb_config.get("period", 20)
            bb_std = bb_config.get("std_dev", 2.0)
            bb_df = features.calculate_bollinger_bands(stock_df, period=bb_period, std_dev=bb_std)
            if not bb_df.empty:
                indicators_dict['Bollinger'] = bb_df
        
        # RSI
        rsi_period = indicator_config.get("rsi_period", 14)
        rsi_series = features.calculate_rsi(stock_df, period=rsi_period)
        if not rsi_series.empty:
            indicators_dict['RSI'] = rsi_series
        
        # MACD
        macd_config = indicator_config.get("macd", {})
        macd_fast = macd_config.get("fast", 12)
        macd_slow = macd_config.get("slow", 26)
        macd_signal = macd_config.get("signal", 9)
        macd_df = features.calculate_macd(stock_df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if not macd_df.empty:
            indicators_dict['MACD'] = macd_df
        
        # Candlestick chart with overlays
        st.subheader("Price Chart with Indicators")
        sma_data = indicators_dict.get('SMA')
        ema_data = indicators_dict.get('EMA')
        bb_data = indicators_dict.get('Bollinger')
        
        fig_candle = charts.plot_candlestick(
            stock_df,
            sma_data=sma_data,
            ema_data=ema_data,
            bollinger_data=bb_data,
            volume=show_volume
        )
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # RSI Chart
        if 'RSI' in indicators_dict:
            st.subheader("RSI (Relative Strength Index)")
            fig_rsi = charts.plot_rsi(stock_df, indicators_dict['RSI'])
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        # MACD Chart
        if 'MACD' in indicators_dict:
            st.subheader("MACD")
            fig_macd = charts.plot_macd(stock_df, indicators_dict['MACD'])
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Signals Panel
        signals_df = features.generate_signals(stock_df, indicators_dict)
        if not signals_df.empty:
            ui.render_signals_panel(signals_df)
        
        # Indicator values table
        st.subheader("Current Indicator Values")
        indicator_values = {}
        
        if 'SMA' in indicators_dict:
            for col in indicators_dict['SMA'].columns:
                if col.startswith('SMA_'):
                    val = indicators_dict['SMA'][col].iloc[-1]
                    if not pd.isna(val):
                        indicator_values[col] = val
        
        if 'EMA' in indicators_dict:
            for col in indicators_dict['EMA'].columns:
                if col.startswith('EMA_'):
                    val = indicators_dict['EMA'][col].iloc[-1]
                    if not pd.isna(val):
                        indicator_values[col] = val
        
        if 'RSI' in indicators_dict:
            rsi_val = indicators_dict['RSI'].iloc[-1]
            if not pd.isna(rsi_val):
                indicator_values['RSI'] = rsi_val
        
        if 'MACD' in indicators_dict:
            if 'MACD' in indicators_dict['MACD'].columns:
                macd_val = indicators_dict['MACD']['MACD'].iloc[-1]
                if not pd.isna(macd_val):
                    indicator_values['MACD'] = macd_val
            if 'MACD_Signal' in indicators_dict['MACD'].columns:
                signal_val = indicators_dict['MACD']['MACD_Signal'].iloc[-1]
                if not pd.isna(signal_val):
                    indicator_values['MACD_Signal'] = signal_val
        
        if indicator_values:
            indicator_df = pd.DataFrame([indicator_values]).T
            indicator_df.columns = ['Value']
            st.dataframe(indicator_df, use_container_width=True)
    
    # ========== TAB 3: RISK & PERFORMANCE ==========
    with tab3:
        st.header("Risk & Performance Metrics")
        
        if 'Daily Return' not in stock_df.columns:
            st.warning("Daily returns not available. Cannot calculate risk metrics.")
        else:
            returns = stock_df['Daily Return'].dropna()
            
            if len(returns) == 0:
                st.warning("Insufficient return data for risk calculations.")
            else:
                # Risk Metrics KPI Cards
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    volatility = features.calculate_volatility(stock_df, annualized=True)
                    ui.render_kpi_card("Annualized Volatility", volatility * 100, format_str="{:.2f}%")
                
                with col2:
                    sharpe = features.calculate_sharpe_ratio(returns, risk_free_rate)
                    ui.render_kpi_card("Sharpe Ratio", sharpe, format_str="{:.2f}")
                
                with col3:
                    sortino = features.calculate_sortino_ratio(returns, risk_free_rate)
                    ui.render_kpi_card("Sortino Ratio", sortino, format_str="{:.2f}")
                
                with col4:
                    var, _ = features.calculate_var_cvar(returns, confidence=0.95)
                    ui.render_kpi_card("VaR (95%)", var * 100, format_str="{:.2f}%")
                
                with col5:
                    _, cvar = features.calculate_var_cvar(returns, confidence=0.95)
                    ui.render_kpi_card("CVaR (95%)", cvar * 100, format_str="{:.2f}%")
                
                st.divider()
                
                # Drawdown chart
                st.subheader("Drawdown Analysis")
                max_dd, dd_series = features.calculate_drawdown(stock_df)
                
                col_dd1, col_dd2 = st.columns([3, 1])
                with col_dd1:
                    fig_dd = charts.plot_drawdown(dd_series)
                    st.plotly_chart(fig_dd, use_container_width=True)
                with col_dd2:
                    st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
                
                # Rolling statistics
                st.subheader("Rolling Statistics")
                rolling_df = features.calculate_rolling_stats(stock_df, windows=[20, 60])
                if not rolling_df.empty:
                    rolling_cols = [col for col in rolling_df.columns if 'Rolling' in col]
                    if rolling_cols:
                        fig_rolling = go.Figure()
                        for col in rolling_cols:
                            fig_rolling.add_trace(go.Scatter(
                                x=rolling_df.index,
                                y=rolling_df[col],
                                name=col,
                                line=dict(width=2)
                            ))
                        fig_rolling.update_layout(
                            title='Rolling Volatility & Returns',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Performance statistics table
                st.subheader("Performance Statistics")
                perf_stats = features.calculate_summary_stats(stock_df)
                if perf_stats:
                    perf_df = pd.DataFrame([perf_stats]).T
                    perf_df.columns = ['Value']
                    st.dataframe(perf_df, use_container_width=True)
    
    # ========== TAB 4: FUNDAMENTALS ==========
    with tab4:
        st.header("Fundamentals")
        
        if stock_info:
            ui.render_fundamentals_cards(stock_info)
            
            st.divider()
            
            # Financial Statements
            st.subheader("Financial Statements")
            
            if financials['income_statement'] is not None:
                st.write("**Income Statement**")
                st.dataframe(financials['income_statement'], use_container_width=True)
            
            if financials['balance_sheet'] is not None:
                st.write("**Balance Sheet**")
                st.dataframe(financials['balance_sheet'], use_container_width=True)
            
            if financials['cashflow'] is not None:
                st.write("**Cash Flow Statement**")
                st.dataframe(financials['cashflow'], use_container_width=True)
            
            # Calculate growth metrics if possible
            if financials['income_statement'] is not None and not financials['income_statement'].empty:
                st.subheader("Growth Metrics")
                income = financials['income_statement']
                
                # Try to find revenue
                revenue_cols = [col for col in income.index if 'revenue' in col.lower() or 'total revenue' in col.lower()]
                if not revenue_cols:
                    revenue_cols = [col for col in income.index if 'sales' in col.lower()]
                
                if revenue_cols and len(income.columns) >= 2:
                    revenue_col = revenue_cols[0]
                    try:
                        recent_revenue = income.loc[revenue_col, income.columns[0]]
                        prev_revenue = income.loc[revenue_col, income.columns[1]]
                        if pd.notna(recent_revenue) and pd.notna(prev_revenue) and prev_revenue != 0:
                            revenue_growth = ((recent_revenue - prev_revenue) / prev_revenue) * 100
                            st.metric("Revenue Growth (YoY)", f"{revenue_growth:.2f}%")
                    except:
                        pass
                
                # Try to find net income
                net_income_cols = [col for col in income.index if 'net income' in col.lower()]
                if net_income_cols and len(income.columns) >= 1:
                    net_income_col = net_income_cols[0]
                    try:
                        net_income = income.loc[net_income_col, income.columns[0]]
                        revenue_val = income.loc[revenue_cols[0], income.columns[0]] if revenue_cols else None
                        if pd.notna(net_income) and pd.notna(revenue_val) and revenue_val != 0:
                            profit_margin = (net_income / revenue_val) * 100
                            st.metric("Profit Margin", f"{profit_margin:.2f}%")
                    except:
                        pass
        else:
            st.info("Fundamentals data not provided by yfinance for this ticker.")
    
    # ========== TAB 5: EVENTS & DIVIDENDS ==========
    with tab5:
        st.header("Events & Dividends")
        
        # Dividends
        st.subheader("Dividends History")
        if actions['dividends'] is not None and not actions['dividends'].empty:
            dividends_df = actions['dividends']
            fig_div = charts.plot_dividends_history(dividends_df)
            st.plotly_chart(fig_div, use_container_width=True)
            
            # Filter dividends by period
            if period_preset != 'MAX':
                period_start = stock_df.index[0]
                period_dividends = dividends_df[dividends_df.index >= period_start]
                total_dividends = period_dividends['Dividends'].sum() if 'Dividends' in period_dividends.columns else period_dividends.iloc[:, 0].sum()
            else:
                total_dividends = dividends_df['Dividends'].sum() if 'Dividends' in dividends_df.columns else dividends_df.iloc[:, 0].sum()
            
            st.metric("Total Dividends (Period)", f"${total_dividends:.4f}")
            
            st.write("**Dividends Table**")
            st.dataframe(dividends_df, use_container_width=True)
        else:
            st.info("No dividend data available for this ticker.")
        
        st.divider()
        
        # Stock Splits
        st.subheader("Stock Splits")
        if actions['splits'] is not None and not actions['splits'].empty:
            splits_df = actions['splits']
            st.dataframe(splits_df, use_container_width=True)
        else:
            st.info("No stock split data available for this ticker.")
        
        st.divider()
        
        # Earnings (if available in info)
        st.subheader("Earnings")
        if stock_info and 'earningsDates' in stock_info:
            earnings_dates = stock_info['earningsDates']
            if earnings_dates:
                earnings_df = pd.DataFrame({'Earnings Date': earnings_dates})
                st.dataframe(earnings_df, use_container_width=True)
            else:
                st.info("Earnings dates not available.")
        else:
            st.info("Earnings dates not provided by yfinance for this ticker.")
    
    # ========== TAB 6: COMPARE & CORRELATION ==========
    with tab6:
        st.header("Compare & Correlation")
        
        if not selected_benchmarks:
            st.info("Please select at least one benchmark from the sidebar to enable comparison.")
        else:
            # Fetch benchmark data
            with st.spinner("Fetching benchmark data..."):
                benchmark_data = data.fetch_benchmark_data(selected_benchmarks, period=period, interval=interval)
            
            # Calculate cumulative returns for comparison
            comparison_data = {}
            
            # Add main ticker
            if 'Cumulative Return' in stock_df.columns:
                comparison_data[ticker] = stock_df['Cumulative Return']
            
            # Add benchmarks
            for bench_ticker, bench_df in benchmark_data.items():
                if bench_df is not None and not bench_df.empty:
                    bench_returns = features.calculate_returns(bench_df)
                    if 'Cumulative Return' in bench_returns.columns:
                        comparison_data[bench_ticker] = bench_returns['Cumulative Return']
            
            if len(comparison_data) > 1:
                # Align all series to common index
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.dropna()
                
                if not comparison_df.empty:
                    # Cumulative returns chart
                    st.subheader("Cumulative Returns Comparison")
                    fig_compare = charts.plot_returns_comparison(comparison_df)
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Correlation heatmap
                    st.subheader("Correlation Matrix")
                    correlation_matrix = comparison_df.corr()
                    fig_corr = charts.plot_correlation_heatmap(correlation_matrix)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Beta calculation
                    st.subheader("Beta Calculation")
                    if ticker in comparison_df.columns:
                        stock_returns_series = stock_df['Daily Return'].dropna() if 'Daily Return' in stock_df.columns else pd.Series()
                        
                        beta_results = {}
                        for bench_ticker in selected_benchmarks:
                            if bench_ticker in comparison_df.columns:
                                bench_df = benchmark_data[bench_ticker]
                                if bench_df is not None and not bench_df.empty:
                                    bench_returns = features.calculate_returns(bench_df)
                                    if 'Daily Return' in bench_returns.columns:
                                        bench_returns_series = bench_returns['Daily Return'].dropna()
                                        beta = features.calculate_beta(stock_returns_series, bench_returns_series)
                                        correlation = features.calculate_correlation(stock_returns_series, bench_returns_series)
                                        beta_results[bench_ticker] = {
                                            'Beta': beta,
                                            'Correlation': correlation
                                        }
                        
                        if beta_results:
                            beta_df = pd.DataFrame(beta_results).T
                            st.dataframe(beta_df, use_container_width=True)
                    
                    # Performance comparison table
                    st.subheader("Performance Comparison")
                    perf_comparison = {}
                    for col in comparison_df.columns:
                        final_return = comparison_df[col].iloc[-1]
                        perf_comparison[col] = {
                            'Total Return': final_return * 100,
                            'Final Value': final_return
                        }
                    
                    perf_comp_df = pd.DataFrame(perf_comparison).T
                    st.dataframe(perf_comp_df, use_container_width=True)
                else:
                    st.warning("Insufficient overlapping data for comparison.")
            else:
                st.warning("Need at least 2 tickers (including main ticker) for comparison.")
    
    # ========== TAB 7: DATA EXPLORER ==========
    with tab7:
        st.header("Data Explorer")
        
        # Combine all data
        explorer_df = stock_df.copy()
        
        # Add indicators if available
        if show_sma and 'SMA' in indicators_dict:
            for col in indicators_dict['SMA'].columns:
                explorer_df[col] = indicators_dict['SMA'][col]
        
        if show_ema and 'EMA' in indicators_dict:
            for col in indicators_dict['EMA'].columns:
                explorer_df[col] = indicators_dict['EMA'][col]
        
        if 'RSI' in indicators_dict:
            explorer_df['RSI'] = indicators_dict['RSI']
        
        if 'MACD' in indicators_dict:
            for col in indicators_dict['MACD'].columns:
                explorer_df[col] = indicators_dict['MACD'][col]
        
        # Display interactive dataframe
        st.subheader("Price Data with Indicators")
        st.dataframe(explorer_df, use_container_width=True, height=400)
        
        # Export options
        st.subheader("Export Data")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # Export price + indicators
            csv_price = explorer_df.to_csv()
            st.download_button(
                label="📥 Download Price Data + Indicators (CSV)",
                data=csv_price,
                file_name=f"{ticker}_price_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            # Export signals
            if not signals_df.empty:
                csv_signals = signals_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Signals Table (CSV)",
                    data=csv_signals,
                    file_name=f"{ticker}_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No signals to export")
        
        with col_exp3:
            # Export risk metrics
            if 'Daily Return' in stock_df.columns:
                returns = stock_df['Daily Return'].dropna()
                if len(returns) > 0:
                    risk_metrics = {
                        'Metric': ['Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
                        'Value': [
                            features.calculate_volatility(stock_df, annualized=True) * 100,
                            features.calculate_sharpe_ratio(returns, risk_free_rate),
                            features.calculate_sortino_ratio(returns, risk_free_rate),
                            features.calculate_var_cvar(returns, confidence=0.95)[0] * 100,
                            features.calculate_var_cvar(returns, confidence=0.95)[1] * 100,
                            features.calculate_drawdown(stock_df)[0] * 100
                        ]
                    }
                    risk_df = pd.DataFrame(risk_metrics)
                    csv_risk = risk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Risk Metrics (CSV)",
                        data=csv_risk,
                        file_name=f"{ticker}_risk_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        st.divider()
        
        # Export to JSON
        st.subheader("Export to JSON")
        if st.button("📥 Export to JSON"):
            export_data = {
                'ticker': ticker,
                'date': datetime.now().isoformat(),
                'price_data': explorer_df.to_dict('records'),
                'signals': signals_df.to_dict('records') if not signals_df.empty else [],
                'risk_metrics': {}
            }
            
            if 'Daily Return' in stock_df.columns:
                returns = stock_df['Daily Return'].dropna()
                if len(returns) > 0:
                    export_data['risk_metrics'] = {
                        'Annualized Volatility': features.calculate_volatility(stock_df, annualized=True) * 100,
                        'Sharpe Ratio': features.calculate_sharpe_ratio(returns, risk_free_rate),
                        'Sortino Ratio': features.calculate_sortino_ratio(returns, risk_free_rate),
                        'VaR (95%)': features.calculate_var_cvar(returns, confidence=0.95)[0] * 100,
                        'CVaR (95%)': features.calculate_var_cvar(returns, confidence=0.95)[1] * 100,
                        'Max Drawdown': features.calculate_drawdown(stock_df)[0] * 100
                    }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{ticker}_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        # Export to Excel
        st.subheader("Export to Excel")
        try:
            import openpyxl
            if st.button("📊 Export to Excel"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    explorer_df.to_excel(writer, sheet_name='Price Data', index=True)
                    if not signals_df.empty:
                        signals_df.to_excel(writer, sheet_name='Signals', index=False)
                    if 'Daily Return' in stock_df.columns:
                        returns = stock_df['Daily Return'].dropna()
                        if len(returns) > 0:
                            risk_metrics = {
                                'Metric': ['Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
                                'Value': [
                                    features.calculate_volatility(stock_df, annualized=True) * 100,
                                    features.calculate_sharpe_ratio(returns, risk_free_rate),
                                    features.calculate_sortino_ratio(returns, risk_free_rate),
                                    features.calculate_var_cvar(returns, confidence=0.95)[0] * 100,
                                    features.calculate_var_cvar(returns, confidence=0.95)[1] * 100,
                                    features.calculate_drawdown(stock_df)[0] * 100
                                ]
                            }
                            risk_df = pd.DataFrame(risk_metrics)
                            risk_df.to_excel(writer, sheet_name='Risk Metrics', index=False)
                
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"{ticker}_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except ImportError:
            st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
    
    # ========== TAB 8: WATCHLIST ==========
    with tab8:
        st.header("Watchlist & Portfolio")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_ticker = st.text_input("Add Ticker to Watchlist", 
                                      placeholder="e.g., AAPL, TSLA, MSFT",
                                      key="watchlist_input")
        
        with col2:
            if st.button("➕ Add", key="add_to_watchlist"):
                if new_ticker:
                    ticker_upper = new_ticker.upper().strip()
                    if ticker_upper not in st.session_state.watchlist:
                        st.session_state.watchlist.append(ticker_upper)
                        st.success(f"✅ Added {ticker_upper} to watchlist")
                        st.rerun()
                    else:
                        st.warning(f"{ticker_upper} is already in watchlist")
        
        # Display watchlist
        if st.session_state.watchlist:
            st.subheader("Your Watchlist")
            
            # Remove ticker option
            col_remove1, col_remove2 = st.columns([3, 1])
            with col_remove1:
                ticker_to_remove = st.selectbox(
                    "Remove Ticker",
                    options=[""] + st.session_state.watchlist,
                    key="remove_ticker_select"
                )
            with col_remove2:
                if st.button("➖ Remove", key="remove_from_watchlist"):
                    if ticker_to_remove and ticker_to_remove in st.session_state.watchlist:
                        st.session_state.watchlist.remove(ticker_to_remove)
                        st.success(f"✅ Removed {ticker_to_remove}")
                        st.rerun()
            
            st.divider()
            
            # Quick comparison table
            watchlist_data = []
            with st.spinner("Loading watchlist data..."):
                for watch_ticker in st.session_state.watchlist:
                    try:
                        ticker_info = data.fetch_stock_info(watch_ticker)
                        ticker_df = data.fetch_stock_data(watch_ticker, period="1mo", interval="1d")
                        
                        if ticker_df is not None and not ticker_df.empty:
                            current_price = ticker_df['Close'].iloc[-1]
                            prev_price = ticker_df['Close'].iloc[-2] if len(ticker_df) > 1 else current_price
                            change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                            
                            market_cap = ticker_info.get('marketCap', 'N/A') if ticker_info else 'N/A'
                            if isinstance(market_cap, (int, float)):
                                market_cap_str = f"${market_cap:,.0f}"
                            else:
                                market_cap_str = str(market_cap)
                            
                            watchlist_data.append({
                                'Ticker': watch_ticker,
                                'Price': f"${current_price:.2f}",
                                'Change %': f"{change_pct:.2f}%",
                                'Market Cap': market_cap_str
                            })
                    except Exception as e:
                        watchlist_data.append({
                            'Ticker': watch_ticker,
                            'Price': 'Error',
                            'Change %': 'N/A',
                            'Market Cap': 'N/A'
                        })
            
            if watchlist_data:
                watchlist_df = pd.DataFrame(watchlist_data)
                st.dataframe(watchlist_df, use_container_width=True)
                
                # Performance Radar Chart
                if len(watchlist_data) >= 3:
                    st.subheader("Performance Comparison (Radar Chart)")
                    try:
                        # Calculate metrics for radar chart
                        radar_metrics = {}
                        for watch_ticker in st.session_state.watchlist[:5]:  # Limit to 5 for clarity
                            try:
                                ticker_df = data.fetch_stock_data(watch_ticker, period="3mo", interval="1d")
                                if ticker_df is not None and not ticker_df.empty:
                                    ticker_returns = features.calculate_returns(ticker_df)
                                    if 'Daily Return' in ticker_returns.columns:
                                        returns = ticker_returns['Daily Return'].dropna()
                                        if len(returns) > 0:
                                            sharpe = features.calculate_sharpe_ratio(returns, risk_free_rate)
                                            volatility = features.calculate_volatility(ticker_returns, annualized=True) * 100
                                            radar_metrics[watch_ticker] = {
                                                'Sharpe Ratio': sharpe,
                                                'Volatility': volatility
                                            }
                            except:
                                pass
                        
                        if radar_metrics:
                            # Create simplified radar for first metric
                            first_metric = list(radar_metrics.values())[0]
                            metric_name = list(first_metric.keys())[0]
                            metric_dict = {k: v[metric_name] for k, v in radar_metrics.items()}
                            fig_radar = charts.plot_performance_radar(metric_dict)
                            st.plotly_chart(fig_radar, use_container_width=True)
                    except Exception as e:
                        st.info("Radar chart requires at least 3 tickers with valid data")
            else:
                st.info("No data available for watchlist tickers")
        else:
            st.info("👆 Add tickers to your watchlist above to track multiple stocks")
    
    # ========== TAB 9: NEWS & INSIGHTS ==========
    with tab9:
        st.header("News & Insights")
        
        # Fetch data on-demand for this tab
        with st.spinner("Loading news and insights data..."):
            news_data = data.fetch_news(ticker, num_articles=10)
            recommendations_data = data.fetch_recommendations(ticker)
            institutional_holders = data.fetch_institutional_holders(ticker)
            major_holders = data.fetch_major_holders(ticker)
            calendar_data = data.fetch_calendar(ticker)
            options_dates = data.fetch_options_dates(ticker)
        
        # News Section
        st.subheader("📰 Latest News")
        if news_data and len(news_data) > 0:
            for i, article in enumerate(news_data[:10]):  # Show top 10 articles
                with st.expander(f"**{article.get('title', 'No Title')}** - {article.get('publisher', 'Unknown')}"):
                    if article.get('link'):
                        st.markdown(f"[Read Full Article]({article['link']})")
                    if article.get('providerPublishTime'):
                        pub_time = datetime.fromtimestamp(article['providerPublishTime'])
                        st.caption(f"Published: {pub_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    if article.get('relatedTickers'):
                        st.caption(f"Related: {', '.join(article['relatedTickers'])}")
        else:
            st.info("No news available for this ticker.")
        
        st.divider()
        
        # Analyst Recommendations
        st.subheader("📊 Analyst Recommendations")
        if recommendations_data is not None and not recommendations_data.empty:
            st.dataframe(recommendations_data, use_container_width=True)
            
            # Summary of recommendations
            if 'To Grade' in recommendations_data.columns:
                grade_counts = recommendations_data['To Grade'].value_counts()
                col_rec1, col_rec2, col_rec3 = st.columns(3)
                with col_rec1:
                    st.metric("Total Recommendations", len(recommendations_data))
                with col_rec2:
                    if len(grade_counts) > 0:
                        most_common = grade_counts.index[0]
                        st.metric("Most Common", most_common)
                with col_rec3:
                    if len(grade_counts) > 0:
                        st.metric("Count", grade_counts.iloc[0])
        else:
            st.info("Analyst recommendations not available for this ticker.")
        
        st.divider()
        
        # Institutional Holders
        st.subheader("🏛️ Institutional Holders")
        if institutional_holders is not None and not institutional_holders.empty:
            st.dataframe(institutional_holders, use_container_width=True)
            
            # Total institutional ownership
            if 'Value' in institutional_holders.columns:
                total_value = institutional_holders['Value'].sum()
                st.metric("Total Institutional Holdings Value", f"${total_value:,.0f}")
        else:
            st.info("Institutional holders data not available for this ticker.")
        
        st.divider()
        
        # Major Holders
        st.subheader("👥 Major Holders")
        if major_holders is not None and not major_holders.empty:
            st.dataframe(major_holders, use_container_width=True)
        else:
            st.info("Major holders data not available for this ticker.")
        
        st.divider()
        
        # Calendar Events
        st.subheader("📅 Calendar Events")
        if calendar_data:
            calendar_df = pd.DataFrame([calendar_data])
            st.dataframe(calendar_df, use_container_width=True)
        else:
            st.info("Calendar events not available for this ticker.")
        
        st.divider()
        
        # Options Data
        st.subheader("📈 Options Data")
        if options_dates and len(options_dates) > 0:
            st.write(f"**Available Options Expiration Dates:** {len(options_dates)} dates")
            st.write("First 10 expiration dates:")
            options_df = pd.DataFrame({'Expiration Date': options_dates[:10]})
            st.dataframe(options_df, use_container_width=True)
            st.caption("💡 Options data is available. Use yfinance directly to fetch specific option chains.")
        else:
            st.info("Options data not available for this ticker.")

# Disclaimer
st.markdown("---")
st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This dashboard is for analysis purposes only and does not constitute investment advice. 
        Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial 
        advisor before making investment decisions.
    </div>
""", unsafe_allow_html=True)

