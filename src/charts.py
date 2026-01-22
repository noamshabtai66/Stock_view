"""
Plotly chart generation functions for Stock Analysis Dashboard.
All charts are interactive and responsive.
Includes fallback to matplotlib if plotly is not available.
"""

# Try to import plotly, fallback to matplotlib if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import warnings
    warnings.warn("Plotly not available, using matplotlib fallback")

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def plot_candlestick(
    df: pd.DataFrame,
    sma_data: Optional[Dict] = None,
    ema_data: Optional[Dict] = None,
    bollinger_data: Optional[pd.DataFrame] = None,
    volume: bool = True
):
    """
    Create candlestick chart with technical indicator overlays.
    Uses plotly if available, falls back to matplotlib.
    
    Args:
        df: DataFrame with OHLCV data
        sma_data: DataFrame with SMA columns
        ema_data: DataFrame with EMA columns
        bollinger_data: DataFrame with Bollinger Bands columns
        volume: Whether to include volume subplot
    
    Returns:
        Plotly figure object or matplotlib figure
    """
    if df is None or df.empty:
        if PLOTLY_AVAILABLE:
            return go.Figure()
        else:
            return plt.figure()
    
    if PLOTLY_AVAILABLE:
        return _plot_candlestick_plotly(df, sma_data, ema_data, bollinger_data, volume)
    else:
        return _plot_candlestick_matplotlib(df, sma_data, ema_data, bollinger_data, volume)


def _plot_candlestick_plotly(
    df: pd.DataFrame,
    sma_data: Optional[Dict] = None,
    ema_data: Optional[Dict] = None,
    bollinger_data: Optional[pd.DataFrame] = None,
    volume: bool = True
) -> go.Figure:
    """Plotly implementation of candlestick chart."""
    if df is None or df.empty:
        return go.Figure()
    
    # Determine subplot configuration
    if volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Candlestick chart with config colors
    try:
        from src import config
        ui_config = config.load_config()
        chart_config = ui_config.get("charts", {})
        candlestick_colors = chart_config.get("candlestick_colors", 
                                              {"increasing": "#26a69a", "decreasing": "#ef5350"})
    except:
        candlestick_colors = {"increasing": "#26a69a", "decreasing": "#ef5350"}
    
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=candlestick_colors["increasing"],
        decreasing_line_color=candlestick_colors["decreasing"],
        increasing_fillcolor=candlestick_colors["increasing"],
        decreasing_fillcolor=candlestick_colors["decreasing"]
    )
    
    if volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add SMA lines
    if sma_data is not None:
        for col in sma_data.columns:
            if col.startswith('SMA_'):
                sma_trace = go.Scatter(
                    x=df.index,
                    y=sma_data[col],
                    name=col,
                    line=dict(width=1.5),
                    hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>'
                )
                if volume:
                    fig.add_trace(sma_trace, row=1, col=1)
                else:
                    fig.add_trace(sma_trace)
    
    # Add EMA lines
    if ema_data is not None:
        for col in ema_data.columns:
            if col.startswith('EMA_'):
                ema_trace = go.Scatter(
                    x=df.index,
                    y=ema_data[col],
                    name=col,
                    line=dict(width=1.5, dash='dot'),
                    hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>'
                )
                if volume:
                    fig.add_trace(ema_trace, row=1, col=1)
                else:
                    fig.add_trace(ema_trace)
    
    # Add Bollinger Bands
    if bollinger_data is not None:
        if 'BB_Upper' in bollinger_data.columns:
            bb_upper = go.Scatter(
                x=df.index,
                y=bollinger_data['BB_Upper'],
                name='BB Upper',
                line=dict(width=1, color='rgba(200,200,200,0.5)'),
                showlegend=False,
                hovertemplate='BB Upper: %{y:.2f}<extra></extra>'
            )
            if volume:
                fig.add_trace(bb_upper, row=1, col=1)
            else:
                fig.add_trace(bb_upper)
        
        if 'BB_Middle' in bollinger_data.columns:
            bb_middle = go.Scatter(
                x=df.index,
                y=bollinger_data['BB_Middle'],
                name='BB Middle',
                line=dict(width=1, color='rgba(150,150,150,0.5)'),
                hovertemplate='BB Middle: %{y:.2f}<extra></extra>'
            )
            if volume:
                fig.add_trace(bb_middle, row=1, col=1)
            else:
                fig.add_trace(bb_middle)
        
        if 'BB_Lower' in bollinger_data.columns:
            bb_lower = go.Scatter(
                x=df.index,
                y=bollinger_data['BB_Lower'],
                name='BB Lower',
                line=dict(width=1, color='rgba(200,200,200,0.5)'),
                fill='tonexty',
                fillcolor='rgba(200,200,200,0.1)',
                hovertemplate='BB Lower: %{y:.2f}<extra></extra>'
            )
            if volume:
                fig.add_trace(bb_lower, row=1, col=1)
            else:
                fig.add_trace(bb_lower)
    
    # Add volume bars with config colors
    if volume and 'Volume' in df.columns:
        try:
            from src import config
            ui_config = config.load_config()
            chart_config = ui_config.get("charts", {})
            volume_colors = chart_config.get("volume_colors", {"up": "#26a69a", "down": "#ef5350"})
        except:
            volume_colors = {"up": "#26a69a", "down": "#ef5350"}
        
        colors = [volume_colors["down"] if df['Close'].iloc[i] < df['Open'].iloc[i] 
                 else volume_colors["up"] for i in range(len(df))]
        
        volume_trace = go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False,
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        )
        fig.add_trace(volume_trace, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title='Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    if volume:
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    return fig


def plot_rsi(df: pd.DataFrame, rsi_data: pd.Series):
    """Create RSI chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_rsi_plotly(df, rsi_data)
    else:
        return _plot_rsi_matplotlib(df, rsi_data)


def _plot_rsi_plotly(df: pd.DataFrame, rsi_data: pd.Series) -> go.Figure:
    """
    Create RSI chart with overbought/oversold lines.
    
    Args:
        df: DataFrame with date index
        rsi_data: Series with RSI values
    
    Returns:
        Plotly figure object
    """
    if df is None or df.empty or rsi_data is None or len(rsi_data) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    # RSI line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi_data,
        name='RSI',
        line=dict(color='blue', width=2),
        hovertemplate='RSI: %{y:.2f}<extra></extra>'
    ))
    
    # Overbought line (70)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Overbought (70)",
        annotation_position="right"
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Oversold (30)",
        annotation_position="right"
    )
    
    # Neutral line (50)
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="gray",
        opacity=0.5
    )
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis_range=[0, 100],
        height=400,
        hovermode='x unified'
    )
    
    return fig


def _plot_rsi_matplotlib(df: pd.DataFrame, rsi_data: pd.Series):
    """Matplotlib fallback for RSI chart."""
    if df is None or df.empty or rsi_data is None or len(rsi_data) == 0:
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, rsi_data.values, label='RSI', linewidth=2, color='blue')
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_title('Relative Strength Index (RSI)')
    ax.set_ylabel('RSI')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_macd(df: pd.DataFrame, macd_data: pd.DataFrame):
    """Create MACD chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_macd_plotly(df, macd_data)
    else:
        return _plot_macd_matplotlib(df, macd_data)


def _plot_macd_plotly(df: pd.DataFrame, macd_data: pd.DataFrame) -> go.Figure:
    """
    Create MACD chart with signal line and histogram.
    
    Args:
        df: DataFrame with date index
        macd_data: DataFrame with MACD, MACD_Signal, MACD_Histogram columns
    
    Returns:
        Plotly figure object
    """
    if df is None or df.empty or macd_data is None or macd_data.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('MACD & Signal', 'Histogram')
    )
    
    # MACD line
    if 'MACD' in macd_data.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=macd_data['MACD'],
            name='MACD',
            line=dict(color='blue', width=2),
            hovertemplate='MACD: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
    
    # Signal line
    if 'MACD_Signal' in macd_data.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=macd_data['MACD_Signal'],
            name='Signal',
            line=dict(color='red', width=2),
            hovertemplate='Signal: %{y:.4f}<extra></extra>'
        ), row=1, col=1)
    
    # Histogram
    if 'MACD_Histogram' in macd_data.columns:
        colors = ['green' if x >= 0 else 'red' for x in macd_data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=df.index,
            y=macd_data['MACD_Histogram'],
            name='Histogram',
            marker_color=colors,
            hovertemplate='Histogram: %{y:.4f}<extra></extra>'
        ), row=2, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)',
        height=500,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='MACD', row=1, col=1)
    fig.update_yaxes(title_text='Histogram', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    return fig


def _plot_macd_matplotlib(df: pd.DataFrame, macd_data: pd.DataFrame):
    """Matplotlib fallback for MACD chart."""
    if df is None or df.empty or macd_data is None or macd_data.empty:
        return plt.figure()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # MACD and Signal lines
    if 'MACD' in macd_data.columns:
        ax1.plot(df.index, macd_data['MACD'].values, label='MACD', 
                linewidth=2, color='blue')
    if 'MACD_Signal' in macd_data.columns:
        ax1.plot(df.index, macd_data['MACD_Signal'].values, label='Signal', 
                linewidth=2, color='red')
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('MACD (Moving Average Convergence Divergence)')
    ax1.set_ylabel('MACD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    if 'MACD_Histogram' in macd_data.columns:
        colors = ['green' if x >= 0 else 'red' for x in macd_data['MACD_Histogram']]
        ax2.bar(df.index, macd_data['MACD_Histogram'].values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Histogram')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_drawdown(drawdown_series: pd.Series):
    """Create drawdown chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_drawdown_plotly(drawdown_series)
    else:
        return _plot_drawdown_matplotlib(drawdown_series)


def _plot_drawdown_plotly(drawdown_series: pd.Series) -> go.Figure:
    """
    Create drawdown visualization chart.
    
    Args:
        drawdown_series: Series with drawdown values
    
    Returns:
        Plotly figure object
    """
    if drawdown_series is None or len(drawdown_series) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    # Drawdown area
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series * 100,  # Convert to percentage
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=2),
        name='Drawdown',
        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def _plot_drawdown_matplotlib(drawdown_series: pd.Series):
    """Matplotlib fallback for drawdown chart."""
    if drawdown_series is None or len(drawdown_series) == 0:
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown_series.index, drawdown_series.values * 100, 0,
                    color='red', alpha=0.3)
    ax.plot(drawdown_series.index, drawdown_series.values * 100, 
           color='red', linewidth=2, label='Drawdown')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Drawdown Analysis')
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_returns_comparison(comparison_df: pd.DataFrame):
    """Create returns comparison chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_returns_comparison_plotly(comparison_df)
    else:
        return _plot_returns_comparison_matplotlib(comparison_df)


def _plot_returns_comparison_plotly(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative returns comparison chart.
    
    Args:
        comparison_df: DataFrame with cumulative returns for multiple tickers
    
    Returns:
        Plotly figure object
    """
    if comparison_df is None or comparison_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    for col in comparison_df.columns:
        fig.add_trace(go.Scatter(
            x=comparison_df.index,
            y=comparison_df[col] * 100,  # Convert to percentage
            name=col,
            line=dict(width=2),
            hovertemplate=f'{col}: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def _plot_returns_comparison_matplotlib(comparison_df: pd.DataFrame):
    """Matplotlib fallback for returns comparison chart."""
    if comparison_df is None or comparison_df.empty:
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in comparison_df.columns:
        ax.plot(comparison_df.index, comparison_df[col].values * 100, 
               label=col, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Cumulative Returns Comparison')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame):
    """Create correlation heatmap with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_correlation_heatmap_plotly(correlation_matrix)
    else:
        return _plot_correlation_heatmap_matplotlib(correlation_matrix)


def _plot_correlation_heatmap_plotly(correlation_matrix: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        correlation_matrix: DataFrame with correlation values
    
    Returns:
        Plotly figure object
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=500,
        xaxis_title='',
        yaxis_title=''
    )
    
    return fig


def _plot_correlation_heatmap_matplotlib(correlation_matrix: pd.DataFrame):
    """Matplotlib fallback for correlation heatmap."""
    if correlation_matrix is None or correlation_matrix.empty:
        return plt.figure()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title('Correlation Heatmap')
    
    # Add text annotations
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_dividends_history(dividends_df: pd.DataFrame):
    """Create dividends chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_dividends_history_plotly(dividends_df)
    else:
        return _plot_dividends_history_matplotlib(dividends_df)


def _plot_dividends_history_plotly(dividends_df: pd.DataFrame) -> go.Figure:
    """
    Create dividends history chart.
    
    Args:
        dividends_df: DataFrame with dividends data (index=date, column=Dividends)
    
    Returns:
        Plotly figure object
    """
    if dividends_df is None or dividends_df.empty:
        return go.Figure()
    
    # Get the dividends column (could be 'Dividends' or first column)
    div_col = 'Dividends' if 'Dividends' in dividends_df.columns else dividends_df.columns[0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dividends_df.index,
        y=dividends_df[div_col],
        name='Dividends',
        marker_color='green',
        hovertemplate='Date: %{x}<br>Dividend: $%{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Dividends History',
        xaxis_title='Date',
        yaxis_title='Dividend Amount ($)',
        height=400,
        hovermode='x unified'
    )
    
    return fig


def _plot_dividends_history_matplotlib(dividends_df: pd.DataFrame):
    """Matplotlib fallback for dividends chart."""
    if dividends_df is None or dividends_df.empty:
        return plt.figure()
    
    div_col = 'Dividends' if 'Dividends' in dividends_df.columns else dividends_df.columns[0]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(dividends_df.index, dividends_df[div_col].values, 
          color='green', alpha=0.7)
    ax.set_title('Dividends History')
    ax.set_ylabel('Dividend Amount ($)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_performance_radar(metrics_dict: Dict[str, float]):
    """Create radar chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_performance_radar_plotly(metrics_dict)
    else:
        return _plot_performance_radar_matplotlib(metrics_dict)


def _plot_performance_radar_plotly(metrics_dict: Dict[str, float]) -> go.Figure:
    """
    Create radar chart for performance metrics.
    
    Args:
        metrics_dict: Dictionary with metric names and values
    
    Returns:
        Plotly figure object
    """
    if not metrics_dict:
        return go.Figure()
    
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Normalize values to 0-100 scale for better visualization
    max_val = max(abs(v) for v in values) if values else 1
    normalized_values = [(v / max_val * 100) if max_val > 0 else 0 for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Performance',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Performance Radar Chart",
        height=500
    )
    
    return fig


def _plot_performance_radar_matplotlib(metrics_dict: Dict[str, float]):
    """Matplotlib fallback for radar chart."""
    if not metrics_dict:
        return plt.figure()
    
    # Matplotlib doesn't have native radar charts, use polar plot
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    # Normalize values
    max_val = max(abs(v) for v in values) if values else 1
    normalized_values = [(v / max_val * 100) if max_val > 0 else 0 for v in values]
    
    angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]  # Complete the circle
    normalized_values += normalized_values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, normalized_values, 'o-', linewidth=2, label='Performance')
    ax.fill(angles, normalized_values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('Performance Radar Chart', pad=20)
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_volume_profile(df: pd.DataFrame, bins: int = 20):
    """Create volume profile chart with fallback to matplotlib."""
    if PLOTLY_AVAILABLE:
        return _plot_volume_profile_plotly(df, bins)
    else:
        return _plot_volume_profile_matplotlib(df, bins)


def _plot_volume_profile_plotly(df: pd.DataFrame, bins: int = 20) -> go.Figure:
    """
    Create volume profile chart.
    
    Args:
        df: DataFrame with 'Close' and 'Volume' columns
        bins: Number of price bins
    
    Returns:
        Plotly figure object
    """
    if df is None or df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        return go.Figure()
    
    # Group prices into bins and sum volume
    try:
        price_bins = pd.cut(df['Close'], bins=bins)
        volume_profile = df.groupby(price_bins)['Volume'].sum()
        
        # Get bin midpoints
        bin_midpoints = [interval.mid for interval in volume_profile.index]
        volumes = volume_profile.values
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=bin_midpoints,
            y=volumes,
            orientation='v',
            name='Volume Profile',
            marker_color='rgba(55, 128, 191, 0.7)',
            hovertemplate='Price: $%{x:.2f}<br>Volume: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Volume Profile',
            xaxis_title='Price',
            yaxis_title='Volume',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    except Exception:
        return go.Figure()


def _plot_volume_profile_matplotlib(df: pd.DataFrame, bins: int = 20):
    """Matplotlib fallback for volume profile chart."""
    if df is None or df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
        return plt.figure()
    
    try:
        price_bins = pd.cut(df['Close'], bins=bins)
        volume_profile = df.groupby(price_bins)['Volume'].sum()
        bin_midpoints = [interval.mid for interval in volume_profile.index]
        volumes = volume_profile.values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(bin_midpoints, volumes, color='steelblue', alpha=0.7)
        ax.set_title('Volume Profile')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        return fig
    except Exception:
        return plt.figure()

