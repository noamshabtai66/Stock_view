# Stock Analysis Dashboard

A comprehensive, production-grade Stock Analysis Dashboard built with Python and Streamlit. This dashboard provides advanced technical analysis, risk metrics, fundamental data, and comparison tools using free data from yfinance.

## Features

### 🎨 Enhanced UI/UX
- **Modern Design**: Gradient backgrounds, smooth animations, and professional styling
- **JSON Configuration**: Customize colors, indicators, and settings via `config/ui_config.json`
- **Responsive Layout**: Optimized for all screen sizes
- **Progress Indicators**: Visual feedback during data loading

### 📊 Overview Tab
- Real-time price display with daily change
- Interactive candlestick chart with volume
- **Volume Profile Chart**: Price distribution analysis
- Summary statistics (mean return, volatility, skewness, etc.)
- Recent price action table

### 📈 Technical Analysis Tab
- **Indicators:**
  - Simple Moving Averages (SMA 20, 50, 200)
  - Exponential Moving Averages (EMA 12, 26)
  - Relative Strength Index (RSI 14)
  - MACD (12, 26, 9) with histogram
  - Bollinger Bands (20, 2)
  - Volume Moving Average
  
- **Trading Signals:**
  - Golden Cross / Death Cross (SMA50 vs SMA200)
  - RSI Overbought/Oversold signals
  - MACD cross signals
  - Price vs SMA200 signals

### ⚠️ Risk & Performance Tab
- **Risk Metrics:**
  - Annualized Volatility
  - Sharpe Ratio
  - Sortino Ratio
  - Value at Risk (VaR 95%)
  - Conditional VaR (CVaR 95%)
  - Maximum Drawdown
  
- **Performance Analysis:**
  - Drawdown visualization
  - Rolling volatility and returns
  - Performance statistics table

### 💼 Fundamentals Tab
- Company information cards (Market Cap, Sector, Industry, P/E ratios)
- Financial statements (Income Statement, Balance Sheet, Cash Flow)
- Growth metrics (Revenue growth, Profit margin)

### 📅 Events & Dividends Tab
- Dividends history chart and total dividends
- Stock splits table
- Earnings dates (if available)

### 🔀 Compare & Correlation Tab
- Multi-ticker comparison (cumulative returns)
- Correlation heatmap
- Beta calculation vs benchmarks
- Performance comparison table

### 💾 Data Explorer Tab
- Interactive dataframe with all data and indicators
- **Multiple Export Formats**:
  - CSV export (Price data + indicators, Signals table, Risk metrics)
  - **JSON export** (Complete data with metadata)
  - **Excel export** (Multi-sheet workbook with all data)

### 👁️ Watchlist Tab (NEW!)
- **Multi-Ticker Tracking**: Add and manage multiple stocks
- Quick comparison table with prices and changes
- Performance radar chart for visual comparison
- Easy add/remove functionality

### 📰 News & Insights Tab (NEW!)
- **Latest News**: Top 10 news articles with links and timestamps
- **Analyst Recommendations**: Historical analyst recommendations and ratings
- **Institutional Holders**: List of major institutional investors
- **Major Holders**: Top shareholders information
- **Calendar Events**: Earnings dates and important events
- **Options Data**: Available options expiration dates

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

## Usage

1. **Enter a ticker symbol** in the sidebar (e.g., AAPL, TSLA, MSFT)
2. **Select time period** (1M, 3M, 6M, 1Y, 5Y, MAX)
3. **Choose data interval** (1d, 1wk, 1mo)
4. **Toggle indicators** on/off as needed
5. **Select benchmarks** for comparison (SPY, QQQ, etc.)
6. **Navigate through tabs** to explore different analyses
7. **Export data** from the Data Explorer tab

## Project Structure

```
Stock Dashboard/
├── app.py                 # Main Streamlit application
├── config/
│   └── ui_config.json    # UI configuration (colors, indicators, etc.)
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── data.py           # Data fetching and caching
│   ├── features.py       # Technical indicators and risk metrics
│   ├── charts.py         # Plotly chart generation
│   ├── ui.py             # UI components
│   └── utils.py          # Utility functions and helpers
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies

- **streamlit** >= 1.28.0 - Web framework
- **pandas** >= 2.0.0 - Data manipulation
- **numpy** >= 1.24.0 - Numerical calculations
- **yfinance** >= 0.2.28 - Stock data API
- **plotly** >= 5.17.0 - Interactive charts
- **openpyxl** >= 3.1.0 - Excel export support (optional but recommended)

## Key Features & Robustness

### Error Handling
- All yfinance API calls are wrapped in try/except blocks
- Missing data displays "Not available" instead of crashing
- Validates data availability before calculations
- Graceful fallbacks for missing indicators
- Comprehensive logging system

### Caching
- **st.cache_data**: Used for serializable data (DataFrames, dictionaries)
- **st.cache_resource**: Used for non-serializable objects (yfinance.Ticker)
- Price data cached for 5 minutes (frequent updates)
- Static info cached for 1 hour
- Manual refresh button to clear cache
- Efficient data fetching with proper cache keys
- Progress indicators during data loading

### Chart Fallback
- **Plotly** (recommended): Interactive, modern charts
- **Matplotlib** (fallback): Automatic fallback if plotly not available
- All charts work with both libraries seamlessly

### Data Validation
- Checks minimum data points before calculating indicators
- Validates date ranges and intervals
- Handles missing columns gracefully
- Sanity checks for all calculations

### Configuration
- **JSON-based configuration**: Customize UI colors, indicator periods, and benchmarks
- Edit `config/ui_config.json` to personalize the dashboard
- Hot-reload configuration without code changes
- Fallback to defaults if config file is missing

### Export Capabilities
- **CSV**: Standard format for data analysis
- **JSON**: Structured data with metadata for programmatic access
- **Excel**: Multi-sheet workbook with formatted data
- All exports include timestamps and ticker information

## Troubleshooting

### yfinance returns None/empty data
**Problem:** No data available for a ticker.

**Solutions:**
- Verify the ticker symbol is correct
- Check if the market is open (for intraday data)
- Try a different time period or interval
- Some tickers may not be available in yfinance

### Missing indicators (insufficient data points)
**Problem:** Indicators like SMA200 require at least 200 data points.

**Solutions:**
- Select a longer time period (e.g., 1Y or 5Y)
- Use a daily interval (1d) instead of weekly/monthly
- The app will automatically skip indicators that can't be calculated

### Network issues
**Problem:** Slow loading or connection errors.

**Solutions:**
- Check your internet connection
- Wait a few seconds and refresh
- Try clicking "Refresh Data" button
- yfinance may have rate limits - wait before retrying

### Charts not displaying
**Problem:** Plotly charts not showing.

**Solutions:**
- Ensure plotly is installed: `pip install plotly`
- Check browser console for errors
- Try refreshing the page
- Clear browser cache

### Fundamentals not available
**Problem:** Some tickers don't have fundamental data in yfinance.

**Solutions:**
- This is normal for some tickers (ETFs, indices, etc.)
- The app will display "Not available" messages
- Try a different ticker (e.g., AAPL, MSFT, GOOGL)

## Performance Tips

1. **Use appropriate time periods:** Longer periods take more time to load
2. **Cache is your friend:** Data is cached for faster subsequent loads
3. **Disable unused indicators:** Turn off indicators you don't need
4. **Limit benchmark selection:** Too many benchmarks slow down comparison
5. **Watchlist size:** Keep watchlist under 10 tickers for optimal performance
6. **Export format:** JSON is fastest, Excel is most comprehensive

## Customization

### UI Configuration

Edit `config/ui_config.json` to customize:

- **Theme colors**: Primary, secondary, success, danger colors
- **Chart colors**: Candlestick and volume colors
- **Indicator periods**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Default benchmarks**: Set your preferred comparison tickers

Example configuration:
```json
{
  "theme": {
    "primary_color": "#1f77b4",
    "secondary_color": "#ff7f0e"
  },
  "indicators": {
    "sma_periods": [20, 50, 200],
    "rsi_period": 14
  }
}
```

## Limitations

- Data depends on yfinance availability and accuracy
- Some tickers may have limited data
- Real-time data may have delays
- No intraday data for intervals < 1d
- Some fundamental data may be incomplete

## Disclaimer

⚠️ **Important:** This dashboard is for analysis purposes only and does not constitute investment advice. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## License

This project is provided as-is for educational and analysis purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review yfinance documentation: https://github.com/ranaroussi/yfinance
3. Check Streamlit documentation: https://docs.streamlit.io

