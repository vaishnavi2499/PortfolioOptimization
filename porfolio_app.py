import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# Streamlit Configuration
# -------------------------
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Portfolio Optimization Dashboard")
st.markdown("**Modern Portfolio Theory** - Find the optimal asset allocation to maximize risk-adjusted returns")


# -------------------------
# Helper Functions
# -------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_stock_data(stocks, start_date, end_date):
    """Download and process stock data with caching"""
    try:
        raw_data = yf.download(stocks, start=start_date, end=end_date, progress=False)

        if isinstance(raw_data.columns, pd.MultiIndex):
            if "Adj Close" in raw_data.columns.get_level_values(0):
                data = raw_data["Adj Close"]
            else:
                data = raw_data["Close"]
        else:
            data = raw_data

        # Handle missing data
        missing_pct = data.isnull().sum() / len(data)
        valid_stocks = missing_pct[missing_pct < 0.1].index.tolist()

        if len(valid_stocks) < len(stocks):
            st.warning(f"Removed stocks with >10% missing data: {[s for s in stocks if s not in valid_stocks]}")

        data = data[valid_stocks].ffill().dropna()
        return data, valid_stocks

    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None, None


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate portfolio performance metrics"""
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Negative Sharpe ratio for maximization"""
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe


def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Portfolio volatility for minimization"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))


def neg_portfolio_return(weights, mean_returns, cov_matrix):
    """Negative portfolio return for maximization"""
    return -np.dot(weights, mean_returns) * 252


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, optimization_method):
    """Run portfolio optimization"""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    if optimization_method == "Maximum Sharpe Ratio":
        result = minimize(neg_sharpe_ratio, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif optimization_method == "Minimum Volatility":
        result = minimize(portfolio_volatility, init_guess, args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif optimization_method == "Maximum Return":
        result = minimize(neg_portfolio_return, init_guess, args=(mean_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else init_guess


def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    """Generate efficient frontier points"""
    results = np.zeros((3, num_portfolios))
    target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, num_portfolios)

    num_assets = len(mean_returns)
    constraints_base = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    for i, target in enumerate(target_returns):
        target_constraint = {'type': 'eq', 'fun': lambda x, target=target: np.dot(x, mean_returns) * 252 - target}
        constraints = [constraints_base, target_constraint]

        try:
            result = minimize(portfolio_volatility, init_guess,
                              args=(mean_returns, cov_matrix),
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                ret, vol, sharpe = portfolio_performance(result.x, mean_returns, cov_matrix)
                results[0, i] = ret
                results[1, i] = vol
                results[2, i] = sharpe
            else:
                results[:, i] = np.nan
        except:
            results[:, i] = np.nan

    return results


# -------------------------
# Sidebar Configuration
# -------------------------
st.sidebar.header("🔧 Portfolio Configuration")

# Stock selection
st.sidebar.subheader("Stock Selection")
default_stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"]
selected_stocks = st.sidebar.multiselect(
    "Select stocks (3-10 recommended):",
    default_stocks + ["SPY", "QQQ", "VTI", "BRK-B", "UNH", "HD", "PG", "DIS", "NFLX", "CRM"],
    default=default_stocks[:5]
)

# Date selection
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", date(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", date(2024, 1, 1))

# Risk-free rate
risk_free_rate = st.sidebar.slider(
    "Risk-free Rate (%)",
    min_value=0.0, max_value=10.0, value=2.0, step=0.1
) / 100

# Optimization method
optimization_method = st.sidebar.selectbox(
    "Optimization Strategy:",
    ["Maximum Sharpe Ratio", "Minimum Volatility", "Maximum Return"]
)

# Run optimization button
run_optimization = st.sidebar.button("🚀 Run Optimization", type="primary")

# -------------------------
# Main Content
# -------------------------
if len(selected_stocks) < 2:
    st.warning("Please select at least 2 stocks to continue.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if run_optimization:
    with st.spinner("Downloading stock data and running optimization..."):
        # Download data
        data, valid_stocks = download_stock_data(selected_stocks, start_date, end_date)

        if data is None or data.empty:
            st.error("Failed to download stock data. Please try different stocks or date range.")
            st.stop()

        # Calculate returns and statistics
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Store results in session state
        st.session_state['data'] = data
        st.session_state['returns'] = returns
        st.session_state['mean_returns'] = mean_returns
        st.session_state['cov_matrix'] = cov_matrix
        st.session_state['valid_stocks'] = valid_stocks
        st.session_state['optimization_run'] = True

# Display results if optimization has been run
if st.session_state.get('optimization_run', False):
    data = st.session_state['data']
    returns = st.session_state['returns']
    mean_returns = st.session_state['mean_returns']
    cov_matrix = st.session_state['cov_matrix']
    valid_stocks = st.session_state['valid_stocks']

    # Run optimization
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, optimization_method)
    opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix,
                                                                   risk_free_rate)

    # -------------------------
    # Results Display
    # -------------------------
    st.header("📊 Optimization Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Annual Return", f"{opt_return:.2%}")
    with col2:
        st.metric("Annual Volatility", f"{opt_volatility:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}")
    with col4:
        st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

    # Portfolio allocation
    st.subheader("🥧 Optimal Portfolio Allocation")

    # Create allocation DataFrame
    allocation_df = pd.DataFrame({
        'Stock': valid_stocks,
        'Weight': optimal_weights,
        'Weight_Pct': [f"{w:.2%}" for w in optimal_weights]
    }).sort_values('Weight', ascending=False)

    # Show only significant allocations
    significant_allocations = allocation_df[allocation_df['Weight'] > 0.01]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            significant_allocations[['Stock', 'Weight_Pct']].reset_index(drop=True),
            hide_index=True,
            use_container_width=True
        )

    with col2:
        # Pie chart
        fig_pie = px.pie(
            significant_allocations,
            values='Weight',
            names='Stock',
            title="Portfolio Allocation"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------
    # Multiple Strategies Comparison
    # -------------------------
    st.subheader("📈 Strategy Comparison")

    strategies = ["Maximum Sharpe Ratio", "Minimum Volatility", "Maximum Return"]
    strategy_results = {}

    for strategy in strategies:
        weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, strategy)
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        strategy_results[strategy] = {
            'Return': ret,
            'Volatility': vol,
            'Sharpe': sharpe,
            'Weights': weights
        }

    # Comparison table
    comparison_df = pd.DataFrame({
        'Strategy': strategies,
        'Expected Return': [f"{strategy_results[s]['Return']:.2%}" for s in strategies],
        'Volatility': [f"{strategy_results[s]['Volatility']:.2%}" for s in strategies],
        'Sharpe Ratio': [f"{strategy_results[s]['Sharpe']:.2f}" for s in strategies]
    })

    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # -------------------------
    # Efficient Frontier
    # -------------------------
    st.subheader("📊 Efficient Frontier")

    with st.spinner("Generating efficient frontier..."):
        frontier_results = generate_efficient_frontier(mean_returns, cov_matrix)

    # Plot efficient frontier
    fig = go.Figure()

    # Add efficient frontier
    valid_points = ~np.isnan(frontier_results[0])
    fig.add_trace(go.Scatter(
        x=frontier_results[1][valid_points],
        y=frontier_results[0][valid_points],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=3)
    ))

    # Add individual stocks
    annualized_returns = mean_returns * 252
    annualized_volatility = returns.std() * np.sqrt(252)

    for stock in valid_stocks:
        fig.add_trace(go.Scatter(
            x=[annualized_volatility[stock]],
            y=[annualized_returns[stock]],
            mode='markers',
            name=stock,
            marker=dict(size=10, opacity=0.7)
        ))

    # Add strategy points
    colors = ['red', 'green', 'orange']
    symbols = ['star', 'diamond', 'triangle-up']

    for i, strategy in enumerate(strategies):
        fig.add_trace(go.Scatter(
            x=[strategy_results[strategy]['Volatility']],
            y=[strategy_results[strategy]['Return']],
            mode='markers',
            name=strategy,
            marker=dict(
                symbol=symbols[i],
                size=15,
                color=colors[i],
                line=dict(width=2, color='black')
            )
        ))

    fig.update_layout(
        title="Risk-Return Profile with Efficient Frontier",
        xaxis_title="Volatility (Annual)",
        yaxis_title="Expected Return (Annual)",
        height=600,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # Additional Analysis
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Correlation Matrix")
        corr_matrix = returns.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Stock Correlation Matrix",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("📈 Historical Performance")

        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()

        fig_perf = go.Figure()
        for stock in valid_stocks:
            fig_perf.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[stock],
                mode='lines',
                name=stock
            ))

        fig_perf.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    # -------------------------
    # Risk Analysis
    # -------------------------
    st.subheader("⚠️ Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Portfolio Beta vs Market", "N/A", help="Add market index for beta calculation")

    with col2:
        # Calculate Value at Risk (VaR)
        portfolio_returns = np.dot(returns, optimal_weights)
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        st.metric("VaR (95%, Annual)", f"{var_95:.2%}", help="Maximum expected loss 95% of the time")

    with col3:
        # Maximum Drawdown
        portfolio_returns_series = pd.Series(portfolio_returns, index=returns.index)
        portfolio_cumret = (1 + portfolio_returns_series).cumprod()
        rolling_max = portfolio_cumret.expanding().max()
        drawdown = (portfolio_cumret - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}", help="Largest peak-to-trough decline")

else:
    # Show initial information
    st.info("👈 Configure your portfolio parameters in the sidebar and click 'Run Optimization' to get started!")

    st.subheader("🎯 How to Use This Tool")
    st.markdown("""
    1. **Select Stocks**: Choose 2-10 stocks from the dropdown
    2. **Set Date Range**: Pick your historical data period
    3. **Choose Risk-Free Rate**: Current treasury bill rate (default: 2%)
    4. **Select Strategy**: 
       - **Maximum Sharpe Ratio**: Best risk-adjusted returns
       - **Minimum Volatility**: Lowest risk portfolio
       - **Maximum Return**: Highest expected returns
    5. **Run Optimization**: Click the button to generate results

    **Features:**
    - Interactive efficient frontier visualization
    - Multiple optimization strategies comparison  
    - Risk metrics and correlation analysis
    - Historical performance charts
    - Portfolio allocation recommendations
    """)

    st.subheader("⚠️ Important Disclaimers")
    st.warning("""
    - **Past performance doesn't guarantee future results**
    - This tool is for educational purposes only
    - Consider transaction costs and taxes in real investing
    - Consult a financial advisor for personalized advice
    - Market conditions can change rapidly
    """)