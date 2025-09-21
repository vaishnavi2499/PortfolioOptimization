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
import time

# Optional Qiskit imports for quantum optimization
try:
    from qiskit.circuit.library import TwoLocal
    from qiskit.result import QuasiDistribution
    from qiskit_aer.primitives import Sampler
    from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.utils import algorithm_globals
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Portfolio Optimization Dashboard")
st.markdown(
    "**Classical & Quantum Portfolio Optimization** - Find optimal asset allocation using modern portfolio theory and quantum algorithms")


# -------------------------
# Helper Functions - Classical
# -------------------------
@st.cache_data(ttl=3600)
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


def optimize_portfolio_classical(mean_returns, cov_matrix, risk_free_rate, optimization_method):
    """Run classical portfolio optimization"""
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
# Helper Functions - Quantum
# -------------------------
class YahooDataProvider:
    """Yahoo Finance data provider for quantum optimization"""

    def __init__(self, tickers, start, end):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.data = None
        self.returns = None

    def run(self):
        """Fetch data from Yahoo Finance and calculate returns"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text('Downloading stock data for quantum optimization...')
            progress_bar.progress(25)

            self.data = yf.download(self.tickers, start=self.start, end=self.end,
                                    progress=False, auto_adjust=False)

            progress_bar.progress(50)
            status_text.text('Processing quantum data...')

            # Handle different data structures
            if len(self.tickers) == 1:
                if 'Adj Close' in self.data.columns:
                    prices = pd.DataFrame(self.data['Adj Close'])
                    prices.columns = self.tickers
                else:
                    prices = pd.DataFrame(self.data['Close'])
                    prices.columns = self.tickers
            else:
                if isinstance(self.data.columns, pd.MultiIndex):
                    if 'Adj Close' in self.data.columns.get_level_values(0):
                        prices = self.data['Adj Close']
                    else:
                        prices = self.data['Close']
                else:
                    prices = self.data

            # Clean the data
            if not isinstance(prices, pd.DataFrame):
                prices = pd.DataFrame(prices)

            prices = prices.dropna()

            if len(prices.columns) == len(self.tickers):
                prices.columns = self.tickers

            progress_bar.progress(75)
            status_text.text('Calculating quantum returns...')

            # Calculate returns
            self.returns = prices.pct_change().dropna()

            progress_bar.progress(100)
            status_text.text('✅ Quantum data loaded successfully!')

            # Clear progress indicators
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            return True, prices

        except Exception as e:
            st.error(f"Error fetching quantum data: {e}")
            return False, None

    def get_period_return_mean_vector(self):
        return self.returns.mean().values if self.returns is not None else None

    def get_period_return_covariance_matrix(self):
        return self.returns.cov().values if self.returns is not None else None


def create_synthetic_data(tickers, seed=123):
    """Create synthetic data for testing"""
    np.random.seed(seed)
    n_days = 252
    n_assets = len(tickers)

    synthetic_returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    returns_df = pd.DataFrame(synthetic_returns, index=dates, columns=tickers)
    return returns_df


def run_qaoa_optimization(stocks, start_date, end_date, risk_factor, budget_ratio, max_iterations, qaoa_reps):
    """Run the QAOA optimization"""
    if not QUANTUM_AVAILABLE:
        st.error("Qiskit libraries not available. Please install qiskit packages for quantum optimization.")
        return None, None, None, None, None, None

    # Initialize data provider
    data_provider = YahooDataProvider(stocks, start_date, end_date)

    # Fetch data
    success, prices = data_provider.run()

    if not success:
        st.warning("Using synthetic data for demonstration")
        synthetic_returns = create_synthetic_data(stocks)
        mu = synthetic_returns.mean().values
        sigma = synthetic_returns.cov().values
    else:
        mu = data_provider.get_period_return_mean_vector()
        sigma = data_provider.get_period_return_covariance_matrix()

    # Portfolio parameters
    num_assets = len(stocks)
    budget = max(1, int(budget_ratio * num_assets))

    # Create portfolio optimization problem
    try:
        portfolio = PortfolioOptimization(
            expected_returns=mu,
            covariances=sigma,
            risk_factor=risk_factor,
            budget=budget
        )

        qp = portfolio.to_quadratic_program()

        # Set up QAOA
        algorithm_globals.random_seed = 1234

        optimization_progress = st.progress(0)
        optimization_status = st.empty()

        optimization_status.text("Setting up QAOA...")
        optimization_progress.progress(10)

        # Setup optimizer and sampler
        cobyla = COBYLA()
        cobyla.set_options(maxiter=max_iterations)

        try:
            sampler = Sampler()
            optimization_status.text("Using Aer Sampler...")
        except:
            from qiskit.primitives import Sampler
            sampler = Sampler()
            optimization_status.text("Using basic Sampler...")

        optimization_progress.progress(25)

        # Create QAOA
        qaoa_mes = QAOA(sampler=sampler, optimizer=cobyla, reps=qaoa_reps)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)

        optimization_status.text("Running QAOA optimization...")
        optimization_progress.progress(50)

        # Run optimization
        result = qaoa_optimizer.solve(qp)

        optimization_progress.progress(75)
        optimization_status.text("Running classical comparison...")

        # Classical comparison
        classical_solver = NumPyMinimumEigensolver()
        classical_optimizer = MinimumEigenOptimizer(classical_solver)
        classical_result = classical_optimizer.solve(qp)

        optimization_progress.progress(100)
        optimization_status.text("✅ QAOA optimization completed!")

        time.sleep(1)
        optimization_progress.empty()
        optimization_status.empty()

        return result, classical_result, mu, sigma, portfolio, prices

    except Exception as e:
        st.error(f"QAOA optimization failed: {e}")
        return None, None, None, None, None, None


# -------------------------
# Visualization Functions
# -------------------------
def create_classical_visualizations(data, returns, mean_returns, cov_matrix, valid_stocks, optimal_weights,
                                    opt_return, opt_volatility, opt_sharpe, risk_free_rate, optimization_method):
    """Create classical optimization visualizations"""

    st.header("📊 Classical Optimization Results")

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

    # Multiple Strategies Comparison
    st.subheader("📈 Strategy Comparison")

    strategies = ["Maximum Sharpe Ratio", "Minimum Volatility", "Maximum Return"]
    strategy_results = {}

    for strategy in strategies:
        weights = optimize_portfolio_classical(mean_returns, cov_matrix, risk_free_rate, strategy)
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

    # Efficient Frontier
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

    # Additional Analysis
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

    # Risk Analysis
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


def create_quantum_visualizations(result, classical_result, stocks, mu, sigma, prices):
    """Create quantum optimization visualizations"""

    st.header("🚀 Quantum Optimization Results")

    # Calculate metrics
    annual_returns = mu * 252
    annual_volatility = np.sqrt(np.diag(sigma)) * np.sqrt(252)

    # Selected assets
    selected_assets = [stock for i, stock in enumerate(stocks) if result.x[i] > 0.5]

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown("**🎯 QAOA Selected Portfolio:**")
        if selected_assets:
            for asset in selected_assets:
                st.markdown(f"✅ **{asset}**")
        else:
            st.markdown("❌ No assets selected")

        st.markdown(f"**Objective Value:** {result.fval:.6f}")

    with result_col2:
        st.markdown("**📊 Classical Comparison:**")
        classical_weights = classical_result.x
        top_classical = sorted(zip(stocks, classical_weights),
                               key=lambda x: x[1], reverse=True)[:3]
        for asset, weight in top_classical:
            st.markdown(f"📈 **{asset}**: {weight:.3f}")

        st.markdown(f"**Objective Value:** {classical_result.fval:.6f}")

    # Performance metrics
    if len(selected_assets) > 0:
        selected_indices = [i for i, x in enumerate(result.x) if x > 0.5]
        portfolio_return = np.mean([annual_returns[i] for i in selected_indices])
        portfolio_vol = np.mean([annual_volatility[i] for i in selected_indices])
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Portfolio Return", f"{portfolio_return:.1%}")
        with metric_col2:
            st.metric("Portfolio Volatility", f"{portfolio_vol:.1%}")
        with metric_col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # Create subplots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio Allocation Comparison")

        # Portfolio allocation comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('QAOA Selection', 'Classical Weights'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # QAOA results
        qaoa_colors = ['green' if x > 0.5 else 'lightgray' for x in result.x]
        fig.add_trace(
            go.Bar(x=stocks, y=result.x, name='QAOA',
                   marker_color=qaoa_colors, showlegend=False),
            row=1, col=1
        )

        # Classical results
        fig.add_trace(
            go.Bar(x=stocks, y=classical_result.x, name='Classical',
                   marker_color='blue', showlegend=False),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(tickangle=45)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Risk-Return Profile")

        # Risk-Return scatter
        colors = ['green' if x > 0.5 else 'red' for x in result.x]
        sizes = [15 if x > 0.5 else 8 for x in result.x]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=annual_volatility,
            y=annual_returns,
            mode='markers+text',
            marker=dict(color=colors, size=sizes, opacity=0.7),
            text=stocks,
            textposition="top center",
            name='Assets',
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title="Annual Volatility",
            yaxis_title="Annual Expected Return",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Covariance matrix heatmap
    st.subheader("🔥 Asset Correlation Matrix")

    correlation_matrix = sigma / np.outer(np.sqrt(np.diag(sigma)), np.sqrt(np.diag(sigma)))

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=stocks,
        y=stocks,
        colorscale='RdBu',
        zmid=0,
        text=np.around(correlation_matrix, decimals=2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Price history (if available)
    if prices is not None and len(prices) > 0:
        st.subheader("📈 Price History")

        # Normalize prices to start at 100
        normalized_prices = (prices / prices.iloc[0]) * 100

        fig = go.Figure()

        for stock in stocks:
            if stock in normalized_prices.columns:
                fig.add_trace(go.Scatter(
                    x=normalized_prices.index,
                    y=normalized_prices[stock],
                    mode='lines',
                    name=stock,
                    line=dict(width=2)
                ))

        fig.update_layout(
            title="Normalized Stock Prices (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Sidebar Configuration
# -------------------------
st.sidebar.header("🔧 Portfolio Configuration")

# Optimization type selection
optimization_type = st.sidebar.radio(
    "Choose Optimization Method:",
    ["Classical Optimization", "Quantum Optimization (QAOA)", "Both Methods"]
)

if optimization_type == "Quantum Optimization (QAOA)" and not QUANTUM_AVAILABLE:
    st.sidebar.error("⚠️ Qiskit libraries not installed. Switching to classical optimization.")
    optimization_type = "Classical Optimization"

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

# Common parameters
risk_free_rate = st.sidebar.slider(
    "Risk-free Rate (%)",
    min_value=0.0, max_value=10.0, value=2.0, step=0.1
) / 100

# Classical-specific parameters
if optimization_type in ["Classical Optimization", "Both Methods"]:
    st.sidebar.subheader("🏛️ Classical Parameters")
    classical_method = st.sidebar.selectbox(
        "Classical Strategy:",
        ["Maximum Sharpe Ratio", "Minimum Volatility", "Maximum Return"]
    )

# Quantum-specific parameters
if optimization_type in ["Quantum Optimization (QAOA)", "Both Methods"]:
    st.sidebar.subheader("🔬 Quantum Parameters")

    quantum_risk_factor = st.sidebar.slider(
        "Quantum Risk Factor",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Higher values = more risk averse"
    )

    budget_ratio = st.sidebar.slider(
        "Budget Ratio",
        min_value=0.1,
        max_value=0.8,
        value=0.4,
        step=0.1,
        help="Fraction of assets to select"
    )

    qaoa_reps = st.sidebar.selectbox(
        "QAOA Layers (p)",
        options=[1, 2, 3],
        index=0,
        help="Number of QAOA repetitions"
    )

    max_iterations = st.sidebar.selectbox(
        "Max Iterations",
        options=[50, 100, 200],
        index=1,
        help="Maximum optimizer iterations"
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
    with st.spinner("Running optimization..."):
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
        st.session_state['optimization_type'] = optimization_type

# Display results if optimization has been run
if st.session_state.get('optimization_run', False):
    data = st.session_state['data']
    returns = st.session_state['returns']
    mean_returns = st.session_state['mean_returns']
    cov_matrix = st.session_state['cov_matrix']
    valid_stocks = st.session_state['valid_stocks']
    current_optimization_type = st.session_state.get('optimization_type', optimization_type)

    # Create tabs for different optimization methods
    if current_optimization_type == "Both Methods":
        tab1, tab2, tab3 = st.tabs(["📊 Classical Results", "🚀 Quantum Results", "⚖️ Comparison"])

        with tab1:
            # Classical optimization
            optimal_weights = optimize_portfolio_classical(mean_returns, cov_matrix, risk_free_rate, classical_method)
            opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix,
                                                                           risk_free_rate)

            create_classical_visualizations(data, returns, mean_returns, cov_matrix, valid_stocks,
                                            optimal_weights, opt_return, opt_volatility, opt_sharpe,
                                            risk_free_rate, classical_method)

        with tab2:
            # Quantum optimization
            if QUANTUM_AVAILABLE:
                with st.spinner("Running QAOA optimization..."):
                    qaoa_result, qaoa_classical, mu, sigma, portfolio_obj, prices = run_qaoa_optimization(
                        valid_stocks, start_date, end_date, quantum_risk_factor,
                        budget_ratio, max_iterations, qaoa_reps
                    )

                if qaoa_result is not None and qaoa_classical is not None:
                    create_quantum_visualizations(qaoa_result, qaoa_classical, valid_stocks, mu, sigma, prices)
                else:
                    st.error("❌ Quantum optimization failed. Please try different parameters.")
            else:
                st.error("Qiskit libraries not available for quantum optimization.")

        with tab3:
            # Comparison between methods
            st.header("⚖️ Classical vs Quantum Comparison")

            if QUANTUM_AVAILABLE:
                # Run both optimizations for comparison
                with st.spinner("Comparing optimization methods..."):
                    # Classical
                    classical_weights = optimize_portfolio_classical(mean_returns, cov_matrix, risk_free_rate,
                                                                     classical_method)
                    classical_return, classical_vol, classical_sharpe = portfolio_performance(classical_weights,
                                                                                              mean_returns, cov_matrix,
                                                                                              risk_free_rate)

                    # Quantum
                    qaoa_result, qaoa_classical, mu, sigma, portfolio_obj, prices = run_qaoa_optimization(
                        valid_stocks, start_date, end_date, quantum_risk_factor,
                        budget_ratio, max_iterations, qaoa_reps
                    )

                if qaoa_result is not None:
                    # Selected assets from quantum
                    quantum_selected = [stock for i, stock in enumerate(valid_stocks) if qaoa_result.x[i] > 0.5]

                    # Comparison metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🏛️ Classical Method")
                        st.metric("Expected Return", f"{classical_return:.2%}")
                        st.metric("Volatility", f"{classical_vol:.2%}")
                        st.metric("Sharpe Ratio", f"{classical_sharpe:.2f}")
                        st.metric("Assets Used", f"{len([w for w in classical_weights if w > 0.01])}")

                        # Top holdings
                        top_classical = sorted(zip(valid_stocks, classical_weights),
                                               key=lambda x: x[1], reverse=True)[:5]
                        st.write("**Top Holdings:**")
                        for asset, weight in top_classical:
                            if weight > 0.01:
                                st.write(f"• {asset}: {weight:.1%}")

                    with col2:
                        st.subheader("🚀 Quantum Method")
                        if len(quantum_selected) > 0:
                            # Calculate quantum portfolio metrics
                            quantum_indices = [i for i, x in enumerate(qaoa_result.x) if x > 0.5]
                            annual_returns = mean_returns * 252
                            annual_volatility = returns.std() * np.sqrt(252)

                            quantum_return = np.mean([annual_returns.iloc[i] for i in quantum_indices])
                            quantum_vol = np.mean([annual_volatility.iloc[i] for i in quantum_indices])
                            quantum_sharpe = quantum_return / quantum_vol if quantum_vol > 0 else 0

                            st.metric("Portfolio Return", f"{quantum_return:.2%}")
                            st.metric("Portfolio Volatility", f"{quantum_vol:.2%}")
                            st.metric("Sharpe Ratio", f"{quantum_sharpe:.2f}")
                            st.metric("Assets Selected", f"{len(quantum_selected)}")

                            st.write("**Selected Assets:**")
                            for asset in quantum_selected:
                                st.write(f"• {asset}")
                        else:
                            st.warning("No assets selected by quantum method")

                    # Method comparison chart
                    st.subheader("📊 Method Comparison")

                    if len(quantum_selected) > 0:
                        comparison_data = {
                            'Method': ['Classical', 'Quantum'],
                            'Expected Return': [classical_return, quantum_return],
                            'Volatility': [classical_vol, quantum_vol],
                            'Sharpe Ratio': [classical_sharpe, quantum_sharpe],
                            'Assets Used': [len([w for w in classical_weights if w > 0.01]), len(quantum_selected)]
                        }

                        comparison_df = pd.DataFrame(comparison_data)

                        # Display comparison table
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                        # Bar chart comparison
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Expected Return', 'Volatility', 'Sharpe Ratio', 'Assets Used'),
                            specs=[[{"type": "bar"}, {"type": "bar"}],
                                   [{"type": "bar"}, {"type": "bar"}]]
                        )

                        methods = ['Classical', 'Quantum']
                        colors = ['blue', 'red']

                        fig.add_trace(go.Bar(x=methods, y=[classical_return, quantum_return],
                                             marker_color=colors, showlegend=False), row=1, col=1)
                        fig.add_trace(go.Bar(x=methods, y=[classical_vol, quantum_vol],
                                             marker_color=colors, showlegend=False), row=1, col=2)
                        fig.add_trace(go.Bar(x=methods, y=[classical_sharpe, quantum_sharpe],
                                             marker_color=colors, showlegend=False), row=2, col=1)
                        fig.add_trace(go.Bar(x=methods,
                                             y=[len([w for w in classical_weights if w > 0.01]), len(quantum_selected)],
                                             marker_color=colors, showlegend=False), row=2, col=2)

                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Quantum comparison not available without Qiskit libraries.")

    elif current_optimization_type == "Classical Optimization":
        # Classical optimization only
        optimal_weights = optimize_portfolio_classical(mean_returns, cov_matrix, risk_free_rate, classical_method)
        opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix,
                                                                       risk_free_rate)

        create_classical_visualizations(data, returns, mean_returns, cov_matrix, valid_stocks,
                                        optimal_weights, opt_return, opt_volatility, opt_sharpe,
                                        risk_free_rate, classical_method)

    elif current_optimization_type == "Quantum Optimization (QAOA)":
        # Quantum optimization only
        if QUANTUM_AVAILABLE:
            with st.spinner("Running QAOA optimization..."):
                qaoa_result, qaoa_classical, mu, sigma, portfolio_obj, prices = run_qaoa_optimization(
                    valid_stocks, start_date, end_date, quantum_risk_factor,
                    budget_ratio, max_iterations, qaoa_reps
                )

            if qaoa_result is not None and qaoa_classical is not None:
                create_quantum_visualizations(qaoa_result, qaoa_classical, valid_stocks, mu, sigma, prices)
            else:
                st.error("❌ Quantum optimization failed. Please try different parameters.")
        else:
            st.error("Qiskit libraries not available for quantum optimization.")

else:
    # Show initial information
    st.info("👈 Configure your portfolio parameters in the sidebar and click 'Run Optimization' to get started!")

    # Create tabs for information
    info_tab1, info_tab2, info_tab3 = st.tabs(["🎯 How to Use", "🏛️ Classical Methods", "🚀 Quantum Methods"])

    with info_tab1:
        st.subheader("🎯 How to Use This Tool")
        st.markdown("""
        1. **Select Optimization Type**: Choose between Classical, Quantum, or Both methods
        2. **Select Stocks**: Choose 2-10 stocks from the dropdown
        3. **Set Date Range**: Pick your historical data period
        4. **Configure Parameters**: 
           - Risk-free rate (current treasury bill rate)
           - Classical strategy (Sharpe, Volatility, Return)
           - Quantum parameters (risk factor, budget ratio, QAOA layers)
        5. **Run Optimization**: Click the button to generate results

        **Features:**
        - Multiple optimization algorithms (Classical MPT + Quantum QAOA)
        - Interactive visualizations and comparisons
        - Risk metrics and correlation analysis
        - Historical performance charts
        - Side-by-side method comparison
        """)

    with info_tab2:
        st.subheader("🏛️ Classical Portfolio Optimization")
        st.markdown("""
        **Modern Portfolio Theory (MPT)** - Developed by Harry Markowitz

        **Methods Available:**
        - **Maximum Sharpe Ratio**: Optimal risk-adjusted returns
        - **Minimum Volatility**: Lowest risk portfolio  
        - **Maximum Return**: Highest expected returns

        **Key Features:**
        - Efficient frontier visualization
        - Continuous weight allocation (0-100% per asset)
        - Risk-return optimization based on historical data
        - Correlation analysis and diversification benefits

        **Outputs:**
        - Portfolio weights for each asset
        - Expected return, volatility, and Sharpe ratio
        - Value at Risk (VaR) and Maximum Drawdown
        - Interactive efficient frontier chart
        """)

    with info_tab3:
        st.subheader("🚀 Quantum Portfolio Optimization (QAOA)")
        if QUANTUM_AVAILABLE:
            st.markdown("""
            **Quantum Approximate Optimization Algorithm (QAOA)** - Quantum combinatorial optimization

            **How QAOA Works:**
            1. **Problem Formulation**: Portfolio selection as QUBO problem
            2. **Quantum Circuit**: Parameterized circuit with cost and mixer layers
            3. **Classical Optimization**: Find optimal circuit parameters  
            4. **Measurement**: Extract binary portfolio selection

            **Key Features:**
            - Binary asset selection (include/exclude each stock)
            - Budget constraints (select subset of available assets)
            - Risk-return tradeoff optimization
            - Quantum advantage for combinatorial problems

            **Parameters:**
            - **Risk Factor**: Higher = more risk averse
            - **Budget Ratio**: Fraction of assets to select
            - **QAOA Layers (p)**: More layers = better solutions (but slower)
            - **Max Iterations**: Optimization convergence limit

            **Outputs:**
            - Binary selection of assets (selected/not selected)
            - Comparison with classical equivalent problem
            - Risk-return profile of selected portfolio
            """)
        else:
            st.error("""
            **⚠️ Quantum optimization not available**

            To enable quantum portfolio optimization, install the required packages:
            ```bash
            pip install qiskit qiskit-aer qiskit-algorithms qiskit-finance qiskit-optimization
            ```

            **Qiskit Packages Required:**
            - `qiskit`: Core quantum computing framework
            - `qiskit-aer`: Quantum simulators
            - `qiskit-algorithms`: Quantum algorithms including QAOA  
            - `qiskit-finance`: Financial applications
            - `qiskit-optimization`: Optimization problems
            """)

    with st.expander("⚠️ Important Disclaimers"):
        st.warning("""
        - **Past performance doesn't guarantee future results**
        - This tool is for educational and research purposes only
        - Consider transaction costs, taxes, and market impact in real investing
        - Consult a financial advisor for personalized investment advice
        - Market conditions and correlations can change rapidly
        - Quantum optimization is experimental and may not always outperform classical methods
        - Results may vary between runs due to quantum measurement randomness
        """)

    # Show system status
    with st.expander("🔧 System Status"):
        st.write("**Available Optimization Methods:**")
        st.write("✅ Classical Portfolio Optimization (Always available)")
        if QUANTUM_AVAILABLE:
            st.write("✅ Quantum QAOA Optimization (Qiskit installed)")
        else:
            st.write("❌ Quantum QAOA Optimization (Qiskit not installed)")

        st.write(f"**Selected Stocks:** {len(selected_stocks)}")
        st.write(f"**Date Range:** {start_date} to {end_date}")
        st.write(f"**Optimization Type:** {optimization_type}")