import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import sqlalchemy

# ─── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data
def generate_placeholder_data(num_days: int = 252) -> pd.DataFrame:
    """Simulate business-day returns for num_days."""
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="B")
    returns = np.random.default_rng(42).normal(0.0005, 0.015, size=num_days)
    df = pd.DataFrame({"Date": dates, "Return": returns}).set_index("Date")
    return df

@st.cache_data(ttl=600)
def load_data_from_db(conn_str: str, query: str) -> pd.DataFrame:
    """Run SQL and return a DataFrame indexed by Date."""
    engine = sqlalchemy.create_engine(conn_str, future=True)
    with engine.connect() as conn:
        df = pd.read_sql_query(sqlalchemy.text(query), conn, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df

# ─── Main app ─────────────────────────────────────────────────────────────────

def run_app():
    st.set_page_config(layout="wide")
    st.title("Financial Asset Return Analysis")

    # ─── Sidebar: Data Source ────────────────────────────────────────────────
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Simulated", "Database"])

    if data_source == "Simulated":
        num_days = st.sidebar.number_input(
            "Days of business‑day data:", min_value=30, max_value=2000,
            value=252*3, step=1
        )
        data = generate_placeholder_data(int(num_days))
    else:
        st.sidebar.markdown("**Enter your DB connection & SQL:**")
        conn_str = st.sidebar.text_input(
            "Connection string:", "sqlite:///test.db"
        )
        query = st.sidebar.text_area(
            "SQL query:", "SELECT Date, Return FROM returns"
        )
        data = load_data_from_db(conn_str, query)

    # ─── Sidebar: CI Settings ────────────────────────────────────────────────
    st.sidebar.header("Confidence Interval Settings")
    vol_method = st.sidebar.radio("Volatility for CI:", ["Entire Dataset", "Rolling Window"])
    rolling_window = None
    if vol_method == "Rolling Window":
        rolling_window = st.sidebar.number_input(
            "Rolling window (days):", min_value=5,
            max_value=len(data)-1, value=30, step=1
        )
    conf_pct = st.sidebar.slider("CI Confidence Level (%)", 50.0, 99.9, 95.0, 0.1)
    z = stats.norm.ppf(1 - (1 - conf_pct/100)/2)

    # ─── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Time Series", "Histogram", "QQ Plot"])

    # --- Tab 1: Time Series with CI ---
    with tab1:
        st.header("Returns with Confidence Bounds")
        df = data.copy()
        mean_ret = df["Return"].mean()
        if vol_method == "Rolling Window" and rolling_window:
            df["Vol"] = df["Return"].rolling(rolling_window).std()
        else:
            df["Vol"] = df["Return"].std()
        df["Upper"] = mean_ret + z * df["Vol"]
        df["Lower"] = mean_ret - z * df["Vol"]

        fig = go.Figure([
            go.Scatter(x=df.index, y=df["Return"], name="Return", mode="lines"),
            go.Scatter(x=df.index, y=df["Upper"], name="Upper CI", mode="lines", line=dict(dash="dash")),
            go.Scatter(x=df.index, y=df["Lower"], name="Lower CI", mode="lines", line=dict(dash="dash")),
        ])
        fig.update_layout(xaxis_title="Date", yaxis_title="Return")
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: Histogram ---
    with tab2:
        st.header("Histogram of Returns")
        bins = st.slider("Number of bins:", 10, 100, 30, key="hist_bins")
        fig = go.Figure([go.Histogram(x=data["Return"], nbinsx=bins)])
        fig.update_layout(xaxis_title="Return", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Skewness: {data['Return'].skew():.4f}")
        st.write(f"Kurtosis: {data['Return'].kurtosis():.4f}")

    # --- Tab 3: QQ Plot (post–rolling warm‑up) ---
    with tab3:
        st.header("QQ Plot of Returns")
        st.write("Only observations after the rolling‑window warm‑up are shown.")
        if vol_method == "Rolling Window" and rolling_window:
            sample = data["Return"].iloc[int(rolling_window):].dropna()
        else:
            sample = data["Return"].dropna()

        qq = stats.probplot(sample, dist="norm")
        theo_q, samp_q = qq[0][0], qq[0][1]
        fig = go.Figure([
            go.Scatter(x=theo_q, y=samp_q, mode="markers", name="Data"),
            go.Scatter(x=theo_q, y=theo_q, mode="lines", name="45°", line=dict(color="red")),
        ])
        fig.update_layout(xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run_app()
