import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

# Set a seed for reproducibility
np.random.seed(42)

# --- Data Generation ---
@st.cache_data
def generate_placeholder_data(num_days=252):
    """Generates placeholder time series data for financial returns."""
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq='B') # Business days
    # Simulate returns with some mean and volatility
    mean_return = 0.0005
    volatility = 0.015
    returns = np.random.normal(mean_return, volatility, num_days)
    data = pd.DataFrame({'Date': dates, 'Return': returns})
    data.set_index('Date', inplace=True)
    return data

# --- Main Application ---
def run_app():
    st.set_page_config(layout="wide")
    st.title("Financial Asset Return Analysis")

    # --- Data Loading ---
    num_days_data = 252 * 3 # Approximately 3 years of data
    data = generate_placeholder_data(num_days_data)

    # --- Tabbed Display ---
    tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "Histogram", "QQ Plot"])

    # --- Tab 1: Time Series Analysis ---
    with tab1:
        st.header("Time Series of Returns with Confidence Bounds")

        # --- User Inputs for Confidence Interval ---
        st.sidebar.header("Confidence Interval Settings")
        volatility_option = st.sidebar.radio(
            "Choose volatility calculation method for confidence interval:",
            ("Entire Dataset", "Rolling Window"),
            index=0  # Default to "Entire Dataset"
        )

        rolling_window_days = None
        if volatility_option == "Rolling Window":
            rolling_window_days = st.sidebar.number_input(
                "Enter window size for rolling volatility (days):",
                min_value=5,
                max_value=len(data)-1,
                value=30, # Default rolling window
                step=1
            )

        confidence_level_pct = st.sidebar.slider(
            "Select Confidence Level (%) for bounds:",
            min_value=50.0,
            max_value=99.9,
            value=95.0, # Default confidence level
            step=0.1
        )
        confidence_level = confidence_level_pct / 100.0
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2) # Two-tailed z-score

        # --- Calculate Volatility and Confidence Bounds ---
        data_analysis = data.copy()

        if volatility_option == "Entire Dataset" or rolling_window_days is None:
            st.subheader(f"Confidence Bounds based on Volatility of Entire Dataset ({confidence_level_pct}%)")
            overall_std = data_analysis['Return'].std()
            overall_mean = data_analysis['Return'].mean() # Using overall mean for simplicity, could also use 0 or a rolling mean
            data_analysis['Volatility'] = overall_std
            data_analysis['Upper_Bound'] = overall_mean + z_score * overall_std
            data_analysis['Lower_Bound'] = overall_mean - z_score * overall_std
            display_volatility = overall_std
        else:
            st.subheader(f"Confidence Bounds based on {rolling_window_days}-Day Rolling Volatility ({confidence_level_pct}%)")
            if rolling_window_days >= len(data_analysis):
                st.warning(f"Rolling window ({rolling_window_days}) cannot be larger than the dataset size ({len(data_analysis)}). Using entire dataset.")
                overall_std = data_analysis['Return'].std()
                overall_mean = data_analysis['Return'].mean()
                data_analysis['Volatility'] = overall_std
                data_analysis['Upper_Bound'] = overall_mean + z_score * overall_std
                data_analysis['Lower_Bound'] = overall_mean - z_score * overall_std
                display_volatility = overall_std
            else:
                data_analysis['Volatility'] = data_analysis['Return'].rolling(window=rolling_window_days).std()
                # For bounds, we can use a rolling mean or the overall mean. Let's use overall mean for consistency in this example.
                # A rolling mean might make the bands adapt more, but could also be noisier.
                overall_mean_for_rolling = data_analysis['Return'].mean() # Or use .rolling(window=rolling_window_days).mean()
                data_analysis['Upper_Bound'] = overall_mean_for_rolling + z_score * data_analysis['Volatility']
                data_analysis['Lower_Bound'] = overall_mean_for_rolling - z_score * data_analysis['Volatility']
                display_volatility = data_analysis['Volatility'].iloc[-1] if not data_analysis['Volatility'].empty else np.nan


        # --- Plot Time Series with Confidence Bounds ---
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=data_analysis.index, y=data_analysis['Return'], mode='lines', name='Daily Return', line=dict(color='blue')))
        fig_ts.add_trace(go.Scatter(x=data_analysis.index, y=data_analysis['Upper_Bound'], mode='lines', name='Upper Confidence Bound', line=dict(color='rgba(255,0,0,0.5)', dash='dash')))
        fig_ts.add_trace(go.Scatter(x=data_analysis.index, y=data_analysis['Lower_Bound'], mode='lines', name='Lower Confidence Bound', line=dict(color='rgba(255,0,0,0.5)', dash='dash')))

        fig_ts.update_layout(
            title=f'Time Series of Returns with {confidence_level_pct}% Confidence Bounds',
            xaxis_title='Date',
            yaxis_title='Return',
            legend_title="Legend"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        if pd.notna(display_volatility):
             st.write(f"**Volatility used for current bounds:** {display_volatility:.4f}")
        else:
             st.write(f"**Volatility used for current bounds:** Not enough data for rolling window calculation.")


        # --- Analysis of Observations Outside Confidence Interval ---
        st.markdown("---")
        st.subheader("Analysis of Observations vs. Confidence Interval")

        # Ensure bounds are calculated for all points where returns exist
        valid_bounds_data = data_analysis.dropna(subset=['Return', 'Upper_Bound', 'Lower_Bound'])

        obs_above_upper = valid_bounds_data[valid_bounds_data['Return'] > valid_bounds_data['Upper_Bound']]
        obs_below_lower = valid_bounds_data[valid_bounds_data['Return'] < valid_bounds_data['Lower_Bound']]
        num_obs_outside = len(obs_above_upper) + len(obs_below_lower)
        total_valid_obs = len(valid_bounds_data)

        st.write(f"**Number of observations higher than the upper confidence bound:** {len(obs_above_upper)}")
        st.write(f"**Number of observations lower than the lower confidence bound:** {len(obs_below_lower)}")
        st.write(f"**Total number of observations outside the confidence interval:** {num_obs_outside}")

        if total_valid_obs > 0:
            percentage_outside_actual = (num_obs_outside / total_valid_obs) * 100
            st.write(f"**Percentage of observations outside the interval (Actual):** {percentage_outside_actual:.2f}%")
        else:
            st.write("**Percentage of observations outside the interval (Actual):** N/A (no valid observations with bounds)")


        # Expected number/percentage outside
        expected_percentage_outside = (1 - confidence_level) * 100
        expected_num_outside = (1 - confidence_level) * total_valid_obs

        st.write(f"**Expected percentage of observations outside the interval (Normal Dist.):** {expected_percentage_outside:.2f}%")
        st.write(f"**Expected number of observations outside the interval (Normal Dist.):** {expected_num_outside:.2f} (out of {total_valid_obs} valid observations)")

        if total_valid_obs > 0:
            st.markdown(f"""
            Based on the {confidence_level_pct}% confidence level and the assumption of a normal distribution,
            we would expect approximately **{expected_percentage_outside:.2f}%** of the observations
            (or **{expected_num_outside:.2f}** out of {total_valid_obs}) to fall outside the confidence bounds.
            The actual number of observations outside the bounds is **{num_obs_outside}**.
            """)
        else:
            st.markdown("Not enough data points with valid bounds to perform this comparison.")


    # --- Tab 2: Histogram ---
    with tab2:
        st.header("Histogram of Returns")

        num_bins = st.slider("Select number of bins for histogram:", min_value=10, max_value=100, value=30, key="hist_bins")

        fig_hist = go.Figure(data=[go.Histogram(x=data['Return'], nbinsx=num_bins)])
        fig_hist.update_layout(
            title='Histogram of Daily Returns',
            xaxis_title='Return',
            yaxis_title='Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.write(f"**Skewness:** {data['Return'].skew():.4f}")
        st.write(f"**Kurtosis:** {data['Return'].kurtosis():.4f} (Excess kurtosis: Kurtosis - 3)")


    # --- Tab 3: QQ Plot ---
    with tab3:
        st.header("QQ Plot of Returns")
        st.write("This plot compares the quantiles of the return data to the quantiles of a standard normal distribution.")

        # Create QQ plot data
        qq_data = stats.probplot(data['Return'].dropna(), dist="norm") # Using dropna() for safety

        # Theoretical quantiles
        theoretical_quantiles = qq_data[0][0]
        # Ordered values (sample quantiles)
        ordered_values = qq_data[0][1]

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_quantiles, y=ordered_values, mode='markers', name='Data Quantiles'))
        fig_qq.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines', name='Normal Quantiles (Line)', line=dict(color='red'))) # y=x line

        fig_qq.update_layout(
            title='QQ Plot: Returns vs. Normal Distribution',
            xaxis_title='Theoretical Quantiles (Normal Distribution)',
            yaxis_title='Sample Quantiles (Returns Data)'
        )
        st.plotly_chart(fig_qq, use_container_width=True)

        st.markdown("""
        **Interpreting the QQ Plot:**
        - If the points lie approximately on the straight red line, the data is likely normally distributed.
        - Deviations from the line suggest departures from normality. For example:
            - **S-shaped curve:** Indicates data with lighter or heavier tails than normal.
            - **Points systematically above or below the line:** Indicates skewness.
        """)

if __name__ == "__main__":
    run_app()