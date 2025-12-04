import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use("seaborn-v0_8-dark")
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['grid.color'] = "#575757"
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams.update({"axes.grid" : True})

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = "#ffffff"

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = "#262731"

# Page configuration
st.set_page_config(page_title="SPAC Monte Carlo Dashboard", layout="wide")

# Main function from original code
def founderShareValue(pShare, dLockup, pSuccess, dDilution):
    return pShare * (1 - dLockup) * pSuccess * (1 - dDilution)

# Distribution generators
def generate_parameter(dist_type, params, size):
    if dist_type == "Fixed":
        return np.full(size, params['value'])
    elif dist_type == "Uniform":
        return np.random.uniform(params['min'], params['max'], size)
    elif dist_type == "Normal":
        return np.random.normal(params['mean'], params['std'], size)
    elif dist_type == "Log-Normal":
        log_std_dev = np.sqrt(np.log(1 + params['volatility'] ** 2))
        log_mean = np.log(params['mean']) - 0.5 * log_std_dev ** 2
        return np.random.lognormal(log_mean, log_std_dev, size)

# Modified Monte Carlo function
def monteCarlo_advanced(share_params, lockup_params, success_params, dilution_params, numberOfSimulations):
    # Parameter generation
    pShare = generate_parameter(share_params['type'], share_params, numberOfSimulations)
    dLockup = generate_parameter(lockup_params['type'], lockup_params, numberOfSimulations)
    dDilution = generate_parameter(dilution_params['type'], dilution_params, numberOfSimulations)
    
    # For success probability generate probabilities then binomial results
    if success_params['type'] == "Fixed":
        pSuccessProbability = np.full(numberOfSimulations, success_params['value'])
    else:
        pSuccessProbability = generate_parameter(success_params['type'], success_params, numberOfSimulations)
    
    pSuccess = np.random.binomial(n=1, p=pSuccessProbability)
    
    # Calculate founder value
    founderValue = founderShareValue(pShare, dLockup, pSuccess, dDilution)

    simulationResults = pd.DataFrame({
        'pShare': pShare,
        'dLockup': dLockup,
        'pSuccessProbability': pSuccessProbability,
        'pSuccess': pSuccess,
        'dDilution': dDilution,
        'founderShareValue': founderValue
    })

    return simulationResults

# Function to create plots (exactly as in original code)
def create_distribution_plots(simulationData):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    cols = ['pShare', 'dLockup', 'pSuccessProbability', 'dDilution', 'pSuccess']
    colors = ['dodgerblue', 'limegreen', 'violet', 'orange', 'red']

    for ax, col, color in zip(axes.ravel(), cols, colors):
        sns.histplot(
            data=simulationData,
            x=col,
            bins=50,
            kde=False,
            color=color,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.75,
            ax=ax,
            zorder=2,
            stat="probability"
        )
        ax.set_title(f'{col}', fontsize=14)
        ax.grid(True)

    plt.tight_layout(h_pad=2, w_pad=2)
    return fig

def create_founder_value_plot(simulationData):
    fig, ax = plt.subplots(figsize=(20, 7))
    sns.histplot(simulationData['founderShareValue'], bins=50, color="#007E76", edgecolor = "white",
                alpha=0.75, zorder=2, linewidth=0.5, stat='probability', ax=ax)
    ax.axvline(simulationData['founderShareValue'].mean(), color='red', 
              linestyle='--', label="mean")
    ax.axvline(simulationData['founderShareValue'].median(), color='limegreen', 
              linestyle='--', label="median")
    # ax.set_title("Histogram of Founder Share Value (INCLUDING failure cases)", fontsize=16)
    ax.legend(fontsize=14, frameon=True)
    # ax.grid(color="lightgray", linewidth=0.5)
    return fig

# Streamlit interface
st.title("SPAC Monte Carlo Simulation Dashboard")

# Sidebar with parameters
st.sidebar.header("Simulation Parameters")

# Number of simulations
numberOfSimulations = st.sidebar.slider("Number of Simulations", 10000, 500000, 200000, 10000)

# Share Price parameters
st.sidebar.subheader("Share Price (pShare)")
share_dist = st.sidebar.selectbox("Distribution Type", ["Fixed", "Uniform", "Normal", "Log-Normal"], key="share")

share_params = {'type': share_dist}
if share_dist == "Fixed":
    if "share_fixed" not in st.session_state:
        st.session_state.share_fixed = 10.0

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.slider(
            "Fixed Value", 1.0, 50.0, step=0.01,
            key="share_fixed" 
        )
    with col2:
        st.number_input(
            " ", 1.0, 50.0, step=0.01,
            key="share_fixed"
        )

    share_params["value"] = st.session_state.share_fixed


elif share_dist == "Uniform":
    share_params['min'] = st.sidebar.slider("Min Value", 1.0, 20.0, 8.0, 0.1)
    share_params['max'] = st.sidebar.slider("Max Value", 5.0, 50.0, 12.0, 0.1)
elif share_dist == "Normal":
    share_params['mean'] = st.sidebar.slider("Mean", 1.0, 50.0, 10.0, 0.1)
    share_params['std'] = st.sidebar.slider("Std Dev", 0.1, 10.0, 1.0, 0.1)
elif share_dist == "Log-Normal":
    share_params['mean'] = st.sidebar.slider("Mean", 1.0, 50.0, 10.0, 0.1)
    share_params['volatility'] = st.sidebar.slider("Volatility", 0.01, 1.0, 0.1, 0.01)

# Lockup parameters
st.sidebar.subheader("Lockup Discount (dLockup)")
lockup_dist = st.sidebar.selectbox("Distribution Type", ["Fixed", "Uniform", "Normal"], key="lockup")

lockup_params = {'type': lockup_dist}
if lockup_dist == "Fixed":
    lockup_params['value'] = st.sidebar.slider("Fixed Value", 0.0, 1.0, 0.15, 0.01)
elif lockup_dist == "Uniform":
    lockup_params['min'] = st.sidebar.slider("Min Value", 0.0, 0.5, 0.25, 0.01, key="lockup_min")
    lockup_params['max'] = st.sidebar.slider("Max Value", 0.0, 1.0, 0.275, 0.001, key="lockup_max")
elif lockup_dist == "Normal":
    lockup_params['mean'] = st.sidebar.slider("Mean", 0.0, 1.0, 0.15, 0.01, key="lockup_mean")
    lockup_params['std'] = st.sidebar.slider("Std Dev", 0.01, 0.5, 0.05, 0.01, key="lockup_std")

# Success Probability parameters
st.sidebar.subheader("Success Probability (pSuccessProbability)")
success_dist = st.sidebar.selectbox("Distribution Type", ["Fixed", "Uniform", "Normal"], key="success")

success_params = {'type': success_dist}
if success_dist == "Fixed":
    success_params['value'] = st.sidebar.slider("Fixed Value", 0.0, 1.0, 0.75, 0.01, key="success_fixed")
elif success_dist == "Uniform":
    success_params['min'] = st.sidebar.slider("Min Value", 0.0, 1.0, 0.75, 0.01, key="success_min")
    success_params['max'] = st.sidebar.slider("Max Value", 0.0, 1.0, 0.80, 0.01, key="success_max")
elif success_dist == "Normal":
    success_params['mean'] = st.sidebar.slider("Mean", 0.0, 1.0, 0.775, 0.01, key="success_mean")
    success_params['std'] = st.sidebar.slider("Std Dev", 0.01, 0.3, 0.05, 0.01, key="success_std")

# Dilution parameters
st.sidebar.subheader("Dilution (dDilution)")
dilution_dist = st.sidebar.selectbox("Distribution Type", ["Fixed", "Uniform", "Normal"], key="dilution")

dilution_params = {'type': dilution_dist}
if dilution_dist == "Fixed":
    dilution_params['value'] = st.sidebar.slider("Fixed Value", 0.0, 1.0, 0.0, 0.01, key="dilution_fixed")
elif dilution_dist == "Uniform":
    dilution_params['min'] = st.sidebar.slider("Min Value", 0.0, 0.5, 0.0, 0.01, key="dilution_min")
    dilution_params['max'] = st.sidebar.slider("Max Value", 0.0, 1.0, 0.10, 0.01, key="dilution_max")
elif dilution_dist == "Normal":
    dilution_params['mean'] = st.sidebar.slider("Mean", 0.0, 1.0, 0.05, 0.01, key="dilution_mean")
    dilution_params['std'] = st.sidebar.slider("Std Dev", 0.01, 0.3, 0.02, 0.01, key="dilution_std")

# Simulation run button
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        # Run simulation
        simulationData = monteCarlo_advanced(
            share_params, lockup_params, success_params, dilution_params, numberOfSimulations
        )
        
        # Save to session state
        st.session_state.simulationData = simulationData

# Display results if data exists
if 'simulationData' in st.session_state:
    simulationData = st.session_state.simulationData
    
    # Statistics
    st.header("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    stats = simulationData['founderShareValue']
    with col1:
        st.metric("Mean", f"${stats.mean():.6f}")
        st.metric("Min", f"${stats.min():.6f}")
    with col2:
        st.metric("Median", f"${stats.median():.6f}")
        st.metric("Max", f"${stats.max():.6f}")
    with col3:
        st.metric("Std Dev", f"${stats.std():.6f}")
        st.metric("5% Percentile", f"${stats.quantile(0.05):.6f}")
    with col4:
        st.metric("95% Percentile", f"${stats.quantile(0.95):.6f}")
    
    # Parameter distribution plots
    st.header("Parameter Distributions")
    fig1 = create_distribution_plots(simulationData)
    st.pyplot(fig1)
    
    # Founder Share Value plot
    st.header("Founder Share Value Distribution (INCLUDING failure cases)")
    fig2 = create_founder_value_plot(simulationData)
    st.pyplot(fig2)
    
    # Limited range (as in original code)
    st.header("Founder Share Value Distribution (EXCLUDING failure cases)")
    fig3, ax = plt.subplots(figsize=(20, 7))
    withoutFailures = simulationData['founderShareValue'][simulationData['founderShareValue'] != 0.0]
    sns.histplot(simulationData['founderShareValue'], bins=50, color="#007E76", edgecolor = "white",
                alpha=0.75, zorder=2, binrange=[withoutFailures.min(), withoutFailures.max()], linewidth=0.5, stat='probability', ax=ax)
    ax.axvline(simulationData['founderShareValue'].mean(), color='red', linestyle='--', label="mean")
    ax.axvline(simulationData['founderShareValue'].median(), color='limegreen', linestyle='--', label="median")
    # ax.set_title("Histogram of Founder Share Value (EXCLUDING failure cases)", fontsize=16)
    ax.legend(fontsize=14, frameon=True)
    # ax.grid(color="lightgray", linewidth=0.5)
    st.pyplot(fig3)
    
    # Data table
    with st.expander("Raw Simulation Data (first 1000 rows)"):
        st.dataframe(simulationData.head(1000))

else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation'")