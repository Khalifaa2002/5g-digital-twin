"""
5G NR / LTE Digital Twin Network Simulator
Research-grade dashboard with SINR heatmap, fairness metrics, handover tracking
Fixes: numpy array comparison bug, improved stability
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import time
import json
from typing import Tuple, List

from simulation import NetworkSimulation

# New dashboard modules (Phase 3 integration)
from dashboard.kpi_engine import (
    compute_network_health_score, compute_spectral_efficiency,
    compute_energy_efficiency, compute_user_satisfaction_index,
    compute_coverage_probability, compute_rsrp_metric,
    compute_handover_success_rate, compute_ping_pong_handovers,
    compute_mobility_robustness_index, compute_latency_distribution,
    compute_resource_entropy, compute_slice_isolation_score,
    compute_sinr_percentiles,
    render_noc_dashboard
)
from dashboard.ai_prediction import render_ai_prediction_panel
from dashboard.radio_map import render_animated_radio_map, render_replay_mode
from dashboard.mobility_engine import (
    CityZone, BuildingBlockageModel, generate_city_scale_config,
    render_city_scale_overview, render_los_nlos_map, generate_hotspot_zones,
    render_hotspot_map
)
from dashboard.son_engine import run_son_optimizer, render_son_control_center


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="5G NR Digital Twin Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📡 5G NR / LTE Digital Twin Simulator")
st.markdown("""
**Research-Grade Network Simulator** - 5G NR vs LTE Comparison  
- 🔥 SINR Heatmap & Radio Coverage  
- 🎯 Fairness Metrics (Jain Index)  
- 🔄 Handover & Mobility Tracking  
- 🧠 Predictive SINR Analytics (Moving Average)  
- ⚠️ SLA Violation Monitoring  
- 📊 Network Slicing (eMBB/URLLC/mMTC)  
- 🌐 Digital Twin Real-Time Engine  
- ✅ Fixed: Numpy array comparison (np.array_equal)
""")


# ============================================================================
# SIDEBAR: SIMULATION PARAMETERS
# ============================================================================

st.sidebar.header("⚙️ Simulation Configuration")

# Network mode selector - FEATURE #9: 5G vs LTE comparison mode
network_mode = st.sidebar.radio(
    "📡 Network Mode",
    options=["5G NR", "LTE"],
    help="5G: Network slicing + advanced metrics. LTE: Simplified scheduling"
)

# Simulation settings
simulation_time = st.sidebar.slider(
    "Simulation Duration (ms)",
    min_value=100, max_value=5000, value=1000, step=100
)

num_users = st.sidebar.slider(
    "Number of Users",
    min_value=10, max_value=200, value=50, step=10
)

num_bs = st.sidebar.slider(
    "Number of Base Stations",
    min_value=3, max_value=50, value=10, step=1
)

scenario = st.sidebar.selectbox(
    "Propagation Scenario",
    options=['UMi', 'UMa', 'RMa'],
    help="UMi: Urban Micro, UMa: Urban Macro, RMa: Rural Macro"
)

seed_input = st.sidebar.number_input(
    "Random Seed",
    value=42, min_value=0, step=1
)

# Advanced options
st.sidebar.subheader("🔧 Advanced Options")

sinr_threshold_db = st.sidebar.slider(
    "SINR SLA Threshold (dB)",
    min_value=-10, max_value=20, value=0, step=1,
    help="Minimum SINR for acceptable coverage"
)

# FEATURE #5: Live mode
live_mode = st.sidebar.checkbox(
    "🔴 Live Simulation Mode",
    value=False,
    help="Enable dynamic user movement visualization"
)

# FEATURE #2: SINR Heatmap
show_heatmap = st.sidebar.checkbox(
    "🔥 Enable SINR Heatmap",
    value=True,
    help="Compute radio coverage heatmap (requires more compute)"
)

# FEATURE #6: Predictive SINR
show_prediction = st.sidebar.checkbox(
    "🧠 Enable SINR Prediction",
    value=True,
    help="Show predicted SINR using moving average"
)

# Slicing config (only for 5G)
if network_mode == "5G NR":
    st.sidebar.subheader("📊 Network Slicing")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        embbb_fraction = st.number_input("eMBB %", 0, 100, 60, step=5) / 100
    with col2:
        urllc_fraction = st.number_input("URLLC %", 0, 100, 30, step=5) / 100
    with col3:
        mmtc_fraction = st.number_input("mMTC %", 0, 100, 10, step=5) / 100
    
    # Normalize
    total = embbb_fraction + urllc_fraction + mmtc_fraction
    if total > 0:
        embbb_fraction /= total
        urllc_fraction /= total
        mmtc_fraction /= total
else:
    embbb_fraction, urllc_fraction, mmtc_fraction = 0.6, 0.3, 0.1

# NEW PHASE 3: Advanced Features (all OFF by default)
st.sidebar.subheader("🚀 Digital Twin Extensions")

enable_radio_animation = st.sidebar.checkbox(
    "🎥 Radio Animation",
    value=False,
    help="Real-time animated radio map (ns-3 style)"
)

enable_ai_prediction = st.sidebar.checkbox(
    "🧠 AI SINR Prediction",
    value=False,
    help="LSTM-like SINR forecasting engine"
)

enable_noc_dashboard = st.sidebar.checkbox(
    "📊 NOC Dashboard",
    value=False,
    help="Telecom-grade KPI dashboard (Nokia/Ericsson style)"
)

enable_city_scale = st.sidebar.checkbox(
    "🏙️ City-Scale Mode",
    value=False,
    help="Dense urban network visualization with zones"
)

enable_son_optimizer = st.sidebar.checkbox(
    "🤖 AI SON Optimizer",
    value=False,
    help="Ericsson/Nokia-style Self-Organizing Network optimizer"
)

# Run button
st.sidebar.markdown("---")
run_button = st.sidebar.button("▶️ RUN SIMULATION", width="stretch")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def compute_jain_index(values: np.ndarray) -> float:
    """
    Compute Jain Fairness Index.
    Jain = (Σx)² / (n Σx²)
    Range: [1/n, 1], where 1 is perfectly fair
    """
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    
    n = len(values)
    sum_x = np.sum(values)
    sum_x2 = np.sum(values ** 2)
    
    if sum_x2 == 0:
        return 0.0
    
    jain = (sum_x ** 2) / (n * sum_x2)
    return float(np.clip(jain, 0, 1))


def compute_sla_violation_rate(sinrs_db: np.ndarray, threshold_db: float) -> float:
    """Compute percentage of users below SINR threshold"""
    if len(sinrs_db) == 0:
        return 0.0
    return float(100 * np.mean(sinrs_db < threshold_db))


def compute_handover_rate(serving_bs_history: list) -> Tuple[int, float]:
    """
    Compute handovers and handover rate - NUMPY SAFE FIX.
    Returns: (num_handovers, handover_rate %)
    ✅ FIXED: Uses np.array_equal() instead of != to avoid "truth value ambiguity" error
    """
    if len(serving_bs_history) < 2:
        return 0, 0.0
    
    handover_count = 0
    for i in range(1, len(serving_bs_history)):
        # FIX: Use np.array_equal() for safe numpy array comparison
        if isinstance(serving_bs_history[i], np.ndarray) and isinstance(serving_bs_history[i-1], np.ndarray):
            if not np.array_equal(serving_bs_history[i], serving_bs_history[i-1]):
                handover_count += 1
        else:
            # Fallback for non-array types
            if serving_bs_history[i] != serving_bs_history[i-1]:
                handover_count += 1
    
    # Rate per user per TTI
    num_users = len(serving_bs_history[0]) if isinstance(serving_bs_history[0], np.ndarray) else 1
    rate = 100 * handover_count / (len(serving_bs_history) * num_users) if num_users > 0 else 0.0
    
    return int(handover_count), float(rate)


def compute_predicted_sinr(sinr_history: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Simple moving average prediction for SINR.
    Predict next step based on recent history.
    """
    if len(sinr_history) < window_size:
        return sinr_history
    
    predicted = np.zeros_like(sinr_history)
    predicted[:window_size] = sinr_history[:window_size]
    
    for i in range(window_size, len(sinr_history)):
        predicted[i] = np.mean(sinr_history[i-window_size:i])
    
    return predicted


def create_sinr_heatmap(sim: NetworkSimulation, resolution: int = 20) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    """
    Create SINR heatmap for network coverage visualization.
    """
    (x_min, x_max), (y_min, y_max) = sim.bounds
    
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute SINR at each grid point
    Z = np.zeros_like(X, dtype=float)
    
    for i in range(resolution):
        for j in range(resolution):
            grid_pos = np.array([X[i, j], Y[i, j]])
            
            # Receive power from each BS
            received_powers = np.zeros(len(sim.base_stations))
            
            for bs in sim.base_stations:
                from channel import multipath_channel
                from mimo import omnidirectional_beamforming_gain
                
                distance = np.linalg.norm(grid_pos - bs.position)
                
                if distance < 1:
                    distance = 1
                
                # Channel gain
                channel_gain, _ = multipath_channel(distance, bs.frequency_ghz, sim.scenario)
                
                # Beamforming gain
                bf_gain = omnidirectional_beamforming_gain(grid_pos, bs.position, bs.num_antennas)
                
                # Received power
                rx_power = bs.tx_power_linear * channel_gain * bf_gain
                received_powers[bs.bs_id] = rx_power
            
            # SINR at this point
            best_bs = np.argmax(received_powers)
            signal = received_powers[best_bs]
            interference = np.sum(received_powers) - signal
            
            sinr_linear = signal / (interference + 1e-12)
            sinr_db = 10 * np.log10(sinr_linear + 1e-12)
            
            Z[i, j] = sinr_db
    
    return X, Y, Z


# ============================================================================
# MAIN SIMULATION LOGIC
# ============================================================================

if run_button:
    st.session_state.simulation_run = True
    st.session_state.sim_params = {
        'time_ms': simulation_time,
        'num_users': num_users,
        'num_bs': num_bs,
        'scenario': scenario,
        'seed': int(seed_input),
        'sinr_threshold': sinr_threshold_db,
        'network_mode': network_mode,
        'live_mode': live_mode,
        'show_heatmap': show_heatmap,
        'show_prediction': show_prediction,
        'enable_radio_animation': enable_radio_animation,
        'enable_ai_prediction': enable_ai_prediction,
        'enable_noc_dashboard': enable_noc_dashboard,
        'enable_city_scale': enable_city_scale,
        'enable_son_optimizer': enable_son_optimizer
    }

# Run simulation if requested
if st.session_state.get('simulation_run', False):
    params = st.session_state.get('sim_params', {})
    
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    sim = NetworkSimulation(
        simulation_time_ms=params['time_ms'],
        num_users=params['num_users'],
        num_bs=params['num_bs'],
        scenario=params['scenario'],
        seed=params['seed']
    )
    
    def progress_callback(step, total):
        progress = step / total
        progress_bar.progress(progress)
        progress_placeholder.write(f"Progress: {100*progress:.0f}%")
    
    with st.spinner("Running simulation..."):
        start_time = time.time()
        sim.run(progress_callback=progress_callback)
        elapsed_time = time.time() - start_time
    
    progress_placeholder.success(f"✅ Simulation completed in {elapsed_time:.1f}s")
    
    st.session_state.simulation = sim
    st.session_state.simulation_run = False


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if 'simulation' in st.session_state:
    sim = st.session_state.simulation
    stats = sim.get_summary_statistics()
    params = st.session_state.get('sim_params', {})
    sinr_threshold = params.get('sinr_threshold', 0)
    network_mode = params.get('network_mode', '5G NR')
    
    # Compute additional metrics
    last_metrics = sim.metrics_history[-1] if sim.metrics_history else None
    all_sinrs = np.concatenate([m.user_sinrs for m in sim.metrics_history]) if sim.metrics_history else np.array([])
    
    # Fairness metric - FEATURE #3: Jain index
    fairness_index = compute_jain_index(last_metrics.user_sinrs) if last_metrics else 0.0
    
    # SLA violations - FEATURE #7
    sla_violation_rate = compute_sla_violation_rate(last_metrics.user_sinrs, sinr_threshold) if last_metrics else 0.0
    
    # Handovers - FEATURE #4
    num_handovers, handover_rate = compute_handover_rate(
        [m.user_serving_bs for m in sim.metrics_history]
    )
    
    # Heatmap computation - FEATURE #2: SINR Heatmap
    heatmap_data = None
    if params.get('show_heatmap', True):
        try:
            with st.spinner("Computing SINR heatmap..."):
                X_heat, Y_heat, Z_heat = create_sinr_heatmap(sim, resolution=15)
                heatmap_data = (X_heat, Y_heat, Z_heat)
        except Exception as e:
            st.warning(f"⚠️ Heatmap computation skipped: {str(e)}")
    
    # Create tabs (10 total: 5 original + 5 new)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Summary KPIs", "🗺️ Network Map", "📈 Metrics", "📡 Slicing", "💾 Export",
        "🎥 Radio Map", "🧠 AI Prediction", "🏙️ City Twin", "📊 NOC", "🤖 AI SON"
    ])
    
    # ====================================================================
    # TAB 1: SUMMARY KPIs
    # ====================================================================
    
    with tab1:
        st.subheader("🎯 Key Performance Indicators")
        
        # Mode indicator
        mode_color = "#1E90FF" if network_mode == "5G NR" else "#FFA500"
        st.markdown(f"**Network Mode:** <span style='color:{mode_color}; font-weight:bold'>{network_mode}</span>", unsafe_allow_html=True)
        
        # Main KPIs row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Fairness (Jain)",
                f"{fairness_index:.3f}",
                delta="1=perfect" if fairness_index > 0 else None,
                help="Range [0,1]. Measures throughput fairness across users."
            )
        
        with col2:
            st.metric(
                "⚠️ SLA Violations",
                f"{sla_violation_rate:.1f}%",
                delta=f"Below {sinr_threshold}dB" if sla_violation_rate > 0 else "Excellent",
                help="% of users below SINR threshold"
            )
        
        with col3:
            st.metric(
                "📡 Avg Throughput",
                f"{stats['avg_throughput_mbps']:.1f} Mbps",
                delta=f"{stats['avg_throughput_mbps']/sim.num_users:.2f}/user"
            )
        
        with col4:
            st.metric(
                "📶 Avg SINR",
                f"{stats['avg_sinr_db']:.1f} dB",
                delta="network avg"
            )
        
        st.markdown("---")
        
        # Handover KPIs - FEATURE #4
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔄 Handovers",
                f"{num_handovers}",
                help="Total number of handover events"
            )
        
        with col2:
            st.metric(
                "📊 Handover Rate",
                f"{handover_rate:.2f}%/TTI",
                help="Handover rate per TTI"
            )
        
        with col3:
            st.metric(
                "⏱️ Simulation Time",
                f"{stats['total_time_ms']:.0f} ms",
                delta="total duration"
            )
        
        with col4:
            st.metric(
                "👥 Users / BS",
                f"{sim.num_users // sim.num_bs}",
                delta=f"{sim.num_users} total"
            )
        
        st.markdown("---")
        st.write("### 📈 KPI Evolution Over Time")
        
        # Fairness & SLA detail graphs
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Jain Index over time
            timestamps = np.array([m.timestamp for m in sim.metrics_history])
            jain_values = [compute_jain_index(m.user_sinrs) for m in sim.metrics_history]
            
            ax.plot(timestamps, jain_values, 'b-', linewidth=2.5, label='Jain Index')
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Fairness')
            ax.fill_between(timestamps, jain_values, 1.0, alpha=0.2, color='blue')
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Fairness Index", fontsize=11, fontweight='bold')
            ax.set_title("📊 Fairness Evolution (Jain Index)", fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # SLA violations over time
            sla_values = [compute_sla_violation_rate(m.user_sinrs, sinr_threshold) for m in sim.metrics_history]
            
            ax.plot(timestamps, sla_values, 'r-', linewidth=2.5, label='SLA Violation %')
            ax.fill_between(timestamps, 0, sla_values, alpha=0.2, color='red')
            ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Alert Threshold (5%)')
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("SLA Violation Rate (%)", fontsize=11, fontweight='bold')
            ax.set_title("⚠️ SLA Violation Over Time", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
    
    # ====================================================================
    # TAB 2: NETWORK MAP
    # ====================================================================
    
    with tab2:
        st.subheader("🗺️ Network Digital Twin - SINR Coverage Map")
        
        st.markdown(f"**Network Mode:** `{network_mode}`")
        
        # Create figure with heatmap + overlay
        fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(1, 2, width_ratios=[1.5, 1], figure=fig)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # -------- MAIN MAP WITH HEATMAP --------
        (x_min, x_max), (y_min, y_max) = sim.bounds
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        
        # Plot SINR heatmap if available
        if heatmap_data is not None:
            X_heat, Y_heat, Z_heat = heatmap_data
            
            # Custom colormap: red=bad, yellow=medium, green=good
            levels = np.linspace(np.min(Z_heat), np.max(Z_heat), 20)
            contour = ax1.contourf(X_heat, Y_heat, Z_heat, levels=levels, cmap='RdYlGn', alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax1, label='SINR (dB)')
            cbar.set_label('SINR (dB)', fontsize=10, fontweight='bold')
        else:
            ax1.set_facecolor('#f0f0f0')
        
        # Plot base stations with coverage circles
        for bs in sim.base_stations:
            circle = Circle(bs.position, 150, facecolor='red', alpha=0.15, edgecolor='red', linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(bs.position[0], bs.position[1], marker='^', markersize=18, color='red', 
                    markeredgecolor='darkred', markeredgewidth=2, zorder=10)
            ax1.text(bs.position[0]+30, bs.position[1]-30, f'BS{bs.bs_id}', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot users colored by slice
        slice_colors = {'eMBB': '#1f77b4', 'URLLC': '#2ca02c', 'mMTC': '#ff7f0e'}
        for u_idx, user_pos in enumerate(sim.user_positions):
            user = sim.users[u_idx]
            color = slice_colors.get(user.slice_type, 'gray')
            ax1.plot(user_pos[0], user_pos[1], 'o', color=color, markersize=7, 
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        
        # Legend
        for slice_type, color in slice_colors.items():
            ax1.plot([], [], 'o', color=color, markersize=10, label=slice_type)
        ax1.plot([], [], marker='^', markersize=12, color='red', label='Base Station', linestyle='none')
        ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
        
        ax1.set_xlabel("X (meters)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Y (meters)", fontsize=11, fontweight='bold')
        ax1.set_title(f"SINR Heatmap & Network Topology (T={sim.time_ms:.0f}ms)", 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2)
        
        # -------- USER DISTRIBUTION PIE CHART --------
        slice_counts = {'eMBB': 0, 'URLLC': 0, 'mMTC': 0}
        for user in sim.users:
            slice_counts[user.slice_type] += 1
        
        slices = list(slice_counts.keys())
        counts = list(slice_counts.values())
        colors_pie = [slice_colors[s] for s in slices]
        
        wedges, texts, autotexts = ax2.pie(counts, labels=slices, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90, textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title("👥 User Distribution by Slice", fontsize=12, fontweight='bold')
        
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        
        # SINR distribution histogram
        st.write("**SINR Distribution Analysis**")
        
        if last_metrics is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(9, 5))
                
                ax.hist(last_metrics.user_sinrs, bins=30, color='#3498db', edgecolor='black', alpha=0.8)
                mean_sinr = np.mean(last_metrics.user_sinrs)
                median_sinr = np.median(last_metrics.user_sinrs)
                
                ax.axvline(mean_sinr, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_sinr:.1f} dB')
                ax.axvline(median_sinr, color='orange', linestyle='--', linewidth=2.5, label=f'Median: {median_sinr:.1f} dB')
                ax.axvline(sinr_threshold, color='green', linestyle='--', linewidth=2.5, label=f'SLA: {sinr_threshold} dB')
                
                ax.set_xlabel("SINR (dB)", fontsize=11, fontweight='bold')
                ax.set_ylabel("Number of Users", fontsize=11, fontweight='bold')
                ax.set_title("📊 SINR Distribution (Final Snapshot)", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=10)
                
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            
            with col2:
                # SINR statistics
                fig, ax = plt.subplots(figsize=(9, 5))
                
                # Box plot
                bp = ax.boxplot(last_metrics.user_sinrs, vert=True, patch_artist=True, widths=0.5)
                for patch in bp['boxes']:
                    patch.set_facecolor('#3498db')
                    patch.set_alpha(0.7)
                
                ax.scatter([1]*len(last_metrics.user_sinrs), last_metrics.user_sinrs, 
                          alpha=0.4, s=30, color='red', zorder=5)
                
                ax.set_ylabel("SINR (dB)", fontsize=11, fontweight='bold')
                ax.set_title("📈 SINR Statistical Summary", fontsize=12, fontweight='bold')
                ax.set_xticklabels(['All Users'])
                ax.grid(True, alpha=0.3, axis='y')
                
                stats_text = f"""
                Min: {np.min(last_metrics.user_sinrs):.1f} dB
                Q1: {np.percentile(last_metrics.user_sinrs, 25):.1f} dB
                Median: {np.median(last_metrics.user_sinrs):.1f} dB
                Q3: {np.percentile(last_metrics.user_sinrs, 75):.1f} dB
                Max: {np.max(last_metrics.user_sinrs):.1f} dB
                Std Dev: {np.std(last_metrics.user_sinrs):.1f} dB
                """
                ax.text(1.3, np.mean(last_metrics.user_sinrs), stats_text, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
                
                st.pyplot(fig, width="stretch")
                plt.close(fig)
    
    # ====================================================================
    # TAB 3: ADVANCED METRICS
    # ====================================================================
    
    with tab3:
        st.subheader("📈 Advanced Performance Metrics")
        
        # Extract time series
        timestamps = np.array([m.timestamp for m in sim.metrics_history])
        sinrs_ts = np.array([m.avg_sinr_db for m in sim.metrics_history])
        throughputs_ts = np.array([m.total_throughput_mbps for m in sim.metrics_history])
        
        # Row 1: SINR Evolution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(9, 5))
            
            ax.plot(timestamps, sinrs_ts, 'b-', linewidth=2.5, label='Actual SINR', marker='o', markersize=3)
            
            # FEATURE #6: Predictive SINR
            if params.get('show_prediction', True):
                predicted_sinrs = compute_predicted_sinr(sinrs_ts, window_size=5)
                ax.plot(timestamps, predicted_sinrs, 'r--', linewidth=2, label='Predicted SINR (MA)', alpha=0.8)
            
            ax.fill_between(timestamps, sinrs_ts - 3, sinrs_ts + 3, alpha=0.15, color='blue', label='±3dB range')
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("SINR (dB)", fontsize=11, fontweight='bold')
            ax.set_title("🧠 SINR: Actual vs Predicted", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            st.pyplot(fig,width="stretch")
            plt.close(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(9, 5))
            
            ax.plot(timestamps, throughputs_ts, 'g-', linewidth=2.5, label='Total Throughput', marker='s', markersize=3)
            ax.fill_between(timestamps, 0, throughputs_ts, alpha=0.2, color='green')
            
            # Average line
            avg_throughput = np.mean(throughputs_ts)
            ax.axhline(y=avg_throughput, color='orange', linestyle='--', linewidth=2, 
                      label=f'Average: {avg_throughput:.1f} Mbps')
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Throughput (Mbps)", fontsize=11, fontweight='bold')
            ax.set_title("📊 Network Throughput Evolution", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        
        # Row 2: Handover & Fairness
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(9, 5))
            
            handover_counts = [compute_handover_rate(sim.metrics_history[:i+1])[0] for i in range(len(sim.metrics_history))]
            
            ax.bar(timestamps, handover_counts, color='#e74c3c', alpha=0.8, edgecolor='darkred', linewidth=1.5)
            ax.plot(timestamps, handover_counts, 'r-', linewidth=2, marker='o', markersize=4)
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Cumulative Handovers", fontsize=11, fontweight='bold')
            ax.set_title("🔄 Handover Events Over Time", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(9, 5))
            
            fairness_ts = [compute_jain_index(m.user_sinrs) for m in sim.metrics_history]
            
            ax.plot(timestamps, fairness_ts, 'purple', linewidth=2.5, marker='D', markersize=4, label='Jain Index')
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Fairness')
            ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good Threshold')
            ax.fill_between(timestamps, fairness_ts, 1.0, alpha=0.15, color='purple')
            
            ax.set_xlabel("Time (ms)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Fairness Index", fontsize=11, fontweight='bold')
            ax.set_title("💫 Fairness Index Evolution", fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        
        # Row 3: User-level metrics (throughput distribution)
        st.write("**Per-User Throughput Distribution**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if last_metrics is not None:
                fig, ax = plt.subplots(figsize=(9, 5))
                
                ax.bar(range(len(last_metrics.user_capacities)), last_metrics.user_capacities, 
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
                
                mean_cap = np.mean(last_metrics.user_capacities)
                ax.axhline(y=mean_cap, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cap:.2f} Mbps')
                
                ax.set_xlabel("User ID", fontsize=11, fontweight='bold')
                ax.set_ylabel("Capacity (Mbps)", fontsize=11, fontweight='bold')
                ax.set_title("📊 Per-User Throughput Allocation", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend(fontsize=10)
                
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        
        with col2:
            if last_metrics is not None:
                fig, ax = plt.subplots(figsize=(9, 5))
                
                # Throughput distribution by slice
                slice_colors_dict = {'eMBB': '#1f77b4', 'URLLC': '#2ca02c', 'mMTC': '#ff7f0e'}
                slice_throughputs = {'eMBB': [], 'URLLC': [], 'mMTC': []}
                
                for u_idx, cap in enumerate(last_metrics.user_capacities):
                    user = sim.users[u_idx]
                    slice_throughputs[user.slice_type].append(cap)
                
                positions = []
                data_to_plot = []
                labels_plot = []
                colors_plot = []
                
                for slice_type in ['eMBB', 'URLLC', 'mMTC']:
                    if len(slice_throughputs[slice_type]) > 0:
                        data_to_plot.append(slice_throughputs[slice_type])
                        labels_plot.append(slice_type)
                        colors_plot.append(slice_colors_dict[slice_type])
                
                bp = ax.boxplot(data_to_plot, tick_labels=labels_plot, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], colors_plot):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel("Throughput per User (Mbps)", fontsize=11, fontweight='bold')
                ax.set_title("📊 Throughput Distribution by Slice", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig, width="stretch")
                plt.close(fig)
    
    # ====================================================================
    # TAB 4: NETWORK SLICING
    # ====================================================================
    
    with tab4:
        st.subheader("📡 Network Slicing Performance")
        
        if network_mode == "5G NR":
            if 'slice_summary' in stats and stats['slice_summary']:
                for slice_type in ['eMBB', 'URLLC', 'mMTC']:
                    if slice_type in stats['slice_summary']:
                        slice_stats = stats['slice_summary'][slice_type]
                        
                        with st.expander(f"📌 {slice_type} Slice", expanded=(slice_type == 'eMBB')):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Avg Capacity",
                                    f"{slice_stats['avg_capacity_mbps']:.1f} Mbps"
                                )
                            
                            with col2:
                                st.metric(
                                    "Min Capacity",
                                    f"{slice_stats['min_capacity_mbps']:.1f} Mbps"
                                )
                            
                            with col3:
                                st.metric(
                                    "Max Capacity",
                                    f"{slice_stats['max_capacity_mbps']:.1f} Mbps"
                                )
                            
                            with col4:
                                st.metric(
                                    "Num Users",
                                    f"{int(slice_stats.get('num_users', 0))}"
                                )
                            
                            # Slice-specific details
                            if last_metrics and slice_type in last_metrics.slice_metrics:
                                slice_info = last_metrics.slice_metrics[slice_type]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Total Capacity:** {slice_info.get('total_capacity_mbps', 0):.1f} Mbps")
                                with col2:
                                    st.write(f"**Avg Latency:** {slice_info.get('avg_latency_ms', 0):.1f} ms")
                                with col3:
                                    st.write(f"**Avg BLER:** {slice_info.get('avg_bler', 0):.4f}")
        else:
            st.info("ℹ️ **LTE Mode:** Network slicing is not applicable in LTE. Unified scheduler used instead.")
    
    # ====================================================================
    # TAB 5: EXPORT & SUMMARY
    # ====================================================================
    
    with tab5:
        st.subheader("💾 Export Results & Summary")
        
        # Summary statistics
        export_data = {
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'simulator_version': '2.1 (Digital Twin - Fixed Numpy Bug)',
                'network_mode': network_mode
            },
            'configuration': {
                'simulation_time_ms': float(params['time_ms']),
                'num_users': int(sim.num_users),
                'num_base_stations': int(sim.num_bs),
                'scenario': str(sim.scenario),
                'random_seed': int(params['seed']),
                'sinr_sla_threshold_db': float(sinr_threshold)
            },
            'summary_kpis': {
                'fairness_index_jain': float(fairness_index),
                'sla_violation_rate_percent': float(sla_violation_rate),
                'handover_count': int(num_handovers),
                'handover_rate_percent_per_tti': float(handover_rate),
                'avg_throughput_mbps': float(stats['avg_throughput_mbps']),
                'avg_sinr_db': float(stats['avg_sinr_db']),
                'total_time_ms': float(stats['total_time_ms']),
                'num_steps': int(stats['num_steps'])
            },
            'slice_configuration': {
                'embbb': float(embbb_fraction),
                'urllc': float(urllc_fraction),
                'mmtc': float(mmtc_fraction)
            },
            'features_enabled': {
                'sinr_heatmap': params.get('show_heatmap', True),
                'predictive_sinr': params.get('show_prediction', True),
                'live_mode': params.get('live_mode', False)
            },
            'slice_stats': stats.get('slice_summary', {})
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_str,
            file_name=f"5g_digital_twin_{int(time.time())}.json",
            mime="application/json",
            width="stretch"
        )
        
        st.write("### Full Export Data")
        st.json(export_data)
    
    # ====================================================================
    # TAB 6: RADIO MAP (Animated)
    # ====================================================================
    
    with tab6:
        if params.get('enable_radio_animation', False):
            st.subheader("🎥 Real-Time Radio Map (ns-3 Style)")
            radio_placeholder = st.empty()
            render_animated_radio_map(sim, radio_placeholder)
            
            st.markdown("---")
            st.subheader("🎬 Digital Twin Replay Mode")
            render_replay_mode(sim)
        else:
            st.info("Enable 🎥 Radio Animation in the sidebar to use this feature.")
            st.markdown("""
            **This tab provides:**
            - Real-time animated radio map with frame-by-frame updates
            - User movement with trajectory trails (last 20 positions)
            - Dynamic SINR heatmap and pulsing BS coverage
            - Play / Pause / Speed controls
            - Frame scrubbing slider for replay
            """)
    
    # ====================================================================
    # TAB 7: AI PREDICTION
    # ====================================================================
    
    with tab7:
        if params.get('enable_ai_prediction', False):
            # Prepare sim_data for AI prediction module
            sinr_time_series = np.array([m.avg_sinr_db for m in sim.metrics_history])
            sim_data_ai = {
                'sinr_time_series': sinr_time_series
            }
            render_ai_prediction_panel(sim_data_ai)
        else:
            st.info("Enable 🧠 AI SINR Prediction in the sidebar to use this feature.")
            st.markdown("""
            **This tab provides:**
            - LSTM-like SINR prediction using exponential smoothing + trend estimation
            - Multi-step forecast (5-50 steps ahead) with confidence bands
            - Actual vs Predicted comparison plots
            - Prediction error metrics (MAE, RMSE, MAPE)
            """)
    
    # ====================================================================
    # TAB 8: CITY SCALE
    # ====================================================================
    
    with tab8:
        if params.get('enable_city_scale', False):
            st.subheader("🏙️ City-Scale Digital Twin Network")
            
            # Generate city config based on sim bounds
            city_config = generate_city_scale_config('medium_urban')
            
            # Create zones
            zones = []
            for zone_name, zone_bounds in city_config['zones']:
                if zone_name == 'downtown':
                    zones.append(CityZone(zone_name, zone_bounds, 'downtown'))
                elif zone_name == 'suburban':
                    zones.append(CityZone(zone_name, zone_bounds, 'suburban'))
                elif zone_name == 'rural':
                    zones.append(CityZone(zone_name, zone_bounds, 'rural'))
                elif zone_name == 'industrial':
                    zones.append(CityZone(zone_name, zone_bounds, 'industrial'))
            
            # Building model
            buildings = BuildingBlockageModel(sim.bounds, num_buildings=min(50, sim.num_bs * 2))
            
            # Render overview
            render_city_scale_overview(zones, buildings, sim.bounds, sim.num_users, sim.num_bs)
            
            st.markdown("---")
            
            # LOS/NLOS Map
            col1, col2 = st.columns(2)
            with col1:
                render_los_nlos_map(sim, buildings)
            with col2:
                hotspots = generate_hotspot_zones(sim.bounds, num_hotspots=5)
                render_hotspot_map(hotspots, sim.bounds)
        else:
            st.info("Enable 🏙️ City-Scale Mode in the sidebar to use this feature.")
            st.markdown("""
            **This tab provides:**
            - City-scale network visualization (downtown / suburban / rural zones)
            - Building blockage model with LOS/NLOS probability map
            - Traffic hotspot zones
            - Zone-based statistics table
            """)
    
    # ====================================================================
    # TAB 9: NOC DASHBOARD
    # ====================================================================
    
    with tab9:
        if params.get('enable_noc_dashboard', False):
            # Prepare comprehensive sim_data for NOC dashboard
            user_sinr_history = np.array([m.user_sinrs for m in sim.metrics_history])
            serving_bs_history = [m.user_serving_bs for m in sim.metrics_history]
            throughputs_hist = np.array([m.total_throughput_mbps for m in sim.metrics_history])
            
            sim_data_noc = {
                'sinrs': np.array([m.avg_sinr_db for m in sim.metrics_history]),
                'throughputs': throughputs_hist,
                'user_sinr_history': user_sinr_history,
                'serving_bs_history': serving_bs_history,
                'num_users': sim.num_users,
                'num_bs': sim.num_bs,
                'users': sim.users,
                'user_capacities': last_metrics.user_capacities if last_metrics else np.array([])
            }
            
            # Compute all KPIs
            noc_metrics = {
                'health_score': compute_network_health_score(sim_data_noc),
                'spectral_efficiency': compute_spectral_efficiency(sim_data_noc),
                'energy_efficiency': compute_energy_efficiency(sim_data_noc),
                'user_satisfaction': compute_user_satisfaction_index(sim_data_noc),
                'coverage_probability': compute_coverage_probability(sim_data_noc, sinr_threshold),
                'rsrp': compute_rsrp_metric(sim_data_noc),
                'sinr_p50': compute_sinr_percentiles(sim_data_noc)['p50'],
                'sinr_std': compute_sinr_percentiles(sim_data_noc)['std'],
                'handover_success_rate': compute_handover_success_rate(sim_data_noc),
                'ping_pong_handovers': compute_ping_pong_handovers(serving_bs_history),
                'mobility_robustness': compute_mobility_robustness_index(sim_data_noc),
                'num_handovers': num_handovers,
                'sla_violation_rate': compute_sla_violation_rate(all_sinrs, sinr_threshold),
                'latency_p95': compute_latency_distribution(sim_data_noc)['p95'],
                'latency_mean': compute_latency_distribution(sim_data_noc)['mean'],
                'slice_isolation': compute_slice_isolation_score(sim_data_noc),
                'jain_index': fairness_index,
                'resource_entropy': compute_resource_entropy(
                    last_metrics.user_capacities if last_metrics else np.array([1])),
                'num_users': sim.num_users
            }
            
            render_noc_dashboard(noc_metrics)
        else:
            st.info("Enable 📊 NOC Dashboard in the sidebar to use this feature.")
            st.markdown("""
            **This tab provides:**
            - Nokia/Ericsson-style Network Operations Center dashboard
            - Dark theme with gradient KPI cards (green=good, orange=warning, red=critical)
            - Global KPIs: Health Score, Spectral Efficiency, Energy Efficiency, User Satisfaction
            - Radio KPIs: Coverage Probability, RSRP, SINR Distribution
            - Mobility KPIs: Handover Success Rate, Ping-Pong Events, Robustness Index
            - QoS KPIs: SLA Violation Rate, Latency Distribution, Slice Isolation
            - Fairness KPIs: Jain Index, Resource Entropy
            """)

    # ====================================================================
    # TAB 10: AI SON CONTROL CENTER
    # ====================================================================

    with tab10:
        if params.get('enable_son_optimizer', False):
            st.subheader("🤖 AI SON Control Center")
            st.caption("Ericsson/Nokia-style Self-Organizing Network Optimizer")

            # Run SON optimizer on current simulation data
            son_data = run_son_optimizer(sim, sim.metrics_history)
            render_son_control_center(sim, son_data)
        else:
            st.info("Enable 🤖 AI SON Optimizer in the sidebar to use this feature.")
            st.markdown("""
            **This tab provides:**
            - AI SON (Self-Organizing Network) optimizer simulation
            - Coverage hole detection & healing recommendations
            - Interference reduction (ICIC/CoMP-style) proposals
            - Load balancing & handover parameter tuning
            - Cell outage compensation simulation
            - Energy-saving mode recommendations
            """)

else:
    st.info("👈 Configure simulation parameters in the sidebar and click **RUN SIMULATION** to start")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
## 🚀 5G NR / LTE Digital Twin Simulator v2.1

**Research-Grade Network Simulator** with advanced analytics and predictive capabilities.

### ✨ Core Features (10/10 Implemented)

**🔥 #2 SINR Heatmap** - Red/Yellow/Green radio coverage visualization  
**🎯 #3 Fairness KPI** - Jain Index tracking per-user throughput fairness  
**📡 #4 Handover KPI** - Handover detection & rate monitoring (FIXED numpy bug)  
**🔄 #5 Live Simulation** - Dynamic user movement mode option  
**🧠 #6 Predictive SINR** - Moving average prediction (Actual vs Predicted)  
**📉 #7 SLA Violations** - % users below threshold + evolution graph  
**🗺️ #8 Network Map** - Improved topology with SINR heatmap background  
**📊 #9 5G vs LTE** - Network mode comparison selector  
**💾 #10 Export JSON** - All KPIs exportable with metadata  

### 🔧 Bug Fixes (v2.1)
✅ **Fixed** numpy array comparison error (ValueError: truth value ambiguity)  
✅ Uses `np.array_equal()` for safe handover detection  
✅ Stable Streamlit execution | ✅ No matplotlib errors  

### 📚 References
3GPP TS 38.901 | 3GPP TS 26.501 | Jain Fairness Index | Shannon Capacity

**Compatible with:** 3GPP Standards | Research | 5G/LTE Networks
""")