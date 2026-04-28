"""
Dashboard UI Core - Shared utilities, dark theme, card renderer, session helpers
"""

import streamlit as st
import numpy as np


# ============================================================================
# DARK THEME CSS (Telecom NOC Style)
# ============================================================================

def inject_dark_theme():
    """Inject custom dark theme CSS for NOC dashboard look."""
    st.markdown("""
    <style>
    /* Dark background */
    .reportview-container, .main .block-container {
        background-color: #0e1117;
        color: #ececec;
    }
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #111827 !important;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px;
    }
    [data-testid="metric-container"] > div {
        color: #ececec !important;
    }
    /* KPI Status indicators */
    .kpi-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .kpi-card.warning { border-left-color: #f59e0b; }
    .kpi-card.critical { border-left-color: #ef4444; }
    .kpi-card.info { border-left-color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)


def kpi_card_html(title: str, value: str, delta: str = "", status: str = "good"):
    """
    Render a dark-themed KPI card with color-coded status.
    
    Args:
        title: KPI name
        value: KPI value string
        delta: Optional delta text
        status: 'good' (green), 'warning' (orange), 'critical' (red), 'info' (blue)
    """
    status_class = status  # good, warning, critical, info
    delta_html = f"<p style='margin:0; font-size:0.85em; color:#9ca3af;'>{delta}</p>" if delta else ""
    
    html = f"""
    <div class="kpi-card {status_class}">
        <p style="margin:0; font-size:0.75em; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em;">{title}</p>
        <h3 style="margin:4px 0; font-size:1.5em; color:#ececec; font-weight:bold;">{value}</h3>
        {delta_html}
    </div>
    """
    return html


def render_kpi_grid(cards: list):
    """
    Render a grid of KPI cards.
    
    Args:
        cards: List of dicts [{'title': ..., 'value': ..., 'delta': ..., 'status': ...}]
    """
    st.markdown(
        "".join([kpi_card_html(**card) for card in cards]),
        unsafe_allow_html=True
    )


# ============================================================================
# SESSION STATE HELPERS
# ============================================================================

def ensure_session_key(key: str, default=None):
    """Ensure a session state key exists."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def get_cached_sim_data():
    """
    Return precomputed simulation data if available.
    Reads from session_state['simulation'].
    """
    sim = st.session_state.get('simulation', None)
    if sim is None:
        return None
    
    # Cache precomputed arrays
    cache_key = '_cached_sim_data'
    if cache_key not in st.session_state:
        timestamps = np.array([m.timestamp for m in sim.metrics_history])
        sinrs = np.array([m.avg_sinr_db for m in sim.metrics_history])
        throughputs = np.array([m.total_throughput_mbps for m in sim.metrics_history])
        
        # Per-user SINR history (users x steps)
        user_sinr_history = np.array([m.user_sinrs for m in sim.metrics_history])
        
        # Serving BS history
        serving_bs_history = [m.user_serving_bs for m in sim.metrics_history]
        
        st.session_state[cache_key] = {
            'timestamps': timestamps,
            'sinrs': sinrs,
            'throughputs': throughputs,
            'user_sinr_history': user_sinr_history,
            'serving_bs_history': serving_bs_history,
            'num_steps': len(timestamps),
            'num_users': sim.num_users,
            'num_bs': sim.num_bs,
            'bounds': sim.bounds,
            'base_stations': sim.base_stations,
            'users': sim.users
        }
    
    return st.session_state[cache_key]


def clear_sim_cache():
    """Clear cached simulation data (call when new simulation runs)."""
    cache_key = '_cached_sim_data'
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    
    # Also clear derived caches
    for key in list(st.session_state.keys()):
        if key.startswith('_cached_'):
            del st.session_state[key]


# ============================================================================
# COMPUTATION WRAPPERS
# ============================================================================

@st.cache_data
def cached_compute_trajectories(metrics_history_tuple, num_users, bounds):
    """
    Precompute approximate user trajectories from metrics history.
    Since positions aren't stored per step, we generate realistic traces.
    """
    # metrics_history is stored as tuple for hashability
    num_steps = len(metrics_history_tuple)
    if num_steps == 0:
        return np.zeros((0, num_users, 2))
    
    # Generate traces using random waypoint model
    from mobility import RandomWaypointMobility
    model = RandomWaypointMobility(bounds, min_velocity=0.5, max_velocity=5.0)
    
    (x_min, x_max), (y_min, y_max) = bounds
    positions = np.random.uniform([x_min, y_min], [x_max, y_max], (num_users, 2))
    
    traces = np.zeros((num_steps, num_users, 2))
    for step in range(num_steps):
        for u_idx in range(num_users):
            positions[u_idx] = model.update(u_idx, positions[u_idx], dt=0.001)
        traces[step] = positions.copy()
    
    return traces


@st.cache_data
def cached_sinr_heatmap_grid(bounds, base_stations_positions, base_stations_configs,
                             scenario, resolution=20):
    """
    Precompute SINR heatmap grid for the area.
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max))
        base_stations_positions: list of (x, y) arrays
        base_stations_configs: list of dicts with 'tx_power_linear', 'frequency_ghz', 'num_antennas'
        scenario: 'UMi', 'UMa', 'RMa'
        resolution: grid resolution
    
    Returns:
        (X, Y, Z) meshgrid arrays
    """
    from channel import multipath_channel
    from mimo import omnidirectional_beamforming_gain
    
    (x_min, x_max), (y_min, y_max) = bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)
    
    for i in range(resolution):
        for j in range(resolution):
            grid_pos = np.array([X[i, j], Y[i, j]])
            received_powers = np.zeros(len(base_stations_positions))
            
            for bs_idx, (bs_pos, bs_cfg) in enumerate(zip(base_stations_positions, base_stations_configs)):
                distance = np.linalg.norm(grid_pos - bs_pos)
                if distance < 1:
                    distance = 1
                
                channel_gain, _ = multipath_channel(
                    distance, bs_cfg['frequency_ghz'], scenario)
                bf_gain = omnidirectional_beamforming_gain(
                    grid_pos, bs_pos, bs_cfg['num_antennas'])
                
                rx_power = bs_cfg['tx_power_linear'] * channel_gain * bf_gain
                received_powers[bs_idx] = rx_power
            
            best_bs = np.argmax(received_powers)
            signal = received_powers[best_bs]
            interference = np.sum(received_powers) - signal
            sinr_linear = signal / (interference + 1e-12)
            sinr_db = 10 * np.log10(sinr_linear + 1e-12)
            Z[i, j] = sinr_db
    
    return X, Y, Z

