"""
Real-Time Animated Radio Map (ns-3 Visualizer Style)
Implements frame-by-frame animated network visualization with replay controls.

FIXED (v4): Separated matplotlib figure rendering from Streamlit widgets
 to prevent StreamlitDuplicateElementKey errors.
FIXED (v5): Safe None placeholder handling + facecolor for Circle patches.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import streamlit as st
import time
from typing import List, Tuple


# ============================================================================
# TRAJECTORY & HISTORY PRECOMPUTATION
# ============================================================================

@st.cache_data
def precompute_user_trajectories(metrics_history: List, num_users: int) -> np.ndarray:
    """
    Precompute user trajectories from metrics history.
    Since full position history is not stored, we generate realistic
    trajectories consistent with the mobility model used.
    """
    num_steps = len(metrics_history)
    trajectories = np.zeros((num_steps, num_users, 2))
    np.random.seed(42)

    for u_idx in range(num_users):
        base_angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(50, 300)
        angular_speed = np.random.uniform(-0.05, 0.05)
        radial_noise = np.random.uniform(0.5, 3.0)

        positions = []
        for step in range(num_steps):
            angle = base_angle + angular_speed * step
            if step < len(metrics_history):
                sinr = metrics_history[step].user_sinrs[u_idx] if u_idx < len(metrics_history[step].user_sinrs) else 0
                r_var = radial_noise * (1 + 5 / (abs(sinr) + 5))
                r = radius + np.random.normal(0, r_var)
            else:
                r = radius
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            positions.append([x, y])

        trajectories[:, u_idx, :] = np.array(positions)

    return trajectories


# ============================================================================
# PURE MATPLOTLIB FRAME RENDERER (NO Streamlit WIDGETS)
# ============================================================================

def _render_radio_frame_figure(sim, frame_idx: int, trajectories: np.ndarray) -> plt.Figure:
    """
    Render a single radio map frame as a matplotlib Figure.
    This function contains ZERO Streamlit widgets — only plotting.
    Called by both render_animated_radio_map and render_replay_mode.
    """
    metrics_history = sim.metrics_history
    metrics = metrics_history[frame_idx]
    timestamp = metrics.timestamp if hasattr(metrics, 'timestamp') else frame_idx
    num_steps = len(metrics_history)

    fig, ax = plt.subplots(figsize=(10, 10))
    (x_min, x_max), (y_min, y_max) = sim.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a1a')

    # SINR heatmap background
    xx = np.linspace(x_min, x_max, 50)
    yy = np.linspace(y_min, y_max, 50)
    XX, YY = np.meshgrid(xx, yy)

    mean_sinr = metrics.avg_sinr_db if hasattr(metrics, 'avg_sinr_db') else 5.0
    ZZ = mean_sinr - 0.008 * np.sqrt(XX**2 + YY**2)
    ZZ += 4 * np.sin(XX / 80) * np.cos(YY / 80)
    ZZ = np.clip(ZZ, -15, 30)

    contour = ax.contourf(XX, YY, ZZ, levels=20, cmap='inferno', alpha=0.6)
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('SINR (dB)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Base stations with pulsing coverage
    for bs in sim.base_stations:
        pulse = 1 + 0.15 * np.sin(frame_idx * 0.3 + bs.bs_id)
        radius = 120 * pulse
        circle = Circle(bs.position, radius, facecolor='cyan', alpha=0.08)
        ax.add_patch(circle)
        circle2 = Circle(bs.position, radius * 0.7, facecolor='cyan', alpha=0.12)
        ax.add_patch(circle2)

        ax.plot(bs.position[0], bs.position[1], marker='^', markersize=14,
                color='cyan', markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        ax.text(bs.position[0] + 20, bs.position[1] + 20, f'BS{bs.bs_id}',
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='navy', alpha=0.7))

    # User trajectories (last 20 positions trail)
    trail_length = 20
    start_trail = max(0, frame_idx - trail_length)
    slice_colors = {'eMBB': '#00ff88', 'URLLC': '#ff3366', 'mMTC': '#ffaa00'}

    for u_idx in range(min(sim.num_users, 100)):
        trail_x = trajectories[start_trail:frame_idx + 1, u_idx, 0]
        trail_y = trajectories[start_trail:frame_idx + 1, u_idx, 1]

        if len(trail_x) > 1:
            for t in range(len(trail_x) - 1):
                alpha = (t + 1) / len(trail_x) * 0.6
                ax.plot([trail_x[t], trail_x[t + 1]], [trail_y[t], trail_y[t + 1]],
                        color='white', alpha=alpha, linewidth=0.8)

        if frame_idx < len(trajectories):
            pos = trajectories[frame_idx, u_idx]
            user = sim.users[u_idx] if u_idx < len(sim.users) else None
            color = slice_colors.get(getattr(user, 'slice_type', 'eMBB'), 'white')
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # Legend
    for stype, color in slice_colors.items():
        ax.plot([], [], 'o', color=color, markersize=8, label=stype)
    ax.plot([], [], marker='^', markersize=10, color='cyan', label='BS', linestyle='none')
    ax.legend(loc='upper left', fontsize=9, facecolor='black', edgecolor='white',
              labelcolor='white')

    ax.set_title(f"Digital Twin Radio Map | T={timestamp:.0f}ms | Frame {frame_idx}/{num_steps}",
                 fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel("X (meters)", fontsize=10, color='white')
    ax.set_ylabel("Y (meters)", fontsize=10, color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='gray')
    fig.patch.set_facecolor('#0a0a1a')

    return fig


# ============================================================================
# ANIMATED RADIO MAP (with its OWN unique controls)
# ============================================================================

def render_animated_radio_map(sim, placeholder: st.empty):
    """
    Render animated radio map with play/pause/speed controls.
    Uses precomputed data and st.empty() for smooth updates.
    FIXED: Safe None handling for placeholder.
    """
    metrics_history = sim.metrics_history
    num_steps = len(metrics_history)

    if num_steps < 2:
        st.warning("Not enough simulation steps for animation.")
        return

    trajectories = precompute_user_trajectories(metrics_history, sim.num_users)

    # Session state init
    if 'radio_map_frame' not in st.session_state:
        st.session_state.radio_map_frame = 0
    if 'radio_map_playing' not in st.session_state:
        st.session_state.radio_map_playing = False
    if 'radio_map_speed' not in st.session_state:
        st.session_state.radio_map_speed = 1.0

    # ---- Playback Controls (UNIQUE keys prefixed with radio_main_) ----
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])

    with col1:
        label = "Play" if not st.session_state.radio_map_playing else "Pause"
        if st.button(label, key="radio_main_playpause"):
            st.session_state.radio_map_playing = not st.session_state.radio_map_playing

    with col2:
        if st.button("Stop", key="radio_main_stop"):
            st.session_state.radio_map_playing = False
            st.session_state.radio_map_frame = 0

    with col3:
        speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 3.0, 5.0],
                                  value=st.session_state.radio_map_speed,
                                  key="radio_main_speed")
        st.session_state.radio_map_speed = speed

    with col4:
        frame_num = st.slider("Frame", 0, num_steps - 1,
                               st.session_state.radio_map_frame,
                               key="radio_main_frame")
        st.session_state.radio_map_frame = frame_num

    # ---- Render Frame ----
    frame_idx = st.session_state.radio_map_frame
    fig = _render_radio_frame_figure(sim, frame_idx, trajectories)
    if placeholder is not None:
        placeholder.pyplot(fig, width="stretch")
    else:
        st.pyplot(fig, width="stretch")
    plt.close(fig)

    # ---- Auto-advance if playing ----
    if st.session_state.radio_map_playing:
        st.session_state.radio_map_frame = (frame_idx + 1) % num_steps
        time.sleep(0.2 / st.session_state.radio_map_speed)
        st.rerun()


# ============================================================================
# REPLAY MODE (with its OWN unique controls, does NOT call render_animated_radio_map)
# ============================================================================

def render_replay_mode(sim):
    """
    Render a stable replay mode with full play/pause/speed and frame scrubbing.
    FIXED: Does NOT call render_animated_radio_map — uses shared _render_radio_frame_figure.
    """
    metrics_history = sim.metrics_history
    num_steps = len(metrics_history)

    if num_steps < 2:
        st.warning("Not enough simulation steps for replay.")
        return

    st.subheader("Digital Twin Replay Mode")

    # Precompute trajectories once
    trajectories = precompute_user_trajectories(metrics_history, sim.num_users)

    # State management
    if 'replay_frame' not in st.session_state:
        st.session_state.replay_frame = 0
    if 'replay_playing' not in st.session_state:
        st.session_state.replay_playing = False

    # Controls in a single row (UNIQUE keys prefixed with replay_)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 3, 1])

    with c1:
        label = "Play" if not st.session_state.replay_playing else "Pause"
        if st.button(label, key="replay_ctrl_pp"):
            st.session_state.replay_playing = not st.session_state.replay_playing

    with c2:
        if st.button("Stop", key="replay_ctrl_stop"):
            st.session_state.replay_playing = False
            st.session_state.replay_frame = 0

    with c3:
        speed = st.selectbox("Speed", [0.5, 1.0, 2.0, 5.0],
                             index=1, key="replay_ctrl_speed")

    with c4:
        frame = st.slider("Frame Scrubber", 0, num_steps - 1,
                          st.session_state.replay_frame, key="replay_ctrl_scrubber")
        st.session_state.replay_frame = frame

    with c5:
        st.write(f"**{frame}/{num_steps}**")

    # Status indicator
    status_color = "green" if st.session_state.replay_playing else "gray"
    status_text = "PLAYING" if st.session_state.replay_playing else "PAUSED"
    st.markdown(
        f"Status: <span style='color:{status_color}; font-weight:bold'>{status_text}</span>",
        unsafe_allow_html=True
    )

    # Render the frame using the SHARED figure renderer
    placeholder = st.empty()
    fig = _render_radio_frame_figure(sim, st.session_state.replay_frame, trajectories)
    placeholder.pyplot(fig, width="stretch")
    plt.close(fig)

    # Auto-advance (independent of radio_map state)
    if st.session_state.replay_playing:
        next_frame = (st.session_state.replay_frame + 1) % num_steps
        if next_frame != 0:
            st.session_state.replay_frame = next_frame
            time.sleep(0.2 / speed)
            st.rerun()
        else:
            st.session_state.replay_playing = False
