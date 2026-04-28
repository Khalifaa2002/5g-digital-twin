"""
Telecom-grade KPI Engine
Implements Nokia/Ericsson-style NOC dashboard KPIs
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Tuple


# ============================================================================
# GLOBAL KPIs
# ============================================================================

def compute_network_health_score(sim_data: Dict) -> float:
    """
    Composite Network Health Score (0-100).
    
    Components:
        - SINR quality (30%)
        - Fairness (20%)
        - SLA compliance (25%)
        - Handover stability (15%)
        - Throughput efficiency (10%)
    """
    sinrs = sim_data['sinrs']
    throughputs = sim_data['throughputs']
    user_sinr_history = sim_data['user_sinr_history']
    serving_bs_history = sim_data['serving_bs_history']
    
    if len(sinrs) == 0:
        return 50.0
    
    # SINR quality score (0-100): map [-10, 30] dB -> [0, 100]
    avg_sinr = np.mean(sinrs)
    sinr_score = np.clip((avg_sinr + 10) / 40 * 100, 0, 100)
    
    # Fairness score (Jain Index * 100)
    last_sinrs = user_sinr_history[-1] if len(user_sinr_history) > 0 else np.array([1])
    jain = compute_jain_index(last_sinrs)
    fairness_score = jain * 100
    
    # SLA compliance (% users above threshold, default 0 dB)
    sla_threshold = st.session_state.get('sim_params', {}).get('sinr_threshold', 0)
    if len(last_sinrs) > 0:
        sla_compliance = 100 * np.mean(last_sinrs >= sla_threshold)
    else:
        sla_compliance = 100.0
    
    # Handover stability (lower rate = higher score)
    num_handovers, _ = compute_handover_rate(serving_bs_history)
    num_steps = len(sinrs)
    num_users = sim_data['num_users']
    ho_rate_norm = num_handovers / max(num_steps * num_users, 1)
    ho_score = max(0, 100 - ho_rate_norm * 5000)  # Penalize high HO rates
    
    # Throughput efficiency (normalized to 2000 Mbps target)
    avg_tp = np.mean(throughputs)
    tp_score = min(100, avg_tp / 2000 * 100)
    
    health_score = (
        0.30 * sinr_score +
        0.20 * fairness_score +
        0.25 * sla_compliance +
        0.15 * ho_score +
        0.10 * tp_score
    )
    
    return float(np.clip(health_score, 0, 100))


def compute_spectral_efficiency(sim_data: Dict) -> float:
    """
    Spectral Efficiency in bits/s/Hz.
    Approximated from average throughput and assumed 100 MHz bandwidth.
    """
    avg_throughput = np.mean(sim_data['throughputs'])  # Mbps
    bandwidth_mhz = 100.0  # Assume 100 MHz NR carrier
    se = avg_throughput / bandwidth_mhz  # bits/s/Hz
    return float(se)


def compute_energy_efficiency(sim_data: Dict) -> float:
    """
    Energy Efficiency in bits/joule.
    Approximate: total bits / total power consumption.
    Assume each BS consumes 5W (37 dBm) + 50W baseband.
    """
    avg_throughput_mbps = np.mean(sim_data['throughputs'])
    num_bs = sim_data['num_bs']
    
    # Total power in Watts (RF + baseband per BS)
    total_power_w = num_bs * 55.0  # 5W RF + 50W baseband
    
    # Throughput in bits/s
    throughput_bps = avg_throughput_mbps * 1e6
    
    # Energy efficiency: bits per joule (W = J/s, so bits/W = bits/J)
    ee = throughput_bps / total_power_w if total_power_w > 0 else 0
    return float(ee)


def compute_user_satisfaction_index(sim_data: Dict) -> float:
    """
    User Satisfaction Index (0-100).
    Based on QoS fulfillment per slice.
    """
    user_sinr_history = sim_data['user_sinr_history']
    if len(user_sinr_history) == 0:
        return 50.0
    
    last_sinrs = user_sinr_history[-1]
    
    # Satisfaction based on SINR thresholds per slice (simplified)
    # eMBB: need > 5 dB, URLLC: > 10 dB, mMTC: > -5 dB
    satisfaction_scores = []
    
    # We don't have slice info per user in sim_data, use general thresholds
    for sinr in last_sinrs:
        if sinr >= 10:
            satisfaction_scores.append(100)
        elif sinr >= 5:
            satisfaction_scores.append(80)
        elif sinr >= 0:
            satisfaction_scores.append(60)
        elif sinr >= -5:
            satisfaction_scores.append(40)
        else:
            satisfaction_scores.append(20)
    
    return float(np.mean(satisfaction_scores))


# ============================================================================
# RADIO KPIs
# ============================================================================

def compute_coverage_probability(sim_data: Dict, threshold_db: float = 0.0) -> float:
    """Probability that a random user has SINR > threshold."""
    user_sinr_history = sim_data['user_sinr_history']
    if len(user_sinr_history) == 0:
        return 0.0
    
    all_sinrs = user_sinr_history.flatten()
    coverage = np.mean(all_sinrs >= threshold_db)
    return float(100 * coverage)


def compute_rsrp_metric(sim_data: Dict) -> float:
    """
    Simulated RSRP-like metric (Reference Signal Received Power).
    Approximated from SINR + typical interference levels.
    """
    avg_sinr = np.mean(sim_data['sinrs'])
    # Approximate RSRP: assume 30 dBm transmit, pathloss ~ 100 dB at median
    # RSRP ≈ SINR + noise_floor + interference_margin
    # Simplified: offset SINR to RSRP range
    rsrp = avg_sinr - 100  # Rough offset to dBm range
    return float(rsrp)


def compute_sinr_percentiles(sim_data: Dict) -> Dict[str, float]:
    """Compute SINR distribution percentiles."""
    user_sinr_history = sim_data['user_sinr_history']
    if len(user_sinr_history) == 0:
        return {'p10': 0, 'p50': 0, 'p90': 0, 'std': 0}
    
    all_sinrs = user_sinr_history.flatten()
    return {
        'p10': float(np.percentile(all_sinrs, 10)),
        'p50': float(np.percentile(all_sinrs, 50)),
        'p90': float(np.percentile(all_sinrs, 90)),
        'std': float(np.std(all_sinrs))
    }


# ============================================================================
# MOBILITY KPIs
# ============================================================================

def compute_handover_success_rate(sim_data: Dict) -> float:
    """
    Simulated handover success rate.
    Assume most handovers succeed; failures increase with high mobility.
    """
    _, ho_rate = compute_handover_rate(sim_data['serving_bs_history'])
    
    # Higher HO rate -> more stress -> lower success rate
    # Base success rate: 98%, penalty for high HO rate
    success_rate = max(70, 98 - ho_rate * 10)
    return float(success_rate)


def compute_ping_pong_handovers(serving_bs_history: List) -> int:
    """
    Detect ping-pong handovers (rapid back-and-forth between same two BSs).
    """
    if len(serving_bs_history) < 3:
        return 0
    
    ping_pong_count = 0
    for i in range(2, len(serving_bs_history)):
        prev2 = serving_bs_history[i-2]
        prev1 = serving_bs_history[i-1]
        curr = serving_bs_history[i]
        
        # Detect pattern: A -> B -> A
        if isinstance(prev2, np.ndarray) and isinstance(prev1, np.ndarray) and isinstance(curr, np.ndarray):
            for u_idx in range(len(prev2)):
                if prev2[u_idx] != prev1[u_idx] and curr[u_idx] == prev2[u_idx]:
                    ping_pong_count += 1
    
    return ping_pong_count


def compute_mobility_robustness_index(sim_data: Dict) -> float:
    """
    Mobility Robustness Index (0-100).
    Combines handover stability, ping-pong rate, and SINR variance.
    """
    serving_bs_history = sim_data['serving_bs_history']
    user_sinr_history = sim_data['user_sinr_history']
    
    if len(serving_bs_history) < 2:
        return 50.0
    
    # Handover rate penalty
    _, ho_rate = compute_handover_rate(serving_bs_history)
    ho_penalty = min(50, ho_rate * 5)
    
    # Ping-pong penalty
    ping_pong = compute_ping_pong_handovers(serving_bs_history)
    pp_penalty = min(30, ping_pong / max(len(serving_bs_history), 1) * 100)
    
    # SINR stability (low variance = good)
    if len(user_sinr_history) > 0:
        sinr_variance = np.var(user_sinr_history.flatten())
        sinr_penalty = min(20, sinr_variance / 10)
    else:
        sinr_penalty = 10
    
    robustness = 100 - ho_penalty - pp_penalty - sinr_penalty
    return float(np.clip(robustness, 0, 100))


# ============================================================================
# QoS KPIs
# ============================================================================

def compute_latency_distribution(sim_data: Dict) -> Dict[str, float]:
    """
    Simulated latency distribution based on SINR.
    Lower SINR -> higher latency (retransmissions, lower MCS).
    """
    user_sinr_history = sim_data['user_sinr_history']
    if len(user_sinr_history) == 0:
        return {'mean': 0, 'p95': 0, 'max': 0}
    
    last_sinrs = user_sinr_history[-1]
    
    # Latency model: base 5ms + penalty for low SINR
    latencies = 5.0 + np.maximum(0, (10 - last_sinrs) * 2)
    
    return {
        'mean': float(np.mean(latencies)),
        'p95': float(np.percentile(latencies, 95)),
        'max': float(np.max(latencies))
    }


def compute_sla_violation_rate(sim_data: Dict, threshold_db: float = 0.0) -> float:
    """Percentage of users below SINR threshold."""
    user_sinr_history = sim_data['user_sinr_history']
    if len(user_sinr_history) == 0:
        return 0.0
    
    last_sinrs = user_sinr_history[-1]
    return float(100 * np.mean(last_sinrs < threshold_db))


def compute_slice_isolation_score(sim_data: Dict) -> float:
    """
    Slice Isolation Score (0-100).
    Measures how well slices are separated in performance.
    High score = distinct performance per slice (good isolation).
    """
    users = sim_data.get('users', [])
    user_capacities = sim_data.get('user_capacities', np.array([]))
    
    if len(users) == 0 or len(user_capacities) == 0:
        return 50.0
    
    # Group capacities by slice
    slice_caps = {}
    for u_idx, user in enumerate(users):
        if u_idx < len(user_capacities):
            stype = getattr(user, 'slice_type', 'eMBB')
            slice_caps.setdefault(stype, []).append(user_capacities[u_idx])
    
    if len(slice_caps) < 2:
        return 50.0
    
    # Compute mean capacity per slice
    slice_means = [np.mean(caps) for caps in slice_caps.values()]
    
    # Isolation score: higher variance between slices = better isolation
    overall_mean = np.mean(slice_means)
    if overall_mean == 0:
        return 50.0
    
    variance = np.var(slice_means)
    isolation = min(100, variance / (overall_mean ** 2 + 1e-6) * 200)
    
    return float(isolation)


# ============================================================================
# FAIRNESS KPIs
# ============================================================================

def compute_jain_index(values: np.ndarray) -> float:
    """
    Compute Jain Fairness Index.
    Jain = (sum(x))^2 / (n * sum(x^2))
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


def compute_resource_entropy(allocations: np.ndarray) -> float:
    """
    Compute resource allocation entropy (0-1).
    Higher entropy = more equal distribution = better fairness.
    """
    if len(allocations) == 0 or np.sum(allocations) == 0:
        return 0.0
    
    # Normalize to probabilities
    p = allocations / np.sum(allocations)
    p = p[p > 0]  # Remove zeros for log
    
    if len(p) == 0:
        return 0.0
    
    # Shannon entropy (normalized by max entropy = log(n))
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(allocations))
    
    return float(np.clip(entropy / max_entropy, 0, 1))


def compute_handover_rate(serving_bs_history: List) -> Tuple[int, float]:
    """
    Compute handovers and handover rate - NUMPY SAFE FIX.
    Returns: (num_handovers, handover_rate %)
    """
    if len(serving_bs_history) < 2:
        return 0, 0.0
    
    handover_count = 0
    for i in range(1, len(serving_bs_history)):
        if isinstance(serving_bs_history[i], np.ndarray) and isinstance(serving_bs_history[i-1], np.ndarray):
            if not np.array_equal(serving_bs_history[i], serving_bs_history[i-1]):
                handover_count += 1
        else:
            if serving_bs_history[i] != serving_bs_history[i-1]:
                handover_count += 1
    
    num_users = len(serving_bs_history[0]) if isinstance(serving_bs_history[0], np.ndarray) else 1
    rate = 100 * handover_count / (len(serving_bs_history) * num_users) if num_users > 0 else 0.0
    
    return int(handover_count), float(rate)


# ============================================================================
# NOC DASHBOARD RENDERER
# ============================================================================

def render_noc_dashboard(metrics: Dict):
    """
    Render Nokia/Ericsson-style NOC KPI dashboard.
    Uses dark theme cards with gradient color indicators.
    """
    st.subheader("🖥️ Network Operations Center (NOC) Dashboard")
    
    st.markdown("""
    <style>
    .noc-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #0f3460;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .noc-title {
        color: #e0e0e0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .noc-value {
        color: #ffffff;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .noc-unit {
        color: #b0b0b0;
        font-size: 13px;
    }
    .noc-indicator-green {
        border-left: 4px solid #00ff88;
    }
    .noc-indicator-orange {
        border-left: 4px solid #ffaa00;
    }
    .noc-indicator-red {
        border-left: 4px solid #ff3366;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Helper: determine color based on value and thresholds
    def indicator_color(value: float, good_threshold: float, warn_threshold: float,
                        higher_is_better: bool = True) -> str:
        if higher_is_better:
            if value >= good_threshold:
                return 'noc-indicator-green'
            elif value >= warn_threshold:
                return 'noc-indicator-orange'
            else:
                return 'noc-indicator-red'
        else:
            if value <= good_threshold:
                return 'noc-indicator-green'
            elif value <= warn_threshold:
                return 'noc-indicator-orange'
            else:
                return 'noc-indicator-red'
    
    def render_card(title: str, value: str, unit: str, color_class: str):
        st.markdown(f"""
        <div class="noc-card {color_class}">
            <div class="noc-title">{title}</div>
            <div class="noc-value">{value}</div>
            <div class="noc-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ---- GLOBAL KPIs ----
    st.markdown("#### 🌐 Global Network Health")
    g1, g2, g3, g4 = st.columns(4)
    
    with g1:
        hs = metrics.get('health_score', 0)
        render_card("Health Score", f"{hs:.1f}", "/ 100", indicator_color(hs, 80, 60))
    
    with g2:
        se = metrics.get('spectral_efficiency', 0)
        render_card("Spectral Efficiency", f"{se:.2f}", "bits/s/Hz",
                    indicator_color(se, 5, 2))
    
    with g3:
        ee = metrics.get('energy_efficiency', 0)
        ee_m = ee / 1e6  # Convert to Mbits/J
        render_card("Energy Efficiency", f"{ee_m:.2f}", "Mbit/J",
                    indicator_color(ee_m, 5, 2))
    
    with g4:
        usi = metrics.get('user_satisfaction', 0)
        render_card("User Satisfaction", f"{usi:.1f}", "%",
                    indicator_color(usi, 80, 60))
    
    st.markdown("---")
    
    # ---- RADIO KPIs ----
    st.markdown("#### 📡 Radio Performance")
    r1, r2, r3, r4 = st.columns(4)
    
    with r1:
        cp = metrics.get('coverage_probability', 0)
        render_card("Coverage Probability", f"{cp:.1f}", "%",
                    indicator_color(cp, 95, 80))
    
    with r2:
        rsrp = metrics.get('rsrp', -100)
        render_card("RSRP (est.)", f"{rsrp:.1f}", "dBm",
                    indicator_color(rsrp, -90, -100, higher_is_better=False))
    
    with r3:
        sinr_p50 = metrics.get('sinr_p50', 0)
        render_card("SINR Median", f"{sinr_p50:.1f}", "dB",
                    indicator_color(sinr_p50, 10, 5))
    
    with r4:
        sinr_std = metrics.get('sinr_std', 0)
        render_card("SINR Std Dev", f"{sinr_std:.1f}", "dB",
                    indicator_color(sinr_std, 3, 5, higher_is_better=False))
    
    st.markdown("---")
    
    # ---- MOBILITY KPIs ----
    st.markdown("#### 🔄 Mobility & Handover")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        ho_sr = metrics.get('handover_success_rate', 0)
        render_card("HO Success Rate", f"{ho_sr:.1f}", "%",
                    indicator_color(ho_sr, 95, 85))
    
    with m2:
        pp = metrics.get('ping_pong_handovers', 0)
        render_card("Ping-Pong HO", f"{pp}", "events",
                    indicator_color(pp, 5, 15, higher_is_better=False))
    
    with m3:
        mri = metrics.get('mobility_robustness', 0)
        render_card("Mobility Robustness", f"{mri:.1f}", "/ 100",
                    indicator_color(mri, 80, 60))
    
    with m4:
        num_ho = metrics.get('num_handovers', 0)
        render_card("Total Handovers", f"{num_ho}", "events",
                    indicator_color(num_ho, 50, 100, higher_is_better=False))
    
    st.markdown("---")
    
    # ---- QoS KPIs ----
    st.markdown("#### ⚠️ Quality of Service")
    q1, q2, q3, q4 = st.columns(4)
    
    with q1:
        sla = metrics.get('sla_violation_rate', 0)
        render_card("SLA Violation Rate", f"{sla:.1f}", "%",
                    indicator_color(sla, 2, 5, higher_is_better=False))
    
    with q2:
        lat_p95 = metrics.get('latency_p95', 0)
        render_card("Latency P95", f"{lat_p95:.1f}", "ms",
                    indicator_color(lat_p95, 10, 20, higher_is_better=False))
    
    with q3:
        iso = metrics.get('slice_isolation', 0)
        render_card("Slice Isolation", f"{iso:.1f}", "/ 100",
                    indicator_color(iso, 70, 50))
    
    with q4:
        lat_mean = metrics.get('latency_mean', 0)
        render_card("Avg Latency", f"{lat_mean:.1f}", "ms",
                    indicator_color(lat_mean, 5, 10, higher_is_better=False))
    
    st.markdown("---")
    
    # ---- FAIRNESS KPIs ----
    st.markdown("#### 🎯 Fairness")
    f1, f2, f3 = st.columns(3)
    
    with f1:
        jain = metrics.get('jain_index', 0)
        render_card("Jain Fairness Index", f"{jain:.3f}", "[0-1]",
                    indicator_color(jain, 0.85, 0.70))
    
    with f2:
        ent = metrics.get('resource_entropy', 0)
        render_card("Resource Entropy", f"{ent:.3f}", "[0-1]",
                    indicator_color(ent, 0.9, 0.7))
    
# ============================================================================
# V4 EXTRA TELECOM KPIs
# ============================================================================

def compute_call_drop_risk_score(sim_data: Dict) -> float:
    """
    Call Drop Risk Score (0-100).
    Higher = more risk of dropped calls.
    Based on: low SINR users + high handover rate + high latency.
    """
    user_sinr_history = sim_data.get('user_sinr_history', np.array([]))
    serving_bs_history = sim_data.get('serving_bs_history', [])
    if len(user_sinr_history) == 0 or len(serving_bs_history) < 2:
        return 0.0

    last_sinrs = user_sinr_history[-1]
    # % of users with very poor SINR (< -5 dB)
    poor_sinr_pct = 100 * np.mean(last_sinrs < -5)

    # Handover stress
    _, ho_rate = compute_handover_rate(serving_bs_history)
    ho_stress = min(50, ho_rate * 3)

    # Latency stress
    lat = compute_latency_distribution(sim_data)
    lat_stress = min(30, lat.get('mean', 0) / 2)

    risk = poor_sinr_pct * 0.4 + ho_stress * 0.4 + lat_stress * 0.2
    return float(np.clip(risk, 0, 100))


def compute_cell_load_imbalance_index(sim_data: Dict) -> float:
    """
    Cell Load Imbalance Index (0-100).
    0 = perfectly balanced, 100 = severely imbalanced.
    Based on variance of users per BS.
    """
    serving_bs_history = sim_data.get('serving_bs_history', [])
    num_bs = sim_data.get('num_bs', 1)
    if len(serving_bs_history) == 0 or num_bs <= 1:
        return 0.0

    last_serving = serving_bs_history[-1]
    if isinstance(last_serving, np.ndarray):
        counts = np.bincount(last_serving.astype(int), minlength=num_bs)
    else:
        counts = np.ones(num_bs)

    avg_load = np.mean(counts)
    if avg_load == 0:
        return 0.0

    # Coefficient of variation * 100
    cv = np.std(counts) / avg_load
    return float(np.clip(cv * 100, 0, 100))


def compute_energy_per_mbps(sim_data: Dict) -> float:
    """
    Energy consumption per Mbps (Joules/Mbit).
    Lower = more efficient.
    """
    avg_throughput_mbps = np.mean(sim_data.get('throughputs', [0]))
    num_bs = sim_data.get('num_bs', 1)
    total_power_w = num_bs * 55.0
    if avg_throughput_mbps <= 0:
        return 999.0
    # Joules per Mbit = Watts / (Mbps * 1e6) * 1e6 = Watts / Mbps
    return float(total_power_w / avg_throughput_mbps)


def compute_latency_sla_score(sim_data: Dict) -> float:
    """
    Latency SLA Score (0-100).
    Measures compliance with slice latency budgets.
    """
    lat = compute_latency_distribution(sim_data)
    mean_lat = lat.get('mean', 0)
    p95_lat = lat.get('p95', 0)

    # Score based on URLLC budget (1ms) and eMBB budget (20ms)
    # Weighted: 60% P95, 40% mean
    score = 100 - (0.6 * min(100, p95_lat * 5) + 0.4 * min(100, mean_lat * 3))
    return float(np.clip(score, 0, 100))


def compute_mos_like_score(sim_data: Dict) -> float:
    """
    Mean Opinion Score (MOS) - like User Experience Score (1-5).
    Based on SINR, latency, and throughput.
    """
    sinrs = sim_data.get('sinrs', np.array([]))
    lat = compute_latency_distribution(sim_data)
    throughputs = sim_data.get('throughputs', np.array([]))

    if len(sinrs) == 0:
        return 3.0

    avg_sinr = np.mean(sinrs)
    mean_lat = lat.get('mean', 20)
    avg_tp = np.mean(throughputs) if len(throughputs) > 0 else 0

    # SINR component (1-5): map [-10, 30] -> [1, 5]
    sinr_mos = np.clip((avg_sinr + 10) / 40 * 4 + 1, 1, 5)

    # Latency component: lower latency = higher MOS
    lat_mos = np.clip(5 - mean_lat / 5, 1, 5)

    # Throughput component: higher = better
    tp_mos = np.clip(avg_tp / 500, 1, 5)

    # Weighted average
    mos = 0.4 * sinr_mos + 0.35 * lat_mos + 0.25 * tp_mos
    return float(np.clip(mos, 1, 5))


