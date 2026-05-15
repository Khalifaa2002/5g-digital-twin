"""
README.md - 5G NR Network Simulator v2.1
Professional simulation project based on 3GPP standards with Advanced Analytics
"""

# 5G NR / LTE Digital Twin Network Simulator v2.1

## 📋 Project Overview

This project is a **production-grade 5G NR & LTE network simulator** based on **3GPP TR 38.901** channel model specifications with advanced analytics, fairness metrics, and predictive capabilities. Built for research, academic projects, and telecommunications engineering workflows.

**🎯 Core Features (v2.1):**
- ✅ 3GPP TR 38.901 channel model with LOS/NLOS probability
- ✅ Small-scale fading: Rayleigh (NLOS) & Rician (LOS) models
- ✅ Massive MIMO beamforming with 64-antenna arrays
- ✅ Network slicing: eMBB, URLLC, mMTC with QoS scheduling
- ✅ Random waypoint mobility model with handover logic
- ✅ Interactive Streamlit dashboard with real-time visualization
- ✅ NumPy-based architecture (lightweight, no heavy ML dependencies)
- ✅ Reproducible simulations with random seeds

**🔥 v2.1 Enhancements:**
- ✅ **SINR Heatmap** - Real-time radio coverage visualization
- ✅ **Fairness KPI** - Jain Index for throughput fairness (0-1 scale)
- ✅ **Handover Tracking** - Numpy-safe detection with rate calculation
- ✅ **Predictive SINR** - Moving average forecasting with confidence bands
- ✅ **SLA Monitoring** - Violation rate tracking with alerts
- ✅ **Network Mode** - 5G vs LTE comparison (switchable)
- ✅ **Export Data** - JSON export with comprehensive metadata
- ✅ **10 Dashboard Tabs** - Summary KPIs, Network Map, Metrics, Slicing, Export, Radio Map, AI Prediction, City Scale, NOC, AI SON
- ✅ **Bug Fix** - Resolved numpy array comparison error (ValueError: truth value ambiguity)

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download project
cd 5G_Project

# Install dependencies
pip install -r requirements.txt
```

### Run Dashboard

```bash
# Start Streamlit dashboard
streamlit run dashboard.py

# Open browser to http://localhost:8501
```

### Run Tests

```bash
# Unit tests
python test_dashboard.py

# Scenario validation
python test_scenarios.py
```

---

## 📊 Dashboard Features

### Tab 1: Summary KPIs
- 📊 **Fairness Index** (Jain) - Measures throughput fairness [0=unfair, 1=perfect]
- ⚠️ **SLA Violations** - % of users below SINR threshold
- 📡 **Avg Throughput** - Network-wide capacity (Mbps)
- 📶 **Avg SINR** - Network-wide signal quality (dB)
- 🔄 **Handovers** - Total count with rate (HOs/TTI)
- ⏱️ **Simulation Time** - Total duration (ms)
- 📈 **Fairness Evolution** - Graph of Jain Index over time
- 📉 **SLA Evolution** - Graph of violation rate over time

### Tab 2: Network Map
- 🗺️ **SINR Heatmap** - Color-coded coverage (Red=bad, Yellow=medium, Green=excellent)
- 📍 **Base Station Overlay** - BS positions with coverage circles
- 👥 **User Distribution** - Color-coded by slice type (eMBB/URLLC/mMTC)
- 📊 **Slice Pie Chart** - Resource allocation by slice
- 📈 **SINR Statistics** - Min/Q1/Median/Q3/Max/Std Dev

### Tab 3: Advanced Metrics
- 🧠 **SINR Prediction** - Actual vs Predicted (Moving Average)
- 📊 **Throughput Evolution** - Capacity over time
- 🔄 **Handover Timeline** - Cumulative HOs per timestamp
- 💫 **Fairness Trends** - Jain Index evolution with thresholds
- 📊 **Per-User Throughput** - Individual user allocation bar chart
- 📊 **Slice Throughput Distribution** - Box plots by slice type

### Tab 4: Network Slicing (5G only)
- eMBB Performance - Capacity, latency, BLER
- URLLC Performance - Priority, reliability, latency metrics
- mMTC Performance - Energy efficiency, coverage probability

### Tab 5: Export
- 📥 **JSON Download** - All metrics with metadata
- 📊 **Full Data View** - Expandable JSON structure

### Tab 6: Radio Map (Optional)
- 🎥 **Animated Radio Map** - Real-time network visualization
- 🎬 **Replay Mode** - Frame-by-frame simulation playback

### Tab 7: AI Prediction (Optional)
- 🧠 **SINR Forecasting** - 20-step ahead prediction
- 📊 **Prediction Metrics** - MAE, RMSE, MAPE
- 📈 **Confidence Bands** - Uncertainty quantification

### Tab 8: City Scale (Optional)
- 🏙️ **Urban Zones** - Downtown, Suburban, Rural, Industrial
- 🏢 **Building Blockage** - LOS/NLOS probability map
- 🔥 **Traffic Hotspots** - High-demand area visualization

### Tab 9: NOC Dashboard (Optional)
- 🏥 **Health Score** - Overall network status (0-100)
- 📊 **KPI Cards** - Color-coded metrics (Green/Orange/Red)
- 📡 **Coverage & RSRP** - Cell-level statistics
- 🤝 **Handover Metrics** - Success rate, ping-pongs, robustness
- 📈 **Slice Isolation** - Network slice separation quality

### Tab 10: AI SON Control Center (Optional)
- 🤖 **SON Recommendations** - Auto-optimization suggestions
- 🕳️ **Coverage Holes** - Detection and healing proposals
- ⚡ **Interference Reduction** - CoMP/ICIC recommendations
- ⚖️ **Load Balancing** - User distribution optimization
- 🔋 **Energy Savings** - Power consumption reduction hints

---

## 🔄 Simulation Pipeline

```
┌─────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                   │
│   - Network setup (BSs, users, slices)             │
│   - Random seed for reproducibility                │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 2. TIME-LOOP SIMULATION (TTI-based, 1ms per step)  │
│   ├─ Update User Positions (Mobility)              │
│   ├─ Compute Channel State (3GPP)                  │
│   │   ├─ Path Loss (3GPP formulas)                 │
│   │   ├─ Shadowing (Lognormal)                     │
│   │   └─ Fading (Rayleigh or Rician)              │
│   ├─ Calculate SINR per User                        │
│   ├─ Handover Decisions (Hysteresis-based)         │
│   ├─ Resource Scheduling (Network Slicing)         │
│   │   ├─ eMBB: Proportional Fair                   │
│   │   ├─ URLLC: Priority scheduling                │
│   │   └─ mMTC: Best-effort                         │
│   ├─ Compute User Throughputs                      │
│   └─ Collect Metrics                               │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│ 3. VISUALIZATION & ANALYSIS                         │
│   - Network topology map                            │
│   - SINR distribution histogram                     │
│   - Throughput evolution                            │
│   - Per-slice performance metrics                   │
│   - QoS compliance analysis                         │
└─────────────────────────────────────────────────────┘
```

---

## 📡 Channel Model (3GPP TR 38.901)

### Path Loss Computation

The simulator implements three scenarios:

#### UMi (Urban Micro)
- **LOS:** PL = 18.6·log₁₀(d) + 46.85 + 20·log₁₀(fc)
- **NLOS:** PL = 36.7·log₁₀(d) + 32.4 + 20·log₁₀(fc) - 30

#### UMa (Urban Macro)
- **LOS:** PL = 28.0·log₁₀(d) + 22.0 + 20·log₁₀(fc)
- **NLOS:** PL = 13.54·log₁₀(d) + 27.23 + 20·log₁₀(fc) - 32.4

#### RMa (Rural Macro)
- **LOS:** PL = 10.0·log₁₀(d) + 49.24 + 20·log₁₀(fc)
- **NLOS:** PL = 15.3·log₁₀(d) + 37.08 + 20·log₁₀(fc) - 31.5

### LOS/NLOS Probability

```
P(LOS) = min(1, C/d) × exp(-d/D)
```

where C and D depend on scenario (3GPP formulas).

### Small-Scale Fading

- **Rayleigh (NLOS):** h ~ CN(0, 1) - suitable for many scatterers
- **Rician (LOS):** K-factor determines LOS power ratio

### Shadowing

Lognormal model: X_dB ~ N(0, σ) where σ ∈ {4, 6, 8} dB depending on scenario.

---

## 📊 Network Slicing (3GPP TS 26.501)

The simulator implements three network slices:

### eMBB (Enhanced Mobile Broadband)
- **Use case:** High-capacity services (streaming, web browsing)
- **Latency budget:** 20 ms
- **Reliability:** 99%
- **Target data rate:** 100 Mbps
- **Priority:** Medium

### URLLC (Ultra-Reliable Low-Latency Communications)
- **Use case:** Critical services (autonomous driving, industrial control)
- **Latency budget:** 1 ms
- **Reliability:** 99.999%
- **Target data rate:** 10 Mbps
- **Priority:** High (preempts eMBB/mMTC)

### mMTC (Massive Machine-Type Communications)
- **Use case:** IoT, sensor networks
- **Latency budget:** 1000 ms
- **Reliability:** 90%
- **Target data rate:** 0.5 Mbps
- **Priority:** Low

### Resource Allocation Strategies

**1. Proportional Fair (PF):**
- Balances throughput maximization and fairness
- Weight = Priority / Average_Rate

**2. URLLC First:**
- Priority-based: URLLC > eMBB > mMTC
- Ensures QoS for critical services

**3. Slice Isolation:**
- Fixed resource fractions per slice
- Default: eMBB=60%, URLLC=30%, mMTC=10%

---

## 🚗 Mobility Models

### Random Waypoint
- Users move toward random waypoints
- Variable velocity: 0.5-5.0 m/s
- Pause probability at waypoints

### Vehicular (V2X)
- High-speed movement: ~25 m/s (90 km/h)
- Lane-like behavior with occasional direction changes
- For highway scenarios

### Stationary
- No movement (baseline for testing)

---

## 📈 MIMO & Beamforming

### Massive MIMO
- Base stations with 64 antennas (configurable)
- Uniform Linear Array (ULA) steering vectors
- Maximum Ratio Combining (MRC) at receiver

### Beamforming Gain
```
G_beam(θ) = |a_beam^H · a_user|² × num_antennas²
```

where a = steering vector at angle θ.

### Capacity Models
- **Shannon Capacity:** C = B·log₂(1 + SINR) (bits/s)
- **Spectral Efficiency:** SE = log₂(1 + SINR) (bits/s/Hz)
- **MIMO Rank:** min(num_tx, num_rx)

---

## 🎮 Usage Guide

### 1. Basic Usage (Python)

```python
from simulation import NetworkSimulation

# Create simulator
sim = NetworkSimulation(
    simulation_time_ms=1000,  # 1 second
    num_users=50,
    num_bs=10,
    scenario='UMi',  # Urban Micro
    seed=42  # Reproducible
)

# Run simulation
metrics_history = sim.run()

# Get results
stats = sim.get_summary_statistics()
print(f"Average Throughput: {stats['avg_throughput_mbps']:.1f} Mbps")
print(f"Average SINR: {stats['avg_sinr_db']:.1f} dB")
```

### 2. Interactive Dashboard

```bash
cd 5G_Project
pip install streamlit matplotlib numpy

streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser.

### 3. Module-Specific Tests

Each module includes test code:

```bash
python channel.py      # Test 3GPP channel model
python mimo.py        # Test beamforming & MIMO
python scheduler.py   # Test scheduling algorithms
python mobility.py    # Test mobility models
python simulation.py  # Full simulation test
```

---

## 📊 Key Metrics

### Per-User Metrics
- **SINR (dB):** Signal-to-Interference-plus-Noise Ratio
- **Capacity (Mbps):** Allocated bandwidth × Spectral Efficiency
- **Latency (ms):** Queuing + transmission delay
- **BLER:** Block Error Rate (depends on channel quality)

### Network-Wide Metrics
- **Total Throughput:** Sum of all user capacities
- **Average SINR:** Mean SINR across all users
- **Energy Efficiency:** Throughput per watt
- **QoS Satisfaction:** Percentage meeting slice requirements

### Per-Slice Metrics
- Number of active users
- Total and per-user capacity
- Average latency
- Block error rate
- QoS compliance

---

## 🔧 Advanced Configuration

### Channel Model Parameters

**In `channel.py`:**
```python
SCENARIO_PARAMS = {
    'UMi': {
        'los_a': 18.6,           # Path loss slope (LOS)
        'nlos_a': 36.7,          # Path loss slope (NLOS)
        'breakpoint': 200,       # Breakpoint distance (m)
        'shadowing_std': 4.0,    # Shadowing std dev (dB)
    },
    # ... UMa, RMa parameters
}
```

### Base Station Configuration

**In `simulation.py`:**
```python
bs = BaseStationConfig(
    bs_id=0,
    position=np.array([0, 0]),
    tx_power_dbm=37,      # 5W
    num_antennas=64,      # Massive MIMO
    bandwidth_mhz=20,     # 20 MHz NR subcarrier spacing
    frequency_ghz=3.5,    # Mid-band 5G
    max_users=100
)
```

### Scheduling Configuration

**In `simulation.py`, `step()` method:**
```python
slice_fractions = {
    'eMBB': 0.6,
    'URLLC': 0.3,
    'mMTC': 0.1
}
```

---

## 📚 References

1. **3GPP TS 38.901:** Channel Model for 5G NR
   - Path loss, shadowing, fading models
   - LOS/NLOS probability

2. **3GPP TS 26.501:** Network Slicing Architecture
   - Service requirements per slice
   - Resource allocation strategies

3. **3GPP TS 38.212:** Physical layer procedures
   - Scheduling and resource allocation

4. **Modern Wireless Communications** (References)
   - MIMO theory and massive beamforming
   - Capacity computation

---

## 💡 Project Ideas & Extensions

### Level 1: Basic
- [ ] Add handover metrics (ping-pong reduction)
- [ ] Implement practical modulation & coding schemes (MCS)
- [ ] Add pathloss calibration from real measurements

### Level 2: Intermediate
- [ ] Multi-connectivity / network coding
- [ ] Uplink simulation (reverse link)
- [ ] Blockage effects for mmWave (60/73 GHz)
- [ ] QoE metrics (video stalling, etc.)

### Level 3: Advanced
- [ ] Machine Learning for resource allocation
- [ ] Network optimization (placement, beam management)
- [ ] RAN slicing with virtualization
- [ ] Full-duplex communication modeling

---

## 🐛 Troubleshooting

### Dashboard won't start
```bash
pip install --upgrade streamlit matplotlib numpy
streamlit run dashboard.py --logger.level=debug
```

### Simulation too slow
- Reduce `simulation_time_ms`
- Reduce `num_users` or `num_bs`
- Increase `dt_ms` in `step()` (less accuracy but faster)

### Non-reproducible results
- Always set `seed` parameter
- Check for floating-point precision issues

---

## 📄 License & Usage

This project is for **educational and research purposes**. 
Suitable for:
- University projects (networking, wireless communications)
- Internship presentations
- Research papers (cite 3GPP standards)
- Portfolio projects

---

## ✍️ Author Notes

**Design Philosophy:**
- Clean, modular Python code (no monolithic scripts)
- NumPy-based (minimal dependencies)
- Comments explain telecom concepts
- Based on real 3GPP standards
- Professional-grade documentation

**Key Innovations:**
- Time-stepped TTI-based simulation
- Realistic 3GPP channel modeling
- Network slicing with QoS scheduling
- Interactive visualization
- Reproducible research

---

📡 1. README professionnel
architecture 5G Digital Twin
AI / ML pipeline
simulation workflow
📊 2. Ajout visuels
dashboard screenshots
KPI graphs
radio map simulation
🤖 3. Section AI
KNN / CNN / supervised learning
TensorFlow Lite edge deployment
 
