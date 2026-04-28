"""
5G NR Network Simulation Engine
Main simulation pipeline with time-stepping, SINR computation, and scheduling
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from channel import multipath_channel, los_probability, FC_DEFAULT
from mimo import omnidirectional_beamforming_gain, shannon_capacity
from scheduler import UserQoSProfile, schedule_slice_isolation, compute_qos_metrics
from mobility import RandomWaypointMobility, HandoverController


# ============================================================================
# SIMULATION DATA STRUCTURES
# ============================================================================

@dataclass
class BaseStationConfig:
    """Base station configuration."""
    bs_id: int
    position: np.ndarray  # [x, y]
    tx_power_dbm: float = 37  # 5W typical for 5G
    num_antennas: int = 64
    bandwidth_mhz: float = 20
    frequency_ghz: float = 3.5
    max_users: int = 100
    is_5g: bool = True  # True for 5G NR, False for LTE
    
    def __post_init__(self):
        self.tx_power_linear = 10 ** ((self.tx_power_dbm - 30) / 10)  # Convert to Watts


@dataclass
class SimulationMetrics:
    """Per-step simulation metrics."""
    timestamp: float = 0.0
    
    # Per-slice metrics
    slice_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    # User metrics
    user_sinrs: np.ndarray = field(default_factory=lambda: np.array([]))
    user_capacities: np.ndarray = field(default_factory=lambda: np.array([]))
    user_latencies: np.ndarray = field(default_factory=lambda: np.array([]))
    user_serving_bs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Network-wide metrics
    total_throughput_mbps: float = 0.0
    avg_sinr_db: float = 0.0
    energy_efficiency: float = 0.0  # bits/joule


# ============================================================================
# MAIN SIMULATION ENGINE
# ============================================================================

class NetworkSimulation:
    """5G NR Network Simulator."""
    
    def __init__(self, simulation_time_ms: float = 1000,
                 num_users: int = 50,
                 num_bs: int = 10,
                 scenario: str = 'UMi',
                 seed: int = None):
        """
        Args:
            simulation_time_ms: Total simulation time in milliseconds
            num_users: Number of users
            num_bs: Number of base stations
            scenario: 'UMi', 'UMa', or 'RMa'
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.simulation_time_ms = simulation_time_ms
        self.num_users = num_users
        self.num_bs = num_bs
        self.scenario = scenario
        
        # Simulation area (1000m x 1000m urban area)
        self.bounds = ((-500, 500), (-500, 500))
        
        # Initialize components
        self.base_stations = self._init_base_stations()
        self.users = self._init_users()
        self.mobility_model = RandomWaypointMobility(
            self.bounds, min_velocity=0.5, max_velocity=5.0)
        self.handover_controller = HandoverController()
        
        # State
        self.user_positions = self._random_positions(num_users)
        self.user_serving_bs = np.zeros(num_users, dtype=int)
        self.metrics_history = []
        
        self.step_count = 0
        self.time_ms = 0.0
    
    def _init_base_stations(self) -> List[BaseStationConfig]:
        """Initialize base stations."""
        bs_list = []
        
        # Create grid of base stations
        grid_size = int(np.ceil(np.sqrt(self.num_bs)))
        spacing = 800 / (grid_size + 1)
        
        bs_id = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if bs_id >= self.num_bs:
                    break
                
                x = -400 + (i + 1) * spacing
                y = -400 + (j + 1) * spacing
                
                bs = BaseStationConfig(
                    bs_id=bs_id,
                    position=np.array([x, y]),
                    tx_power_dbm=37,
                    num_antennas=64,
                    frequency_ghz=FC_DEFAULT,
                    is_5g=True
                )
                bs_list.append(bs)
                bs_id += 1
            
            if bs_id >= self.num_bs:
                break
        
        return bs_list
    
    def _init_users(self) -> List[UserQoSProfile]:
        """Initialize users with random slice assignments."""
        users = []
        
        # Distribute users across slices
        num_embbb = int(0.5 * self.num_users)
        num_urllc = int(0.3 * self.num_users)
        num_mmtc = self.num_users - num_embbb - num_urllc
        
        user_id = 0
        
        # eMBB users
        for _ in range(num_embbb):
            users.append(UserQoSProfile(user_id, 'eMBB', qos_class=2))
            user_id += 1
        
        # URLLC users
        for _ in range(num_urllc):
            users.append(UserQoSProfile(user_id, 'URLLC', qos_class=3))
            user_id += 1
        
        # mMTC users
        for _ in range(num_mmtc):
            users.append(UserQoSProfile(user_id, 'mMTC', qos_class=1))
            user_id += 1
        
        return users
    
    def _random_positions(self, num_users: int) -> np.ndarray:
        """Generate random user positions."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.random.uniform(
            [x_min, y_min], [x_max, y_max], (num_users, 2))
    
    # ========================================================================
    # CHANNEL COMPUTATION
    # ========================================================================
    
    def compute_sinr_matrix(self) -> np.ndarray:
        """
        Compute SINR from all BSs to all users.
        
        Returns:
            sinr_matrix: shape (num_users, num_bs), SINR in linear scale
        """
        sinr_matrix = np.zeros((self.num_users, self.num_bs))
        
        for u_idx, user_pos in enumerate(self.user_positions):
            # Compute received power from all BSs
            received_powers = np.zeros(self.num_bs)
            
            for bs in self.base_stations:
                distance = np.linalg.norm(user_pos - bs.position)
                
                # Channel computation (pathloss + fading + shadowing)
                channel_gain, _ = multipath_channel(
                    distance, bs.frequency_ghz, self.scenario)
                
                # Beamforming gain
                bf_gain = omnidirectional_beamforming_gain(
                    user_pos, bs.position, bs.num_antennas)
                
                # Received power
                rx_power = bs.tx_power_linear * channel_gain * bf_gain
                received_powers[bs.bs_id] = rx_power
            
            # SINR computation (best serving BS)
            best_bs = np.argmax(received_powers)
            signal_power = received_powers[best_bs]
            
            # Interference from other BSs
            interference_power = np.sum(received_powers) - signal_power
            
            # SINR
            sinr = signal_power / (interference_power + 1e-12)
            
            # Broadcast SINR to all BSs for handover
            for bs_idx in range(self.num_bs):
                sinr_matrix[u_idx, bs_idx] = received_powers[bs_idx] / \
                                              (np.sum(received_powers) - received_powers[bs_idx] + 1e-12)
        
        return sinr_matrix
    
    # ========================================================================
    # SCHEDULING & THROUGHPUT
    # ========================================================================
    
    def schedule_users(self, total_capacity_mbps: float = 1000.0) -> Dict[int, float]:
        """
        Schedule users using network slicing.
        
        Args:
            total_capacity_mbps: Total available capacity
        
        Returns:
            Allocations: user_id -> capacity in Mbps
        """
        return schedule_slice_isolation(
            self.users, total_capacity_mbps,
            slice_fractions={'eMBB': 0.6, 'URLLC': 0.3, 'mMTC': 0.1})
    
    def compute_throughputs(self, sinr_matrix: np.ndarray,
                           allocations: Dict[int, float]) -> np.ndarray:
        """
        Compute throughput for each user.
        
        Args:
            sinr_matrix: SINR from all BSs
            allocations: Resource allocation per user
        
        Returns:
            throughputs: array of throughput in Mbps
        """
        throughputs = np.zeros(self.num_users)
        
        for u_idx in range(self.num_users):
            # Get best BS for this user
            best_bs = np.argmax(sinr_matrix[u_idx])
            sinr_best = sinr_matrix[u_idx, best_bs]
            
            # Allocated capacity
            allocated = allocations.get(self.users[u_idx].user_id, 0)
            
            # Compute spectral efficiency from SINR
            se = np.log2(1 + sinr_best) * 0.9  # 90% of Shannon
            
            # Throughput = SE * allocated_bandwidth
            # Assuming equal bandwidth share
            bandwidth_share = allocated / 1000.0  # normalized to 1GHz
            throughput = se * bandwidth_share * 1000  # in Mbps
            
            throughputs[u_idx] = max(0, throughput)
        
        return throughputs
    
    # ========================================================================
    # SIMULATION STEP
    # ========================================================================
    
    def step(self, dt_ms: float = 1.0) -> SimulationMetrics:
        """
        Execute one simulation step.
        
        Args:
            dt_ms: Time step in milliseconds
        
        Returns:
            Metrics for this step
        """
        # Update time
        self.time_ms += dt_ms
        self.step_count += 1
        
        # 1. Update user positions (mobility)
        for u_idx in range(self.num_users):
            self.user_positions[u_idx] = self.mobility_model.update(
                u_idx, self.user_positions[u_idx], dt=dt_ms/1000)
        
        # 2. Compute SINR matrix
        sinr_matrix = self.compute_sinr_matrix()
        
        # 3. Handover decisions
        for u_idx in range(self.num_users):
            serving_bs = self.handover_controller.compute_handover_decision(
                u_idx, sinr_matrix[u_idx], self.user_serving_bs[u_idx])
            self.user_serving_bs[u_idx] = serving_bs
        
        # 4. Resource scheduling
        allocations = self.schedule_users(total_capacity_mbps=1000.0)
        
        # 5. Compute throughputs
        throughputs = self.compute_throughputs(sinr_matrix, allocations)
        
        # 6. Collect per-user metrics
        user_sinrs_db = 10 * np.log10(np.maximum(
            np.max(sinr_matrix, axis=1), 1e-9))
        
        # 7. Update user objects
        for u_idx, user in enumerate(self.users):
            user.allocated_capacity_mbps = allocations.get(user.user_id, 0)
            
            # Estimate QoS
            channel_quality = min(1.0, user_sinrs_db[u_idx] / 30.0)
            capacity, latency, bler = compute_qos_metrics(
                user.allocated_capacity_mbps, user.slice_type, channel_quality)
            
            user.allocated_capacity_mbps = capacity
            user.latency_ms = latency
            user.packet_loss_ratio = bler
        
        # 8. Compute aggregate metrics
        from scheduler import slice_statistics
        slice_stats = slice_statistics(self.users)
        
        # Assemble metrics
        metrics = SimulationMetrics(
            timestamp=self.time_ms,
            slice_metrics=slice_stats,
            user_sinrs=user_sinrs_db,
            user_capacities=throughputs,
            user_latencies=np.array([u.latency_ms for u in self.users]),
            user_serving_bs=self.user_serving_bs.copy(),
            total_throughput_mbps=np.sum(throughputs),
            avg_sinr_db=np.mean(user_sinrs_db),
            energy_efficiency=np.sum(throughputs) / (self.num_bs * 5.0)  # rough estimate
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def run(self, progress_callback=None) -> List[SimulationMetrics]:
        """
        Run complete simulation.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of metrics for each step
        """
        dt_ms = 1.0  # 1 ms per step (1 TTI)
        num_steps = int(self.simulation_time_ms / dt_ms)
        
        for step in range(num_steps):
            self.step(dt_ms)
            
            if progress_callback and step % 100 == 0:
                progress_callback(step, num_steps)
        
        return self.metrics_history
    
    # ========================================================================
    # ANALYSIS & REPORTING
    # ========================================================================
    
    def get_summary_statistics(self) -> Dict:
        """Compute summary statistics of the simulation."""
        if not self.metrics_history:
            return {}
        
        metrics_array = self.metrics_history
        
        summary = {
            'total_time_ms': self.time_ms,
            'num_steps': len(metrics_array),
            
            'avg_throughput_mbps': np.mean([m.total_throughput_mbps 
                                            for m in metrics_array]),
            'avg_sinr_db': np.mean([m.avg_sinr_db for m in metrics_array]),
            
            'slice_summary': {}
        }
        
        # Per-slice summary
        for slice_type in ['eMBB', 'URLLC', 'mMTC']:
            slice_capacities = []
            for m in metrics_array:
                if slice_type in m.slice_metrics:
                    cap = m.slice_metrics[slice_type]['avg_capacity_per_user']
                    slice_capacities.append(cap)
            
            if slice_capacities:
                summary['slice_summary'][slice_type] = {
                    'avg_capacity_mbps': np.mean(slice_capacities),
                    'min_capacity_mbps': np.min(slice_capacities),
                    'max_capacity_mbps': np.max(slice_capacities)
                }
        
        return summary


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_quick_simulation(num_users: int = 50, duration_ms: int = 1000,
                        seed: int = 42) -> NetworkSimulation:
    """
    Run a quick simulation with default parameters.
    
    Args:
        num_users: Number of users
        duration_ms: Simulation duration in milliseconds
        seed: Random seed
    
    Returns:
        Completed simulation object
    """
    sim = NetworkSimulation(
        simulation_time_ms=duration_ms,
        num_users=num_users,
        num_bs=10,
        scenario='UMi',
        seed=seed
    )
    
    sim.run()
    return sim


if __name__ == "__main__":
    # Test the simulator
    print("=" * 70)
    print("5G NR Network Simulation Test")
    print("=" * 70)
    
    # Quick simulation
    print("\nRunning 500ms simulation with 50 users and 10 base stations...")
    
    sim = NetworkSimulation(
        simulation_time_ms=500,
        num_users=50,
        num_bs=10,
        scenario='UMi',
        seed=42
    )
    
    def progress(step, total):
        percent = 100 * step / total
        print(f"\r  Progress: {percent:.0f}%", end="", flush=True)
    
    sim.run(progress_callback=progress)
    print("\r  Progress: 100%")
    
    # Print statistics
    stats = sim.get_summary_statistics()
    
    print(f"\nSimulation Results:")
    print(f"  Total Time: {stats['total_time_ms']:.0f} ms")
    print(f"  Number of Steps: {stats['num_steps']}")
    print(f"  Average Total Throughput: {stats['avg_throughput_mbps']:.1f} Mbps")
    print(f"  Average SINR: {stats['avg_sinr_db']:.1f} dB")
    
    print(f"\nPer-Slice Statistics:")
    for slice_type, slice_stats in stats['slice_summary'].items():
        print(f"  {slice_type}:")
        print(f"    Avg Capacity: {slice_stats['avg_capacity_mbps']:.1f} Mbps")
        print(f"    Min Capacity: {slice_stats['min_capacity_mbps']:.1f} Mbps")
        print(f"    Max Capacity: {slice_stats['max_capacity_mbps']:.1f} Mbps")