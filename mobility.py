"""
Mobility Model Module
Implements realistic user movement patterns for 5G network simulation
"""

import numpy as np
from typing import Tuple


# ============================================================================
# MOBILITY MODELS
# ============================================================================

class RandomWaypointMobility:
    """
    Random Waypoint mobility model.
    Users move toward random destinations at varying speeds.
    """
    
    def __init__(self, bounds: Tuple[float, float], 
                 min_velocity: float = 0.5,
                 max_velocity: float = 5.0,
                 pause_prob: float = 0.1):
        """
        Args:
            bounds: Simulation area bounds [(x_min, x_max), (y_min, y_max)]
            min_velocity: Minimum velocity in m/s
            max_velocity: Maximum velocity in m/s
            pause_prob: Probability user pauses at waypoint
        """
        self.bounds = bounds
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.pause_prob = pause_prob
        
        self.users_waypoints = {}  # user_id -> next waypoint
        self.users_velocities = {}  # user_id -> current velocity
        self.users_pause_time = {}  # user_id -> pause timer
    
    def initialize_user(self, user_id: int, position: np.ndarray):
        """Initialize a user with random waypoint."""
        self.users_waypoints[user_id] = self._random_position()
        self.users_velocities[user_id] = np.random.uniform(
            self.min_velocity, self.max_velocity)
        self.users_pause_time[user_id] = 0
    
    def update(self, user_id: int, position: np.ndarray, 
              dt: float = 0.001) -> np.ndarray:
        """
        Update user position (Random Waypoint).
        
        Args:
            user_id: User identifier
            position: Current position [x, y]
            dt: Time step in seconds
        
        Returns:
            New position [x, y]
        """
        if user_id not in self.users_waypoints:
            self.initialize_user(user_id, position)
        
        # Check if paused
        if self.users_pause_time[user_id] > 0:
            self.users_pause_time[user_id] -= dt
            return position
        
        # Get destination
        waypoint = self.users_waypoints[user_id]
        velocity = self.users_velocities[user_id]
        
        # Direction toward waypoint
        direction = waypoint - position
        distance = np.linalg.norm(direction)
        
        if distance < 1.0:
            # Reached waypoint
            if np.random.random() < self.pause_prob:
                # Pause at waypoint
                self.users_pause_time[user_id] = np.random.uniform(1, 5)
            
            # Generate new waypoint
            self.users_waypoints[user_id] = self._random_position()
            self.users_velocities[user_id] = np.random.uniform(
                self.min_velocity, self.max_velocity)
            return position
        
        # Move toward waypoint
        direction_normalized = direction / distance
        displacement = velocity * dt * direction_normalized
        new_position = position + displacement
        
        # Bounce off boundaries
        new_position = self._apply_boundaries(new_position)
        
        return new_position
    
    def _random_position(self) -> np.ndarray:
        """Generate random position within bounds."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
    
    def _apply_boundaries(self, position: np.ndarray) -> np.ndarray:
        """Apply boundary conditions (reflection)."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        
        if position[0] < x_min or position[0] > x_max:
            position[0] = np.clip(position[0], x_min, x_max)
        if position[1] < y_min or position[1] > y_max:
            position[1] = np.clip(position[1], y_min, y_max)
        
        return position


class VehicularMobility:
    """
    Vehicular (V2X) mobility model.
    Models high-speed, highway-like movement patterns.
    """
    
    def __init__(self, bounds: Tuple[float, float],
                 velocity: float = 25.0,  # m/s = 90 km/h
                 direction_change_prob: float = 0.05):
        """
        Args:
            bounds: Simulation area bounds
            velocity: Vehicle speed in m/s
            direction_change_prob: Probability of direction change per step
        """
        self.bounds = bounds
        self.velocity = velocity
        self.direction_change_prob = direction_change_prob
        
        self.users_direction = {}  # user_id -> direction angle (radians)
    
    def initialize_user(self, user_id: int, position: np.ndarray):
        """Initialize a vehicle with random direction."""
        self.users_direction[user_id] = np.random.uniform(0, 2*np.pi)
    
    def update(self, user_id: int, position: np.ndarray,
              dt: float = 0.001) -> np.ndarray:
        """
        Update vehicle position.
        
        Args:
            user_id: Vehicle identifier
            position: Current position [x, y]
            dt: Time step in seconds
        
        Returns:
            New position [x, y]
        """
        if user_id not in self.users_direction:
            self.initialize_user(user_id, position)
        
        # Random direction change
        if np.random.random() < self.direction_change_prob:
            self.users_direction[user_id] += np.random.normal(0, 0.2)
        
        # Movement
        direction = self.users_direction[user_id]
        displacement = np.array([
            np.cos(direction) * self.velocity * dt,
            np.sin(direction) * self.velocity * dt
        ])
        
        new_position = position + displacement
        
        # Wrap around boundaries
        (x_min, x_max), (y_min, y_max) = self.bounds
        
        if new_position[0] < x_min or new_position[0] > x_max:
            new_position[0] = ((new_position[0] - x_min) % (x_max - x_min)) + x_min
        if new_position[1] < y_min or new_position[1] > y_max:
            new_position[1] = ((new_position[1] - y_min) % (y_max - y_min)) + y_min
        
        return new_position


class StationaryUser:
    """
    Stationary user (for comparison/testing).
    """
    
    def update(self, user_id: int, position: np.ndarray,
              dt: float = 0.001) -> np.ndarray:
        """No movement."""
        return position


# ============================================================================
# HANDOVER LOGIC
# ============================================================================

class HandoverController:
    """
    Manages user handover between base stations.
    Implements hysteresis-based handover decisions.
    """
    
    def __init__(self, hysteresis_db: float = 3.0,
                ttc_threshold_ms: float = 100.0):
        """
        Args:
            hysteresis_db: Hysteresis margin in dB
            ttc_threshold_ms: Time-to-trigger in milliseconds
        """
        self.hysteresis_db = hysteresis_db
        self.ttc_threshold_ms = ttc_threshold_ms
        
        self.user_serving_bs = {}  # user_id -> serving BS index
        self.user_measurement_gap = {}  # user_id -> measurement history
    
    def compute_handover_decision(self, user_id: int,
                                 sinr_per_bs: np.ndarray,
                                 current_bs: int) -> int:
        """
        Handover decision based on SINR.
        
        Args:
            user_id: User identifier
            sinr_per_bs: SINR from each BS (linear scale)
            current_bs: Current serving BS index
        
        Returns:
            Target BS index (current_bs if no handover)
        """
        # Find best BS
        best_bs = np.argmax(sinr_per_bs)
        
        # Hysteresis: stay with current BS unless significantly better
        current_sinr_db = 10 * np.log10(np.maximum(sinr_per_bs[current_bs], 1e-9))
        best_sinr_db = 10 * np.log10(np.maximum(sinr_per_bs[best_bs], 1e-9))
        
        if best_sinr_db > current_sinr_db + self.hysteresis_db:
            # Handover triggered
            return best_bs
        else:
            return current_bs


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_user_traces(num_users: int, num_steps: int,
                        bounds: Tuple[float, float],
                        model_type: str = 'random_waypoint',
                        seed: int = None) -> np.ndarray:
    """
    Generate complete user mobility traces.
    
    Args:
        num_users: Number of users
        num_steps: Number of simulation steps
        bounds: Simulation area bounds
        model_type: 'random_waypoint', 'vehicular', or 'stationary'
        seed: Random seed for reproducibility
    
    Returns:
        Traces: shape (num_steps, num_users, 2) with positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize model
    if model_type == 'random_waypoint':
        model = RandomWaypointMobility(bounds)
    elif model_type == 'vehicular':
        model = VehicularMobility(bounds)
    else:
        model = StationaryUser()
    
    # Initialize positions
    (x_min, x_max), (y_min, y_max) = bounds
    positions = np.random.uniform(
        [x_min, y_min], [x_max, y_max], (num_users, 2))
    
    # Trace
    traces = np.zeros((num_steps, num_users, 2))
    
    # Generate traces
    for step in range(num_steps):
        for user_id in range(num_users):
            positions[user_id] = model.update(user_id, positions[user_id], dt=0.001)
        traces[step] = positions.copy()
    
    return traces


if __name__ == "__main__":
    # Test mobility models
    print("=" * 60)
    print("Mobility Model Test")
    print("=" * 60)
    
    np.random.seed(42)
    bounds = ((-1000, 1000), (-1000, 1000))
    
    # Random waypoint test
    print("\n1. Random Waypoint Mobility:")
    model_rwp = RandomWaypointMobility(bounds)
    pos = np.array([0.0, 0.0])
    
    for step in range(100):
        pos = model_rwp.update(0, pos)
    
    print(f"   Final position: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Vehicular test
    print("\n2. Vehicular Mobility:")
    model_veh = VehicularMobility(bounds, velocity=25.0)
    pos = np.array([0.0, 0.0])
    
    for step in range(100):
        pos = model_veh.update(0, pos, dt=0.01)
    
    print(f"   Final position: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Trace generation
    print("\n3. Generated Traces:")
    traces = generate_user_traces(5, 1000, bounds, seed=42)
    print(f"   Trace shape: {traces.shape}")
    print(f"   User 0 final position: ({traces[-1, 0, 0]:.1f}, {traces[-1, 0, 1]:.1f})")
