"""
5G Network Slicing and Scheduling Module
Implements eMBB, URLLC, and mMTC network slices with QoS scheduling
"""

import numpy as np
from typing import Dict, List, Tuple


# ============================================================================
# SLICE DEFINITIONS (3GPP TS 26.501)
# ============================================================================

SLICE_CONFIGS = {
    'eMBB': {  # Enhanced Mobile Broadband
        'latency_budget_ms': 20,
        'reliability': 0.99,
        'data_rate_mbps': 100,
        'priority': 2,
        'description': 'High capacity, moderate latency'
    },
    'URLLC': {  # Ultra-Reliable Low-Latency Communications
        'latency_budget_ms': 1,
        'reliability': 0.99999,
        'data_rate_mbps': 10,
        'priority': 3,
        'description': 'Critical, low latency, high reliability'
    },
    'mMTC': {  # Massive Machine Type Communications
        'latency_budget_ms': 1000,
        'reliability': 0.90,
        'data_rate_mbps': 0.5,
        'priority': 1,
        'description': 'Massive connectivity, low data rate'
    }
}


# ============================================================================
# QoS METRICS & TRACKING
# ============================================================================

class UserQoSProfile:
    """QoS profile for a user."""
    
    def __init__(self, user_id: int, slice_type: str, qos_class: int = 1):
        """
        Args:
            user_id: Unique user identifier
            slice_type: 'eMBB', 'URLLC', or 'mMTC'
            qos_class: QoS Class Indicator (1-9, higher = better)
        """
        self.user_id = user_id
        self.slice_type = slice_type
        self.qos_class = qos_class
        
        self.config = SLICE_CONFIGS.get(slice_type, SLICE_CONFIGS['eMBB'])
        self.priority = self.config['priority'] + (qos_class - 1) / 10
        
        # Tracking
        self.allocated_rbs = 0
        self.allocated_capacity_mbps = 0.0
        self.latency_ms = 0.0
        self.packet_loss_ratio = 0.0
    
    def is_qos_satisfied(self, capacity_mbps: float, latency_ms: float) -> bool:
        """Check if QoS is satisfied."""
        return (capacity_mbps >= self.config['data_rate_mbps'] and
                latency_ms <= self.config['latency_budget_ms'])
    
    def get_priority_weight(self, bler: float = 0.0) -> float:
        """
        Compute scheduling priority weight.
        Higher = more urgent
        """
        # Base priority from slice type
        base_priority = self.priority
        
        # Boost if QoS not satisfied
        qos_met = self.is_qos_satisfied(
            self.allocated_capacity_mbps,
            self.latency_ms
        )
        
        if not qos_met:
            base_priority *= 2.0  # Urgency boost
        
        return base_priority


# ============================================================================
# SCHEDULING ALGORITHMS
# ============================================================================

def schedule_proportional_fair(users: List[UserQoSProfile], 
                              available_capacity: float) -> Dict[int, float]:
    """
    Proportional Fair (PF) scheduler.
    Balances between maximizing throughput and ensuring fairness.
    
    Args:
        users: List of user QoS profiles
        available_capacity: Total available capacity (Mbps)
    
    Returns:
        Dictionary: user_id -> allocated_capacity_mbps
    """
    if not users or available_capacity <= 0:
        return {}
    
    # Compute metric for each user: instantaneous_rate / average_rate
    # Simplified: use priority weight
    total_weight = sum(u.get_priority_weight() for u in users)
    
    if total_weight == 0:
        # Equal allocation
        allocation = available_capacity / len(users)
        return {u.user_id: allocation for u in users}
    
    # Proportional allocation
    allocations = {}
    for u in users:
        weight = u.get_priority_weight()
        share = (weight / total_weight) * available_capacity
        allocations[u.user_id] = share
    
    return allocations


def schedule_urllc_first(users: List[UserQoSProfile],
                        available_capacity: float) -> Dict[int, float]:
    """
    Priority-based scheduler: URLLC > eMBB > mMTC
    
    Args:
        users: List of user QoS profiles
        available_capacity: Total available capacity (Mbps)
    
    Returns:
        Dictionary: user_id -> allocated_capacity_mbps
    """
    allocations = {}
    remaining_capacity = available_capacity
    
    # Sort by priority (higher first)
    sorted_users = sorted(users, key=lambda u: u.priority, reverse=True)
    
    for u in sorted_users:
        # Allocate minimum required capacity
        min_capacity = u.config['data_rate_mbps']
        
        if remaining_capacity >= min_capacity:
            allocations[u.user_id] = min_capacity
            remaining_capacity -= min_capacity
        else:
            # Best effort
            allocations[u.user_id] = remaining_capacity * (u.priority / 5.0)
            remaining_capacity = 0
    
    # Distribute remaining capacity
    if remaining_capacity > 0:
        for u in sorted_users:
            bonus = remaining_capacity * (u.priority / 5.0) / len(users)
            allocations[u.user_id] += bonus
    
    return allocations


def schedule_slice_isolation(users: List[UserQoSProfile],
                            available_capacity: float,
                            slice_fractions: Dict[str, float] = None) -> Dict[int, float]:
    """
    Network slicing with resource isolation.
    Allocate fixed fraction of resources to each slice.
    
    Args:
        users: List of user QoS profiles
        available_capacity: Total available capacity (Mbps)
        slice_fractions: Dict with keys 'eMBB', 'URLLC', 'mMTC'
                        Values sum to 1.0
    
    Returns:
        Dictionary: user_id -> allocated_capacity_mbps
    """
    if slice_fractions is None:
        slice_fractions = {
            'eMBB': 0.6,
            'URLLC': 0.3,
            'mMTC': 0.1
        }
    
    # Group users by slice
    users_by_slice = {}
    for slice_type in slice_fractions:
        users_by_slice[slice_type] = [u for u in users if u.slice_type == slice_type]
    
    allocations = {}
    
    # Allocate per slice
    for slice_type, fraction in slice_fractions.items():
        slice_capacity = available_capacity * fraction
        slice_users = users_by_slice[slice_type]
        
        if slice_users:
            # Proportional fair within slice
            total_weight = sum(u.priority for u in slice_users)
            
            for u in slice_users:
                share = (u.priority / total_weight) * slice_capacity
                allocations[u.user_id] = share
        
    return allocations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_qos_metrics(allocated_capacity: float, 
                       slice_type: str,
                       channel_quality: float = 1.0) -> Tuple[float, float, float]:
    """
    Estimate QoS metrics (capacity, latency, BLER) from channel quality.
    
    Args:
        allocated_capacity: Allocated capacity (Mbps)
        slice_type: 'eMBB', 'URLLC', or 'mMTC'
        channel_quality: Channel condition (0-1, where 1 is excellent)
    
    Returns:
        (actual_capacity_mbps, latency_ms, bler)
    """
    # Capacity scales with channel quality
    actual_capacity = allocated_capacity * (0.5 + 0.5 * channel_quality)
    
    # Latency depends on slice and load
    if slice_type == 'URLLC':
        base_latency = 0.5  # Ultra-low
    elif slice_type == 'eMBB':
        base_latency = 10.0
    else:  # mMTC
        base_latency = 100.0
    
    # Add queuing delay if capacity insufficient
    latency = base_latency + max(0, 10 * (1 - channel_quality))
    
    # BLER from channel quality
    bler = 0.01 * (1 - channel_quality) ** 2
    
    return actual_capacity, latency, bler


def slice_statistics(users: List[UserQoSProfile]) -> Dict[str, Dict]:
    """
    Compute aggregate statistics per slice.
    
    Args:
        users: List of user QoS profiles
    
    Returns:
        Dictionary: slice_type -> statistics
    """
    stats = {}
    
    for slice_type in SLICE_CONFIGS:
        slice_users = [u for u in users if u.slice_type == slice_type]
        
        if slice_users:
            total_capacity = sum(u.allocated_capacity_mbps for u in slice_users)
            avg_latency = np.mean([u.latency_ms for u in slice_users])
            num_users = len(slice_users)
            
            stats[slice_type] = {
                'num_users': num_users,
                'total_capacity_mbps': total_capacity,
                'avg_capacity_per_user': total_capacity / num_users if num_users > 0 else 0,
                'avg_latency_ms': avg_latency,
                'avg_bler': np.mean([u.packet_loss_ratio for u in slice_users])
            }
        else:
            stats[slice_type] = {
                'num_users': 0,
                'total_capacity_mbps': 0,
                'avg_capacity_per_user': 0,
                'avg_latency_ms': 0,
                'avg_bler': 0
            }
    
    return stats


if __name__ == "__main__":
    # Test scheduling algorithms
    print("=" * 60)
    print("Network Slicing & Scheduling Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test users
    users = []
    user_id = 0
    
    for slice_type in ['eMBB', 'URLLC', 'mMTC']:
        for _ in range(3):
            users.append(UserQoSProfile(user_id, slice_type, qos_class=3))
            user_id += 1
    
    total_capacity = 100.0  # Mbps
    
    print(f"\nTotal users: {len(users)}")
    print(f"Available capacity: {total_capacity} Mbps")
    
    # Test different schedulers
    for sched_name, sched_func in [
        ("Proportional Fair", schedule_proportional_fair),
        ("URLLC First", schedule_urllc_first),
        ("Slice Isolation", schedule_slice_isolation)
    ]:
        print(f"\n{sched_name}:")
        allocations = sched_func(users, total_capacity)
        
        for slice_type in ['eMBB', 'URLLC', 'mMTC']:
            slice_allocs = [allocations.get(u.user_id, 0) 
                          for u in users if u.slice_type == slice_type]
            if slice_allocs:
                print(f"  {slice_type}: {np.sum(slice_allocs):.1f} Mbps "
                      f"({np.mean(slice_allocs):.1f} per user)")