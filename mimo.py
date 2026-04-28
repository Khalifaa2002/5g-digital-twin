"""
MIMO and Beamforming Module
Implements massive MIMO beamforming, precoding, and capacity computation
"""

import numpy as np
from typing import Tuple


# ============================================================================
# BEAMFORMING PATTERNS
# ============================================================================

def uniform_linear_array_response(angle_deg: float, M: int = 8) -> np.ndarray:
    """
    Compute steering vector for uniform linear array (ULA).
    
    Args:
        angle_deg: Direction of arrival/departure in degrees
        M: Number of antennas
    
    Returns:
        Complex steering vector (M,)
    """
    angle_rad = np.radians(angle_deg)
    # Assume half-wavelength spacing
    n = np.arange(M)
    steering = np.exp(1j * np.pi * n * np.sin(angle_rad))
    return steering / np.sqrt(M)


def maximum_ratio_combiner(channel: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Maximum Ratio Combining (MRC) - optimal combiner for AWGN channels.
    
    Args:
        channel: Complex channel vector (M,)
    
    Returns:
        (SNR improvement factor, combining weights)
    """
    # Optimal weights: conjugate of channel coefficients
    w = np.conj(channel) / (np.linalg.norm(channel) ** 2 + 1e-10)
    
    # SNR gain = sum of squared channel magnitudes
    snr_gain = np.sum(np.abs(channel) ** 2)
    
    return float(snr_gain), w


def zero_forcing_precoder(H: np.ndarray, num_users: int) -> np.ndarray:
    """
    Zero Forcing (ZF) precoding for MU-MIMO.
    Nulls out inter-user interference.
    
    Args:
        H: Channel matrix (M, K) where M=antennas, K=users
        num_users: Number of users
    
    Returns:
        Precoding matrix (M, K)
    """
    try:
        # Pseudo-inverse for ZF precoding
        H_pseudo_inv = np.linalg.pinv(H)
        # Normalize to unit power
        P = H_pseudo_inv / np.sqrt(np.trace(H_pseudo_inv @ np.conj(H_pseudo_inv.T)))
        return P
    except:
        return np.eye(num_users)


# ============================================================================
# BEAMFORMING GAIN
# ============================================================================

def directional_beamforming_gain(user_angle_deg: float, bs_angle_deg: float, 
                               num_antennas: int = 8) -> float:
    """
    Compute beamforming gain based on angle mismatch.
    
    Args:
        user_angle_deg: Angle to user (degrees)
        bs_angle_deg: Base station beam angle (degrees)
        num_antennas: Number of antennas in array
    
    Returns:
        Beamforming gain (0 to num_antennas^2)
    """
    # Steering vectors
    a_user = uniform_linear_array_response(user_angle_deg, num_antennas)
    a_beam = uniform_linear_array_response(bs_angle_deg, num_antennas)
    
    # Array gain = |a_beam^H * a_user|^2
    gain = np.abs(np.dot(np.conj(a_beam), a_user)) ** 2
    
    return float(gain * num_antennas ** 2)


def omnidirectional_beamforming_gain(user_pos: np.ndarray, bs_pos: np.ndarray,
                                    num_antennas: int = 4) -> float:
    """
    Simplified isotropic beamforming (no beam adaptation).
    
    Args:
        user_pos: User position [x, y]
        bs_pos: Base station position [x, y]
        num_antennas: Number of antennas
    
    Returns:
        Beamforming gain
    """
    # Simple cosine-based directivity
    diff = user_pos - bs_pos
    angle = np.arctan2(diff[1], diff[0])
    
    # Gain falls off with angle deviation from BS orientation
    directivity = (np.cos(angle) ** 2 + 0.5) / 1.5
    
    # Directivity gain combined with array gain
    gain = directivity * num_antennas
    
    return float(np.maximum(gain, 0.5))


def adaptive_beamforming(channel: np.ndarray, num_antennas: int = 8) -> float:
    """
    Compute gain from adaptive beamforming (e.g., MRC).
    
    Args:
        channel: Complex channel coefficients
        num_antennas: Number of antennas
    
    Returns:
        Array gain
    """
    if isinstance(channel, np.ndarray) and len(channel) > 0:
        power = np.sum(np.abs(channel) ** 2)
    else:
        power = 1.0
    
    return float(power)


# ============================================================================
# MIMO CAPACITY
# ============================================================================

def shannon_capacity(sinr: float, bandwidth_mhz: float = 20.0) -> float:
    """
    Shannon capacity for SISO link.
    C = B * log2(1 + SINR)
    
    Args:
        sinr: SINR in linear scale
        bandwidth_mhz: Bandwidth in MHz
    
    Returns:
        Capacity in Mbps
    """
    sinr = np.maximum(sinr, 0)
    capacity_mhz = bandwidth_mhz * np.log2(1 + sinr)
    return float(capacity_mhz)


def mimo_capacity_siso(sinr: float, bandwidth_mhz: float = 20.0) -> float:
    """
    SISO capacity (baseline for comparison).
    """
    return shannon_capacity(sinr, bandwidth_mhz)


def mimo_capacity_massive(sinr: float, num_tx: int = 64, num_rx: int = 4,
                          bandwidth_mhz: float = 20.0) -> float:
    """
    Simplified massive MIMO capacity (antenna gain effect).
    Assumes perfect CSI and independent channels.
    
    Args:
        sinr: SINR at single antenna
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        bandwidth_mhz: Bandwidth in MHz
    
    Returns:
        Capacity in Mbps
    """
    # Spatial multiplexing order
    rank = min(num_tx, num_rx)
    
    # Effective SINR per stream (rough approximation)
    # In massive MIMO, SINR improves with antenna count
    effective_sinr = sinr * np.sqrt(num_tx)
    
    capacity_per_stream = shannon_capacity(effective_sinr, bandwidth_mhz)
    
    return float(capacity_per_stream * rank)


def spectral_efficiency(sinr: float, mimo_config: str = 'SISO') -> float:
    """
    Spectral efficiency (bits per Hz per second).
    
    Args:
        sinr: SINR in linear scale
        mimo_config: 'SISO', '2x2', '4x4', '8x8', 'MASSIVE'
    
    Returns:
        Spectral efficiency in bits/s/Hz
    """
    configs = {
        'SISO': 1,
        '2x2': 2,
        '4x4': 4,
        '8x8': 8,
        'MASSIVE': 16
    }
    
    rank = configs.get(mimo_config, 1)
    
    # Log2(1 + SINR) per stream * number of streams
    se_per_stream = np.log2(1 + sinr)
    
    return float(rank * se_per_stream)


def link_spectral_efficiency(sinr: float, bler_target: float = 0.1) -> float:
    """
    Practical link-level spectral efficiency with BLER constraint.
    Uses Shannon bound minus 1.5 dB gap.
    
    Args:
        sinr: SINR in linear scale
        bler_target: Block error rate target
    
    Returns:
        Spectral efficiency in bits/s/Hz
    """
    # Shannon limit with implementation loss (~1.5 dB = 1.41x in linear)
    implementation_gap = 1.41
    
    effective_sinr = sinr / implementation_gap
    se = np.log2(1 + effective_sinr)
    
    return float(np.maximum(se, 0))


if __name__ == "__main__":
    # Test MIMO functions
    np.random.seed(42)
    
    print("=" * 60)
    print("MIMO & Beamforming Test")
    print("=" * 60)
    
    # Test beamforming gain
    print("\n1. Directional Beamforming:")
    for angle_error in [0, 15, 30, 45]:
        gain = directional_beamforming_gain(0, angle_error, num_antennas=8)
        gain_db = 10 * np.log10(gain)
        print(f"   Angle mismatch {angle_error}°: {gain_db:.2f} dB")
    
    # Test MIMO capacity
    print("\n2. MIMO Capacity (SINR = 10 dB):")
    sinr_linear = 10.0
    for config in ['SISO', '2x2', '4x4', '8x8', 'MASSIVE']:
        cap = mimo_capacity_massive(sinr_linear, bandwidth_mhz=20.0)
        print(f"   {config}: {cap:.2f} Mbps")
    
    # Test spectral efficiency
    print("\n3. Spectral Efficiency vs SINR:")
    for sinr_db in [0, 5, 10, 15, 20]:
        sinr_lin = 10 ** (sinr_db / 10)
        se = spectral_efficiency(sinr_lin, 'MASSIVE')
        print(f"   SINR {sinr_db} dB: {se:.2f} bits/s/Hz")