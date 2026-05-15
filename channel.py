"""
3GPP TR 38.901 Channel Model Implementation
Based on 3rd Generation Partnership Project Technical Report on channel modeling
Includes: Pathloss, Shadowing, and Small-scale Fading (Rayleigh/Rician)
"""

import numpy as np # pyright: ignore[reportMissingImports]
from typing import Tuple

# ============================================================================
# CONSTANTS (3GPP TR 38.901)
# ============================================================================
FC_DEFAULT = 3.5  # Carrier frequency in GHz (NR mmWave FR1)
SCENARIO_PARAMS = {
    'UMi': {  # Urban Micro
        'los_a': 18.6,
        'los_b': 46.85,
        'nlos_a': 36.7,
        'nlos_b': 32.4,
        'nlos_c': 20.0,
        'nlos_d': 30.0,
        'breakpoint': 200,  # meters
        'shadowing_std': 4.0,  # dB
    },
    'UMa': {  # Urban Macro
        'los_a': 28.0,
        'los_b': 22.0,
        'nlos_a': 13.54,
        'nlos_b': 27.23,
        'nlos_c': 20.0,
        'nlos_d': 32.4,
        'breakpoint': 1000,
        'shadowing_std': 6.0,
    },
    'RMa': {  # Rural Macro
        'los_a': 10.0,
        'los_b': 49.24,
        'nlos_a': 15.3,
        'nlos_b': 37.08,
        'nlos_c': 20.0,
        'nlos_d': 31.5,
        'breakpoint': 5000,
        'shadowing_std': 8.0,
    }
}


# ============================================================================
# PATHLOSS MODELS (3GPP TR 38.901)
# ============================================================================

def los_probability(distance: float, scenario: str = 'UMi') -> float:
    """
    Calculate LOS probability based on distance and scenario.
    3GPP TR 38.901 formula: P_LOS = min(1, C/d) * exp(-d/D)
    
    Args:
        distance: Distance in meters
        scenario: 'UMi', 'UMa', or 'RMa'
    
    Returns:
        Probability between 0 and 1
    """
    if scenario == 'UMi':
        C, D = 32.45, 10.0
    elif scenario == 'UMa':
        C, D = 32.4, 31.0
    elif scenario == 'RMa':
        C, D = 32.6, 120.0
    else:
        C, D = 32.45, 10.0
    
    p_los = np.minimum(1.0, C / distance) * np.exp(-distance / D) if distance > 0 else 1.0
    return float(p_los)


def pathloss_3gpp(distance: float, fc: float = FC_DEFAULT, 
                   scenario: str = 'UMi', los: bool = True) -> float:
    """
    3GPP TR 38.901 pathloss model.
    
    Args:
        distance: Distance in meters
        fc: Carrier frequency in GHz
        scenario: 'UMi', 'UMa', or 'RMa'
        los: True for LOS, False for NLOS
    
    Returns:
        Pathloss in linear scale (0-1 range typically)
    """
    distance = np.maximum(distance, 1.0)  # Avoid log(0)
    params = SCENARIO_PARAMS.get(scenario, SCENARIO_PARAMS['UMi'])
    
    if los:
        # LOS pathloss
        a = params['los_a']
        b = params['los_b']
        bp = params['breakpoint']
        
        if distance <= bp:
            pl_db = a * np.log10(distance) + b + 20 * np.log10(fc)
        else:
            pl_1 = a * np.log10(bp) + b + 20 * np.log10(fc)
            pl_db = pl_1 + 40 * np.log10(distance / bp)
    else:
        # NLOS pathloss
        a = params['nlos_a']
        b = params['nlos_b']
        c = params['nlos_c']
        d = params['nlos_d']
        pl_db = a * np.log10(distance) + b + c * np.log10(fc) - d
    
    # Convert dB to linear
    return 10 ** (-pl_db / 10)


def shadowing_3gpp(scenario: str = 'UMi', std_dev: float = None) -> float:
    """
    Lognormal shadowing model.
    
    Args:
        scenario: 'UMi', 'UMa', or 'RMa'
        std_dev: Standard deviation in dB (uses scenario default if None)
    
    Returns:
        Shadowing factor in linear scale
    """
    params = SCENARIO_PARAMS.get(scenario, SCENARIO_PARAMS['UMi'])
    sigma = std_dev if std_dev is not None else params['shadowing_std']
    
    # Lognormal: X_dB ~ N(0, sigma)
    shadow_db = np.random.normal(0, sigma)
    return 10 ** (shadow_db / 10)


# ============================================================================
# SMALL-SCALE FADING MODELS
# ============================================================================

def fading_rayleigh(num_paths: int = 1) -> np.ndarray:
    """
    Rayleigh fading model (NLOS condition).
    Typical for non-line-of-sight with many scatterers.
    
    Args:
        num_paths: Number of multipath components
    
    Returns:
        Complex channel coefficients
    """
    real = np.random.normal(0, 1/np.sqrt(2), num_paths)
    imag = np.random.normal(0, 1/np.sqrt(2), num_paths)
    h = real + 1j * imag
    return h / np.linalg.norm(h)  # Normalize


def fading_rician(k_factor: float = 1.0, num_paths: int = 1) -> np.ndarray:
    """
    Rician fading model (LOS + scattered components).
    Used when dominant LOS component exists.
    
    Args:
        k_factor: Rician K factor (LOS power / scattered power)
                 0 → Rayleigh, large value → strong LOS
        num_paths: Number of multipath components
    
    Returns:
        Complex channel coefficients
    """
    # LOS component
    los_component = np.sqrt(k_factor / (k_factor + 1))
    
    # Scattered components
    scattered = np.random.normal(0, 1/np.sqrt(2*(k_factor+1)), num_paths) + \
                1j * np.random.normal(0, 1/np.sqrt(2*(k_factor+1)), num_paths)
    
    h = los_component + scattered
    return h / np.linalg.norm(h)


def multipath_channel(distance: float, fc: float = FC_DEFAULT, 
                     scenario: str = 'UMi', num_paths: int = 4) -> Tuple[float, float]:
    """
    Compute combined channel attenuation (pathloss + fading + shadowing).
    
    Args:
        distance: Distance in meters
        fc: Carrier frequency in GHz
        scenario: 'UMi', 'UMa', or 'RMa'
        num_paths: Number of multipath components
    
    Returns:
        (channel_gain_linear, los_flag) - channel gain (0-1) and LOS indicator
    """
    # Determine LOS/NLOS
    p_los = los_probability(distance, scenario)
    is_los = np.random.random() < p_los
    
    # Pathloss
    pl = pathloss_3gpp(distance, fc, scenario, is_los)
    
    # Shadowing
    shadow = shadowing_3gpp(scenario)
    
    # Small-scale fading
    if is_los:
        k_factor = 3.0 + 0.05 * distance  # K-factor increases with distance
        h = fading_rician(k_factor, num_paths)
    else:
        h = fading_rayleigh(num_paths)
    
    # Combined gain
    fading_power = np.abs(h[0]) ** 2  # Use first path
    channel_gain = pl * shadow * fading_power
    
    return channel_gain, float(is_los)


# ============================================================================
# ANTENNA ARRAY EFFECTS
# ============================================================================

def antenna_gain(angle_deg: float, num_antennas: int = 8) -> float:
    """
    Simple phased array antenna gain model.
    
    Args:
        angle_deg: Angle between beam and user direction (degrees)
        num_antennas: Number of antennas in array
    
    Returns:
        Antenna gain (linear)
    """
    # Beamwidth depends on number of antennas
    beamwidth = 180.0 / num_antennas
    
    # Simplified gain pattern (cosine squared)
    angle_rad = np.radians(np.abs(angle_deg))
    if angle_rad > np.pi / 2:
        return 0.01
    
    gain_linear = (np.cos(angle_rad) ** 2) ** num_antennas
    return gain_linear


if __name__ == "__main__":
    # Test the channel model
    np.random.seed(42)
    
    print("=" * 60)
    print("3GPP TR 38.901 Channel Model Test")
    print("=" * 60)
    
    distances = [50, 200, 500, 1000]
    
    for d in distances:
        p_los = los_probability(d, 'UMi')
        gain, is_los = multipath_channel(d, fc=3.5, scenario='UMi')
        gain_db = 10 * np.log10(gain + 1e-9)
        
        print(f"\nDistance: {d}m")
        print(f"  P(LOS): {p_los:.3f}")
        print(f"  Is LOS: {bool(is_los)}")
        print(f"  Channel Gain: {gain:.6f} ({gain_db:.2f} dB)")