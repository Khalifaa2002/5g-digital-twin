"""
City-Scale Digital Twin Network Extension
Adds dense urban deployment, realistic mobility models, and spatial modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import streamlit as st
from typing import List, Dict, Tuple


# ============================================================================
# CITY ZONE DEFINITIONS
# ============================================================================

class CityZone:
    """Defines a zone in the city-scale network."""
    
    ZONE_TYPES = {
        'downtown': {
            'user_density': 5.0,      # users per 100x100m
            'bs_density': 2.0,        # BS per 100x100m
            'pathloss_factor': 1.2,   # Higher pathloss (buildings)
            'shadowing_std': 8.0,     # dB
            'mobility_type': 'pedestrian',
            'color': '#ff4444'
        },
        'suburban': {
            'user_density': 1.5,
            'bs_density': 0.5,
            'pathloss_factor': 1.0,
            'shadowing_std': 4.0,
            'mobility_type': 'mixed',
            'color': '#44ff44'
        },
        'rural': {
            'user_density': 0.3,
            'bs_density': 0.1,
            'pathloss_factor': 0.8,
            'shadowing_std': 2.0,
            'mobility_type': 'vehicular',
            'color': '#4444ff'
        },
        'industrial': {
            'user_density': 2.0,
            'bs_density': 1.0,
            'pathloss_factor': 1.3,
            'shadowing_std': 6.0,
            'mobility_type': 'mixed',
            'color': '#ffaa00'
        }
    }
    
    def __init__(self, name: str, bounds: Tuple, zone_type: str):
        """
        Args:
            name: Zone identifier
            bounds: ((x_min, x_max), (y_min, y_max))
            zone_type: 'downtown', 'suburban', 'rural', 'industrial'
        """
        self.name = name
        self.bounds = bounds
        self.zone_type = zone_type
        self.config = self.ZONE_TYPES.get(zone_type, self.ZONE_TYPES['suburban'])
    
    def contains(self, position: np.ndarray) -> bool:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max
    
    def area_km2(self) -> float:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return (x_max - x_min) * (y_max - y_min) / 1e6


# ============================================================================
# BUILDING BLOCKAGE MODEL
# ============================================================================

class BuildingBlockageModel:
    """
    Simple building blockage model.
    Defines rectangular buildings that block LOS.
    """
    
    def __init__(self, bounds: Tuple, num_buildings: int = 50, seed: int = 42):
        np.random.seed(seed)
        (x_min, x_max), (y_min, y_max) = bounds
        
        self.buildings = []
        for _ in range(num_buildings):
            bx = np.random.uniform(x_min, x_max - 50)
            by = np.random.uniform(y_min, y_max - 50)
            bw = np.random.uniform(20, 80)
            bh = np.random.uniform(20, 80)
            
            self.buildings.append({
                'x': bx, 'y': by, 'width': bw, 'height': bh,
                'blockage_prob': np.random.uniform(0.6, 0.95)
            })
    
    def check_blockage(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Check if line segment between pos1 and pos2 intersects any building.
        Returns probability of blockage (0 = clear, 1 = blocked).
        """
        for b in self.buildings:
            if self._line_intersects_rect(pos1, pos2, b):
                return b['blockage_prob']
        return 0.0
    
    def _line_intersects_rect(self, p1: np.ndarray, p2: np.ndarray, rect: Dict) -> bool:
        """Check if line segment intersects rectangle."""
        x_min, x_max = rect['x'], rect['x'] + rect['width']
        y_min, y_max = rect['y'], rect['y'] + rect['height']
        
        if x_min <= p1[0] <= x_max and y_min <= p1[1] <= y_max:
            return True
        if x_min <= p2[0] <= x_max and y_min <= p2[1] <= y_max:
            return True
        
        # CCW check for line-rect intersection
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        for i in range(4):
            c1 = np.array(corners[i])
            c2 = np.array(corners[(i+1) % 4])
            if ccw(p1, c1, c2) != ccw(p2, c1, c2) and ccw(p1, p2, c1) != ccw(p1, p2, c2):
                return True
        return False
    
    def render_buildings(self, ax):
        """Render buildings on matplotlib axes."""
        for b in self.buildings:
            rect = Rectangle((b['x'], b['y']), b['width'], b['height'],
                             facecolor='#333333', edgecolor='#555555',
                             alpha=0.6, linewidth=1)
            ax.add_patch(rect)


# ============================================================================
# REALISTIC MOBILITY MODELS
# ============================================================================

class PedestrianMobility:
    """Slow, erratic movement typical of pedestrians."""
    
    def __init__(self, bounds: Tuple, avg_speed: float = 1.3):
        self.bounds = bounds
        self.avg_speed = avg_speed
        self.destinations = {}
    
    def update(self, user_id: int, position: np.ndarray, dt: float) -> np.ndarray:
        if user_id not in self.destinations:
            self.destinations[user_id] = self._random_destination()
        
        dest = self.destinations[user_id]
        direction = dest - position
        distance = np.linalg.norm(direction)
        
        if distance < 2.0:
            self.destinations[user_id] = self._random_destination()
            return position
        
        speed = np.random.normal(self.avg_speed, 0.3)
        displacement = speed * dt * direction / distance
        new_pos = position + displacement
        return self._clip_bounds(new_pos)
    
    def _random_destination(self) -> np.ndarray:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array([np.random.uniform(x_min, x_max),
                          np.random.uniform(y_min, y_max)])
    
    def _clip_bounds(self, pos: np.ndarray) -> np.ndarray:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array([np.clip(pos[0], x_min, x_max),
                          np.clip(pos[1], y_min, y_max)])


class VehicularMobility:
    """High-speed movement typical of vehicles on roads."""
    
    def __init__(self, bounds: Tuple, avg_speed: float = 15.0):
        self.bounds = bounds
        self.avg_speed = avg_speed
        self.velocities = {}
    
    def update(self, user_id: int, position: np.ndarray, dt: float) -> np.ndarray:
        if user_id not in self.velocities:
            angle = np.random.uniform(0, 2*np.pi)
            self.velocities[user_id] = self.avg_speed * np.array([np.cos(angle), np.sin(angle)])
        
        # Small random steering
        steering = np.random.normal(0, 0.1)
        angle = np.arctan2(self.velocities[user_id][1], self.velocities[user_id][0])
        angle += steering
        speed = np.linalg.norm(self.velocities[user_id])
        self.velocities[user_id] = speed * np.array([np.cos(angle), np.sin(angle)])
        
        new_pos = position + self.velocities[user_id] * dt
        (x_min, x_max), (y_min, y_max) = self.bounds
        
        # Wrap around
        if new_pos[0] < x_min or new_pos[0] > x_max:
            self.velocities[user_id][0] *= -1
            new_pos[0] = np.clip(new_pos[0], x_min, x_max)
        if new_pos[1] < y_min or new_pos[1] > y_max:
            self.velocities[user_id][1] *= -1
            new_pos[1] = np.clip(new_pos[1], y_min, y_max)
        
        return new_pos


class HotspotMobility:
    """Users clustered around hotspots with occasional jumps."""
    
    def __init__(self, bounds: Tuple, hotspots: List[np.ndarray], jump_prob: float = 0.01):
        self.bounds = bounds
        self.hotspots = hotspots
        self.jump_prob = jump_prob
        self.current_hotspot = {}
    
    def update(self, user_id: int, position: np.ndarray, dt: float) -> np.ndarray:
        if user_id not in self.current_hotspot:
            self.current_hotspot[user_id] = np.random.randint(0, len(self.hotspots))
        
        # Occasionally jump to new hotspot
        if np.random.random() < self.jump_prob:
            self.current_hotspot[user_id] = np.random.randint(0, len(self.hotspots))
        
        hotspot = self.hotspots[self.current_hotspot[user_id]]
        direction = hotspot - position
        distance = np.linalg.norm(direction)
        
        # Move toward hotspot with noise
        speed = np.random.normal(2.0, 0.5)
        noise = np.random.normal(0, 3.0, 2)
        
        if distance > 5.0:
            displacement = speed * dt * direction / distance + noise * dt
        else:
            # Orbit around hotspot
            angle = np.arctan2(direction[1], direction[0]) + np.pi/2
            displacement = speed * dt * np.array([np.cos(angle), np.sin(angle)])
        
        return position + displacement


# ============================================================================
# CITY-SCALE NETWORK GENERATOR
# ============================================================================

def generate_city_scale_config(city_mode: str = 'dense_urban') -> Dict:
    """
    Generate city-scale network configuration.
    
    Returns:
        Dict with num_users, num_bs, bounds, zones, buildings
    """
    configs = {
        'dense_urban': {
            'num_users': 500,
            'num_bs': 100,
            'bounds': ((-2000, 2000), (-2000, 2000)),
            'zones': [
                ('downtown', ((-2000, -500), (-2000, 2000))),
                ('suburban', ((-500, 1500), (-2000, 2000))),
                ('rural', ((1500, 2000), (-2000, 2000))),
            ],
            'num_buildings': 200
        },
        'medium_urban': {
            'num_users': 200,
            'num_bs': 50,
            'bounds': ((-1500, 1500), (-1500, 1500)),
            'zones': [
                ('downtown', ((-1500, 0), (-1500, 1500))),
                ('suburban', ((0, 1500), (-1500, 1500))),
            ],
            'num_buildings': 100
        },
        'megacity': {
            'num_users': 2000,
            'num_bs': 200,
            'bounds': ((-5000, 5000), (-5000, 5000)),
            'zones': [
                ('downtown', ((-5000, -2000), (-5000, 5000))),
                ('industrial', ((-2000, 0), (-5000, 5000))),
                ('suburban', ((0, 3000), (-5000, 5000))),
                ('rural', ((3000, 5000), (-5000, 5000))),
            ],
            'num_buildings': 500
        }
    }
    
    return configs.get(city_mode, configs['medium_urban'])


def render_city_scale_overview(zones: List[CityZone], buildings: BuildingBlockageModel,
                                bounds: Tuple, num_users: int, num_bs: int):
    """
    Render city-scale network overview with zones and buildings.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    (x_min, x_max), (y_min, y_max) = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    
    # Render zones
    for zone in zones:
        (zx_min, zx_max), (zy_min, zy_max) = zone.bounds
        rect = Rectangle((zx_min, zy_min), zx_max - zx_min, zy_max - zy_min,
                         facecolor=zone.config['color'], alpha=0.1,
                         edgecolor=zone.config['color'], linewidth=2, linestyle='--')
        ax.add_patch(rect)
        cx, cy = (zx_min + zx_max) / 2, (zy_min + zy_max) / 2
        ax.text(cx, cy, zone.name.upper(), fontsize=10, color=zone.config['color'],
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # Render buildings
    if buildings:
        buildings.render_buildings(ax)
    
    ax.set_title("City-Scale Digital Twin Network", fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel("X (m)", color='white')
    ax.set_ylabel("Y (m)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1, color='white')
    
    # Legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=z.config['color'],
                                   markersize=10, label=z.name.upper(), alpha=0.5)
                       for z in zones]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black',
              edgecolor='white', labelcolor='white')
    
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    
    # Zone statistics table
    st.markdown("### Zone Statistics")
    stats_data = []
    for zone in zones:
        area = zone.area_km2()
        est_users = int(area * zone.config['user_density'] * 1e4)
        est_bs = int(area * zone.config['bs_density'] * 1e4)
        stats_data.append({
            'Zone': zone.name.title(),
            'Type': zone.zone_type.title(),
            'Area (km2)': round(area, 2),
            'Est. Users': est_users,
            'Est. BS': est_bs,
            'Pathloss Factor': zone.config['pathloss_factor'],
            'Shadowing (dB)': zone.config['shadowing_std']
        })
    
    import pandas as pd
    st.dataframe(pd.DataFrame(stats_data),  width="stretch")


def render_los_nlos_map(sim, buildings: BuildingBlockageModel = None,
                        resolution: int = 40):
    """
    Render LOS/NLOS probability map over the simulation area.
    """
    (x_min, x_max), (y_min, y_max) = sim.bounds
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute LOS probability from nearest BS
    los_prob = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i,j], Y[i,j]])
            
            # Find nearest BS
            min_dist = float('inf')
            for bs in sim.base_stations:
                d = np.linalg.norm(pos - bs.position)
                if d < min_dist:
                    min_dist = d
            
            # LOS probability from 3GPP
            if min_dist < 10:
                p_los = 1.0
            else:
                from channel import los_probability
                p_los = los_probability(min_dist, sim.scenario)
            
            # Building blockage penalty
            if buildings and len(buildings.buildings) > 0:
                pen = 0
                
                pen = np.random.uniform(0, 0.3)
                p_los *= (1 - pen)
            
            los_prob[i,j] = p_los
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, los_prob, levels=20, cmap='RdYlGn', alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax, label='P(LOS)')
    
    # Mark BS positions
    for bs in sim.base_stations:
        ax.plot(bs.position[0], bs.position[1], 'w^', markersize=10,
                markeredgecolor='black')
    
    ax.set_title('LOS/NLOS Probability Map', fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ============================================================================
# HOTSPOT TRAFFIC ZONES
# ============================================================================

def generate_hotspot_zones(bounds, num_hotspots=5, seed=42):
    np.random.seed(seed)
    (x_min, x_max), (y_min, y_max) = bounds
    
    hotspots = []
    for i in range(num_hotspots):
        cx = np.random.uniform(x_min + 100, x_max - 100)
        cy = np.random.uniform(y_min + 100, y_max - 100)
        radius = np.random.uniform(50, 200)
        intensity = np.random.uniform(0.5, 1.0)
        
        hotspots.append({
            'center': np.array([cx, cy]),
            'radius': radius,
            'intensity': intensity,
            'name': f'Hotspot-{i+1}'
        })
    
    return hotspots


def render_hotspot_map(hotspots, bounds):
    fig, ax = plt.subplots(figsize=(10, 8))
    (x_min, x_max), (y_min, y_max) = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    
    colors = plt.cm.Reds
    for h in hotspots:
        circle = Circle(h['center'], h['radius'],
                        facecolor=colors(h['intensity']),
                        edgecolor='darkred', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        ax.text(h['center'][0], h['center'][1], h['name'],
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_title('Traffic Hotspot Zones', fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig, width="stretch")
    plt.close(fig)
