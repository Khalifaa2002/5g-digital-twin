"""
5G NR / LTE Digital Twin - Dashboard Modules Package
Research-grade network intelligence platform modules

Modules:
    ui_core          - Shared UI utilities, dark theme, card renderer
    kpi_engine       - Telecom-grade KPI computations (NOC style)
    radio_map        - Real-time animated radio map (ns-3 style)
    ai_prediction    - LSTM-like SINR prediction engine
    mobility_engine  - City-scale digital twin & realistic mobility
"""

from . import ui_core
from . import kpi_engine
from . import radio_map
from . import ai_prediction
from . import mobility_engine

__version__ = "3.0"
__all__ = ["ui_core", "kpi_engine", "radio_map", "ai_prediction", "mobility_engine"]

