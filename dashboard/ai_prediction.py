"""
AI-Driven SINR Prediction Engine (Simulated LSTM)
No real ML model - simulates LSTM behavior with exponential smoothing,
memory decay, and trend estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Tuple, Dict


def lstm_like_sinr_predict(history: np.ndarray, window: int = 10,
                           alpha: float = 0.7, forecast_steps: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate LSTM-like behavior using sliding window memory.
    
    Formula:
        pred(t+1) = alpha * current + (1-alpha) * past_mean + trend_factor
    
    Args:
        history: 1D array of historical SINR values (dB)
        window: Sliding window size for memory
        alpha: Weight for current value vs past mean (0-1)
        forecast_steps: Number of future steps to predict
    
    Returns:
        (fitted, forecast, lower_band, upper_band)
        - fitted: in-sample predictions aligned with history
        - forecast: future predictions (length = forecast_steps)
        - lower_band / upper_band: confidence bands based on prediction variance
    """
    if len(history) < window + 1:
        # Not enough data: return simple extrapolation
        fitted = history.copy()
        last_val = history[-1] if len(history) > 0 else 0
        forecast = np.full(forecast_steps, last_val)
        std_est = np.std(history) if len(history) > 1 else 3.0
        return fitted, forecast, forecast - 1.96*std_est, forecast + 1.96*std_est
    
    # In-sample fitted predictions
    fitted = np.zeros_like(history)
    fitted[:window] = history[:window]  # Warmup: use actual values
    
    # Memory decay factor (exponential decay of older values)
    decay = np.exp(-np.arange(window) / (window / 2))
    decay = decay / np.sum(decay)
    
    predictions = []
    errors = []
    
    for i in range(window, len(history)):
        window_data = history[i-window:i]
        
        # Compute weighted past mean with decay
        past_mean = np.sum(window_data * decay)
        
        # Trend estimation: linear slope of recent 5 points
        if i >= 5:
            recent = history[i-5:i]
            x = np.arange(5)
            slope = np.sum((x - np.mean(x)) * (recent - np.mean(recent))) / (np.sum((x - np.mean(x))**2) + 1e-9)
            trend_factor = slope
        else:
            trend_factor = history[i-1] - history[i-2] if i >= 2 else 0
        
        # Temperature-based noise injection (simulates LSTM uncertainty)
        noise_scale = 0.1 * np.std(window_data)
        noise = np.random.normal(0, noise_scale)
        
        # LSTM-like prediction formula
        pred = alpha * history[i-1] + (1 - alpha) * past_mean + trend_factor + noise
        fitted[i] = pred
        predictions.append(pred)
        errors.append(abs(pred - history[i]))
    
    # Compute prediction variance for confidence bands
    mae = np.mean(errors) if len(errors) > 0 else 3.0
    variance = np.var(errors) if len(errors) > 0 else 9.0
    std_pred = np.sqrt(variance) + 1.0  # Add base uncertainty
    
    # Multi-step forecast
    forecast = np.zeros(forecast_steps)
    last_window = list(history[-window:])
    
    for step in range(forecast_steps):
        # Update window: shift and add last prediction
        if step > 0:
            last_window.pop(0)
            last_window.append(forecast[step-1])
        
        window_arr = np.array(last_window)
        past_mean = np.sum(window_arr * decay)
        
        # Trend from last 5 actual/predicted values
        if len(window_arr) >= 5:
            recent = np.array(last_window[-5:])
            x = np.arange(5)
            slope = np.sum((x - np.mean(x)) * (recent - np.mean(recent))) / (np.sum((x - np.mean(x))**2) + 1e-9)
            trend_factor = slope
        else:
            trend_factor = 0
        
        # Increasing uncertainty for further steps
        step_noise = np.random.normal(0, std_pred * (1 + step * 0.15))
        
        forecast[step] = alpha * window_arr[-1] + (1 - alpha) * past_mean + trend_factor + step_noise
    
    # Smooth the forecast (LSTM-like state persistence)
    from scipy.ndimage import gaussian_filter1d
    forecast = gaussian_filter1d(forecast, sigma=1.0)
    
    # Confidence bands: widen with forecast horizon
    horizon_factor = 1 + np.arange(forecast_steps) * 0.2
    lower_band = forecast - 1.96 * std_pred * horizon_factor
    upper_band = forecast + 1.96 * std_pred * horizon_factor
    
    return fitted, forecast, lower_band, upper_band


@st.cache_data
def compute_prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """Compute prediction accuracy metrics."""
    # Only compare where both exist
    min_len = min(len(actual), len(predicted))
    a = actual[:min_len]
    p = predicted[:min_len]
    
    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p)**2))
    mape = 100 * np.mean(np.abs((a - p) / (np.abs(a) + 1e-6)))
    
    # R-squared
    ss_res = np.sum((a - p)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }


def render_ai_prediction_panel(sim_data: Dict):
    """
    Render the AI prediction panel with actual vs predicted curves,
    forecast with confidence bands, and error metrics.
    """
    import scipy.ndimage
    
    st.subheader("🧠 AI-Driven SINR Prediction (LSTM-like)")
    
    sinr_history = sim_data.get('sinr_time_series', np.array([]))
    
    if len(sinr_history) < 5:
        st.warning("Not enough data for prediction. Run simulation with more steps.")
        return
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        window = st.slider("Memory Window", 5, min(50, len(sinr_history)-1), 10,
                           help="Sliding window size for LSTM-like memory")
    with col2:
        alpha = st.slider("Current Weight (alpha)", 0.1, 0.95, 0.7, 0.05,
                          help="Balance between current value and past mean")
    with col3:
        forecast_steps = st.slider("Forecast Steps", 5, 50, 15,
                                   help="Number of future steps to predict")
    
    # Run prediction
    try:
        fitted, forecast, lower_band, upper_band = lstm_like_sinr_predict(
            sinr_history, window=window, alpha=alpha, forecast_steps=forecast_steps
        )
    except Exception:
        # Fallback if scipy not available
        fitted = np.zeros_like(sinr_history)
        fitted[:window] = sinr_history[:window]
        for i in range(window, len(sinr_history)):
            fitted[i] = alpha * sinr_history[i-1] + (1-alpha) * np.mean(sinr_history[i-window:i])
        
        # Simple trend-based forecast
        trend = (sinr_history[-1] - sinr_history[-min(10, len(sinr_history))]) / min(10, len(sinr_history))
        forecast = sinr_history[-1] + trend * np.arange(1, forecast_steps+1)
        std_est = np.std(sinr_history[-window:]) if len(sinr_history) >= window else 3.0
        lower_band = forecast - 1.96 * std_est
        upper_band = forecast + 1.96 * std_est
    
    # Prediction metrics
    metrics = compute_prediction_metrics(sinr_history[window:], fitted[window:])
    
    # Display metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("MAE", f"{metrics['mae']:.2f} dB")
    with mcol2:
        st.metric("RMSE", f"{metrics['rmse']:.2f} dB")
    with mcol3:
        st.metric("MAPE", f"{metrics['mape']:.1f}%")
    with mcol4:
        st.metric("R", f"{np.sqrt(max(metrics['r2'], 0)):.3f}")
    
    st.markdown("---")
    
    # Create comprehensive plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    timesteps = np.arange(len(sinr_history))
    forecast_timesteps = np.arange(len(sinr_history), len(sinr_history) + forecast_steps)
    
    # ---- TOP PLOT: Actual vs Predicted with Forecast ----
    ax1.plot(timesteps, sinr_history, 'b-', linewidth=2, label='Actual SINR', alpha=0.8)
    ax1.plot(timesteps, fitted, 'r--', linewidth=1.5, label='LSTM-like Predicted (in-sample)', alpha=0.7)
    
    # Forecast region
    ax1.axvline(x=len(sinr_history)-1, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_betweenx([np.min(sinr_history)-5, np.max(sinr_history)+5],
                       len(sinr_history)-1, len(sinr_history)+forecast_steps,
                       alpha=0.1, color='purple', label='Forecast Region')
    
    ax1.plot(forecast_timesteps, forecast, 'g-', linewidth=2.5, label='AI Forecast', marker='o', markersize=4)
    ax1.fill_between(forecast_timesteps, lower_band, upper_band, alpha=0.25, color='green',
                      label='95% Confidence Band')
    
    ax1.set_xlabel("Time Step", fontsize=11, fontweight='bold')
    ax1.set_ylabel("SINR (dB)", fontsize=11, fontweight='bold')
    ax1.set_title("🧠 Actual vs AI-Predicted SINR with Multi-Step Forecast", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    # ---- BOTTOM PLOT: Prediction Error and Uncertainty ----
    errors = sinr_history[window:] - fitted[window:]
    error_timesteps = timesteps[window:]
    
    ax2.plot(error_timesteps, np.abs(errors), 'purple', linewidth=1.5, alpha=0.7, label='|Prediction Error|')
    ax2.axhline(y=metrics['mae'], color='orange', linestyle='--', linewidth=2, label=f"MAE = {metrics['mae']:.2f} dB")
    ax2.fill_between(error_timesteps, 0, np.abs(errors), alpha=0.2, color='purple')
    
    ax2.set_xlabel("Time Step", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Absolute Error (dB)", fontsize=11, fontweight='bold')
    ax2.set_title("⚡ Prediction Error Analysis", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    
    # Forecast table
    st.markdown("### 📋 Forecast Values")
    forecast_df_data = {
        'Step': range(1, forecast_steps + 1),
        'Predicted SINR (dB)': np.round(forecast, 2),
        'Lower 95%': np.round(lower_band, 2),
        'Upper 95%': np.round(upper_band, 2)
    }
    
    import pandas as pd
    df = pd.DataFrame(forecast_df_data)
    st.dataframe(df,  width="stretch")

