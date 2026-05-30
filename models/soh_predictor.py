"""
Battery State of Health (SOH) Prediction Model
Uses XGBoost with synthetic training data for hackathon demo.
In production, this would be trained on NASA/CALCE battery aging datasets.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import json
import os

# Global model and scaler
_model = None
_scaler = None


def _generate_training_data(n_samples=2000):
    """Generate synthetic battery aging data for demo purposes."""
    np.random.seed(42)

    cycle_counts = np.random.uniform(100, 3000, n_samples)
    voltages = np.random.uniform(3.0, 4.2, n_samples)
    currents = np.random.uniform(0.5, 5.0, n_samples)
    temperatures = np.random.uniform(15, 45, n_samples)
    charge_capacities = np.random.uniform(20, 100, n_samples)
    internal_resistances = np.random.uniform(10, 200, n_samples)
    depths_of_discharge = np.random.uniform(50, 100, n_samples)
    max_temperatures = temperatures + np.random.uniform(5, 20, n_samples)
    energy_throughputs = cycle_counts * np.random.uniform(0.5, 2.0, n_samples)

    # SOH degrades with cycles, high temperature, high resistance
    base_soh = 100.0
    cycle_degradation = cycle_counts * np.random.uniform(0.005, 0.015, n_samples)
    temp_degradation = np.where(temperatures > 35, (temperatures - 35) * 0.3, 0)
    resistance_degradation = (internal_resistances - 10) * 0.05
    dod_degradation = (depths_of_discharge - 50) * 0.02

    soh = base_soh - cycle_degradation - temp_degradation - resistance_degradation - dod_degradation
    soh = np.clip(soh + np.random.normal(0, 2, n_samples), 20, 100)

    discharge_capacities = charge_capacities * (soh / 100) * np.random.uniform(0.95, 1.0, n_samples)
    rated_capacities = charge_capacities * np.random.uniform(1.0, 1.1, n_samples)

    data = pd.DataFrame({
        'cycle_count': cycle_counts,
        'voltage': voltages,
        'current': currents,
        'temperature': temperatures,
        'charge_capacity': charge_capacities,
        'discharge_capacity': discharge_capacities,
        'internal_resistance': internal_resistances,
        'rated_capacity': rated_capacities,
        'depth_of_discharge': depths_of_discharge,
        'max_temperature': max_temperatures,
        'energy_throughput': energy_throughputs,
    })

    return data, soh


def _train_model():
    """Train the SOH prediction model."""
    global _model, _scaler

    X, y = _generate_training_data()

    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    _model.fit(X_scaled, y)

    return _model, _scaler


def get_model():
    """Get or train the SOH prediction model."""
    global _model, _scaler
    if _model is None:
        _train_model()
    return _model, _scaler


def predict_soh(input_data: dict) -> dict:
    """
    Predict battery State of Health.

    Returns predicted SOH, RUL, confidence, and feature importances.
    """
    model, scaler = get_model()

    features = np.array([[
        input_data['cycle_count'],
        input_data['voltage'],
        input_data['current'],
        input_data['temperature'],
        input_data['charge_capacity'],
        input_data['discharge_capacity'],
        input_data['internal_resistance'],
        input_data['rated_capacity'],
        input_data['depth_of_discharge'],
        input_data['max_temperature'],
        input_data['energy_throughput'],
    ]])

    features_scaled = scaler.transform(features)
    predicted_soh = float(model.predict(features_scaled)[0])
    predicted_soh = max(0.0, min(100.0, predicted_soh))

    # Estimate RUL based on SOH and degradation rate
    if input_data['cycle_count'] > 0:
        degradation_rate = (100 - predicted_soh) / input_data['cycle_count']
        if degradation_rate > 0:
            rul_cycles = int((predicted_soh - 60) / degradation_rate)  # Until 60% SOH
            rul_cycles = max(0, rul_cycles)
        else:
            rul_cycles = 5000
    else:
        rul_cycles = 5000

    # Feature importances
    feature_names = [
        'cycle_count', 'voltage', 'current', 'temperature',
        'charge_capacity', 'discharge_capacity', 'internal_resistance',
        'rated_capacity', 'depth_of_discharge', 'max_temperature',
        'energy_throughput'
    ]
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    top_features = [feature_names[i] for i in top_indices]

    # Confidence estimation
    if predicted_soh > 85 or predicted_soh < 50:
        confidence = "High"
    elif predicted_soh > 70:
        confidence = "Medium"
    else:
        confidence = "Medium"

    # SHAP-like feature contributions (simplified)
    shap_values = {}
    for i, name in enumerate(feature_names):
        shap_values[name] = round(float(importances[i] * 100), 2)

    return {
        'predicted_soh': round(predicted_soh, 1),
        'rul_cycles': rul_cycles,
        'confidence': confidence,
        'top_features': top_features,
        'shap_values': shap_values,
        'feature_importances': {feature_names[i]: round(float(importances[i]), 4) for i in range(len(feature_names))}
    }

