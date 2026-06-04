"""
Battery State of Health (SOH) Prediction Model
Uses XGBoost with advanced synthetic non-linear training data modeling.
Includes Second-Life Grading, Non-Linear RUL Estimation, and Dynamic Confidence.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import json
import os

# Global model and scaler
_model = None
_scaler = None


def _generate_training_data(n_samples=2500):
    """Generate synthetic battery aging data with non-linear exponential degradation."""
    np.random.seed(42)

    cycle_counts = np.random.uniform(0, 4000, n_samples)
    voltages = np.random.uniform(3.0, 4.2, n_samples)
    currents = np.random.uniform(0.5, 6.0, n_samples)
    temperatures = np.random.uniform(10, 55, n_samples)
    charge_capacities = np.random.uniform(30, 120, n_samples)
    
    # Internal resistance increases non-linearly with cycles and temperature
    internal_resistances = 10 + (cycle_counts / 1000)**1.5 * np.random.uniform(2, 5, n_samples) + \
                           np.where(temperatures > 40, (temperatures - 40)**2 * 0.1, 0)
                           
    depths_of_discharge = np.random.uniform(20, 100, n_samples)
    max_temperatures = temperatures + np.random.uniform(2, 25, n_samples)
    energy_throughputs = cycle_counts * np.random.uniform(0.5, 2.5, n_samples)

    # SOH degrades non-linearly. The "knee" effect happens at high cycles or high resistance.
    base_soh = 100.0
    
    # Linear phase
    cycle_degradation = cycle_counts * 0.005 
    
    # Exponential "knee" phase
    knee_effect = (np.maximum(0, cycle_counts - 2000) / 1000)**2.5 * 10
    
    # Stressors
    temp_stress = np.maximum(0, max_temperatures - 45)**1.2 * 0.5
    dod_stress = (depths_of_discharge / 100)**2 * (cycle_counts / 1000) * 2
    ir_stress = np.maximum(0, internal_resistances - 50) * 0.2

    soh = base_soh - cycle_degradation - knee_effect - temp_stress - dod_stress - ir_stress
    
    # Add natural variance
    soh = np.clip(soh + np.random.normal(0, 1.5, n_samples), 10, 100)

    discharge_capacities = charge_capacities * (soh / 100) * np.random.uniform(0.97, 1.0, n_samples)
    rated_capacities = charge_capacities * np.random.uniform(1.0, 1.05, n_samples)

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
    """Train the advanced XGBoost SOH prediction model."""
    global _model, _scaler

    X, y = _generate_training_data()

    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
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
    Predict battery State of Health with Second-Life Grading and Non-Linear RUL.
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

    # Safety & Grading Logic
    temp = input_data['temperature']
    res = input_data['internal_resistance']
    max_temp = input_data.get('max_temperature', temp)
    
    safety_status = "pass_with_monitoring"
    risk_flags = []
    
    if max_temp > 60:
        safety_status = "quarantine_thermal_risk"
        risk_flags.append("High thermal event history")
    if res > 100:
        safety_status = "quarantine_high_resistance"
        risk_flags.append("Internal resistance exceeded safe limits")
        
    if safety_status != "pass_with_monitoring":
        grade = "Reject / R"
        recommendation = "End of Life - Safety failure overrides SOH."
        recommended_route = "quarantine_then_recycle"
    elif predicted_soh >= 85.0:
        grade = "Grade A"
        recommendation = "Excellent - Suitable for EV reuse or high-demand applications."
        recommended_route = "remanufacture_for_vehicle"
    elif predicted_soh >= 70.0:
        grade = "Grade B"
        recommendation = "Good - Suitable for Stationary Battery Energy Storage Systems (BESS)."
        recommended_route = "second_life_stationary_storage"
    elif predicted_soh >= 60.0:
        grade = "Grade C"
        recommendation = "Fair - Module-level refurbishment or limited backup use."
        recommended_route = "low_duty_storage"
    else:
        grade = "Grade D"
        recommendation = "End of Life - Recommended for Material Recycling and Recovery."
        recommended_route = "direct_material_recycling"

    # Non-linear RUL Estimation
    # Assume EOL threshold is 60% SOH. Use an exponential curve decay.
    # Current SOH = 100 * e^(-k * cycle_count). We find 'k' based on current state.
    cycle_count = input_data['cycle_count']
    if cycle_count > 10 and predicted_soh < 99:
        # Solve for decay constant k: k = -ln(SOH/100) / cycles
        k = -np.log(predicted_soh / 100.0) / cycle_count
        # Target cycles at 60% SOH: cycles = -ln(0.60) / k
        target_cycles = -np.log(0.60) / k
        rul_cycles = int(target_cycles - cycle_count)
        rul_cycles = max(0, rul_cycles)
    else:
        # If very new, assume roughly 2500-4000 total cycles
        rul_cycles = int(3500 - cycle_count)
        rul_cycles = max(0, rul_cycles)
        
    # Assuming stationary storage operates at approx 1 full cycle equivalent per day
    rul_years_stationary = round(rul_cycles / 365.0, 1)

    # Dynamic Confidence Estimation (Penalize outliers)
    if temp > 50 or temp < 0 or res > 150:
        confidence = 0.65
    elif temp > 40 or res > 100 or predicted_soh < 50:
        confidence = 0.78
    else:
        confidence = 0.92

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

    # SHAP-like feature contributions
    shap_values = {}
    for i, name in enumerate(feature_names):
        shap_values[name] = round(float(importances[i] * 100), 2)

    return {
        'predicted_soh': round(predicted_soh, 1),
        'rul_cycles': rul_cycles,
        'rul_years_stationary': rul_years_stationary,
        'grade': grade,
        'safety_status': safety_status,
        'recommendation': recommendation,
        'recommended_route': recommended_route,
        'confidence': confidence,
        'risk_flags': risk_flags,
        'top_features': top_features,
        'shap_values': shap_values,
        'feature_importances': {feature_names[i]: round(float(importances[i]), 4) for i in range(len(feature_names))}
    }
