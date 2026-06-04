from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class BatteryInput(BaseModel):
    component_id: Optional[str] = None
    cycle_count: float = Field(..., description="Number of charge/discharge cycles")
    voltage: float = Field(..., description="Current voltage (V)")
    current: float = Field(..., description="Current (A)")
    temperature: float = Field(..., description="Average temperature (°C)")
    charge_capacity: float = Field(..., description="Charge capacity (Ah)")
    discharge_capacity: float = Field(..., description="Discharge capacity (Ah)")
    internal_resistance: float = Field(..., description="Internal resistance (mΩ)")
    rated_capacity: float = Field(..., description="Rated capacity (Ah)")
    depth_of_discharge: float = Field(default=80.0, description="Depth of discharge (%)")
    max_temperature: float = Field(default=35.0, description="Max recorded temperature (°C)")
    energy_throughput: float = Field(default=0.0, description="Total energy throughput (kWh)")


class SOHPredictionResponse(BaseModel):
    component_id: str
    predicted_soh: float
    rul_cycles: int
    rul_years_stationary: float = 0.0
    grade: str
    safety_status: str = "pass_with_monitoring"
    recommendation: str
    recommended_route: str = "second_life_stationary_storage"
    confidence: float = 0.85
    risk_flags: List[str]
    top_features: List[str]
    shap_values: Optional[dict] = None

class DisassemblyInput(BaseModel):
    vehicle_id: str = "VIN_HASH_8291"
    make: str = "Generic"
    model: str = "Model EV"
    year: int = 2020
    accident_condition: str = "Frontal Impact"
    passport_status: str = "incomplete"

class DisassemblyResponse(BaseModel):
    vehicle_id: str
    make: str = "Generic"
    model: str = "Model EV"
    year: int = 2020
    accident_condition: str = "None"
    priority_parts: List[dict]  # each: {part, route, weight_kg, value_usd, material, recyclability_pct, safety_risk}
    depollution_steps: List[dict]  # each: {step, action, reason, duration_min, safety_level}
    disassembly_sequence: List[dict]  # each: {step, component, tool_required, time_min, safety_note}
    estimated_revenue: float
    estimated_co2e_saving_kg: float
    total_weight_kg: float
    material_breakdown: List[dict]  # each: {material, weight_kg, recovery_rate_pct, value_per_kg}
    passport_status: str
    risk_score: float = 0.0  # 0-100, higher = more risk
    automation_feasibility: float = 0.0  # 0-100, how much can be automated


class PassportInput(BaseModel):
    component_id: str
    battery_id: str = "BAT-001"
    vehicle_id: str = "VEH-001"
    manufacturer: str = "Generic EV Manufacturer"
    model: str = "EV Model X"
    chemistry: str = "NMC (Nickel Manganese Cobalt)"
    rated_capacity_kwh: float = 75.0
    voltage: float = 400.0
    module_count: int = 16
    cell_count: int = 192
    manufacturing_date: str = "2020-01-15"
    service_history: List[str] = []
    materials: Optional[dict] = None


class PassportResponse(BaseModel):
    component_id: str
    passport_data: dict
    qr_code_url: str
    completeness_score: float


class RecoveryInput(BaseModel):
    component_id: str
    component_type: str = "EV Battery Pack"
    grade: str
    soh: float
    chemistry: str = "NMC"
    module_count: int = 16
    cell_count: int = 192
    rated_capacity_kwh: float = 75.0
    motor_type: Optional[str] = "PMSM"
    semiconductor_type: Optional[str] = "Silicon Carbide (SiC)"
    wear_level: Optional[str] = "Moderate"
    materials: Optional[dict] = None


class RecoveryResponse(BaseModel):
    component_id: str
    recovery_plan: List[dict]
    material_recovery: dict
    carbon_impact: dict
    economic_value: dict
    recovery_score: float


class CircularityInput(BaseModel):
    component_id: str
    component_type: str = "EV Battery Pack"
    soh: float
    grade: str
    materials_recovered_pct: float = 0.0
    carbon_avoided_kg: float = 0.0
    second_life_potential: bool = True
    recycled_content_pct: float = 0.0
    dfd_rating: float = 5.0
    origin: str = "Local"


class CircularityResponse(BaseModel):
    component_id: str
    circularity_score: float
    breakdown: dict


class DesignInput(BaseModel):
    component_name: str = "Battery Pack"
    fastener_count: int = 42
    adhesive_use: str = "High"  # Low, Medium, High
    material_mix: List[str] = ["Aluminum", "Plastic", "Steel", "Copper"]
    labeling_quality: str = "Poor"  # Poor, Fair, Good, Excellent
    modularity: str = "Low"  # Low, Medium, High
    hazard_separation: str = "Difficult"  # Easy, Moderate, Difficult


class DesignResponse(BaseModel):
    component_name: str
    recyclability_score: float
    suggestions: List[dict]
    priority_actions: List[str]

