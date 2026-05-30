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
    grade: str
    recommendation: str
    confidence: str
    risk_flags: List[str]
    top_features: List[str]
    shap_values: Optional[dict] = None


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
    grade: str
    soh: float
    chemistry: str = "NMC"
    module_count: int = 16
    cell_count: int = 192
    rated_capacity_kwh: float = 75.0
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
    soh: float
    grade: str
    materials_recovered_pct: float = 0.0
    carbon_avoided_kg: float = 0.0
    second_life_potential: bool = True


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

