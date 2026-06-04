
"""
LangGraph Circularity Agent for CircuVolt AI.

This agent uses LangGraph's ReAct pattern to orchestrate multi-step
battery circularity workflows. It connects to the MCP server for
domain tools and provides a conversational interface via CopilotKit.

Architecture:
  User (CopilotKit/AG-UI) → LangGraph Agent → MCP Tools → ML Models
"""
import os
import json
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Import domain functions directly (in-process MCP alternative for reliability)
from models.soh_predictor import predict_soh, get_model
from models.grading_engine import calculate_grade
from models.passport_generator import generate_passport
from models.recovery_optimizer import generate_recovery_plan
from models.recyclability_advisor import calculate_recyclability_score
from models.carbon_calculator import calculate_circularity_score
from utils.qr_generator import generate_qr_code

# Pre-train model
get_model()

# ─── System Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CircularDrive AI, an expert AI assistant for EV battery circularity intelligence.

You help users with:
1. **Battery SOH Prediction** - Predict State of Health and Remaining Useful Life
2. **Second-Life Grading** - Assign A/B/C/D grades with safety overrides
3. **Material Passport Generation** - Create EU Regulation 2023/1542 compliant passports
4. **Recovery Planning** - Intelligent disassembly and material recovery optimization
5. **Circularity Scoring** - Quantify circular economy impact
6. **Design for Recyclability** - Suggest improvements for end-of-life processing

## Key Rules
- Always use tools to perform calculations. Never make up numbers.
- Present results clearly with the grade, SOH percentage, and recommendation.
- Flag any safety risks prominently.
- When doing a full assessment, run SOH prediction first, then grading, then passport/recovery.
- Remind users this is a prototype, not a certified safety assessment.
- Use metric units (kg, kWh, °C, mΩ).

## Grading Quick Reference
- Grade A (≥85% SOH): Continue EV use or premium second-life
- Grade B (70-85%): Stationary energy storage
- Grade C (60-70%): Module-level refurbishment
- Grade D (<60%): Direct recycling

## EU Battery Passport Context
Under EU Regulation 2023/1542 (effective Feb 2027), EV batteries must have
an electronic passport with identity, health, materials, lifecycle, and
traceability data, accessible via QR code.
"""


# ─── LangChain Tool Definitions ──────────────────────────────────

@tool
def predict_battery_soh(
    cycle_count: float,
    voltage: float,
    current: float,
    temperature: float,
    charge_capacity: float,
    discharge_capacity: float,
    internal_resistance: float,
    rated_capacity: float,
    depth_of_discharge: float = 80.0,
    max_temperature: float = 35.0,
    energy_throughput: float = 0.0,
) -> str:
    """Predict battery State of Health (SOH) and Remaining Useful Life (RUL).

    Use this when the user wants to assess a battery's health.

    Args:
        cycle_count: Number of charge/discharge cycles completed
        voltage: Current voltage in Volts (3.0-4.2V typical)
        current: Current in Amperes
        temperature: Average operating temperature in Celsius
        charge_capacity: Current charge capacity in Ah
        discharge_capacity: Current discharge capacity in Ah
        internal_resistance: Internal resistance in milliOhms
        rated_capacity: Original rated capacity in Ah
        depth_of_discharge: Depth of discharge percentage (0-100)
        max_temperature: Maximum recorded temperature in Celsius
        energy_throughput: Total energy throughput in kWh
    """
    input_data = {
        "cycle_count": cycle_count,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "charge_capacity": charge_capacity,
        "discharge_capacity": discharge_capacity,
        "internal_resistance": internal_resistance,
        "rated_capacity": rated_capacity,
        "depth_of_discharge": depth_of_discharge,
        "max_temperature": max_temperature,
        "energy_throughput": energy_throughput,
    }
    result = predict_soh(input_data)
    return json.dumps(result, indent=2)


@tool
def grade_battery(
    soh: float,
    internal_resistance: float = 50.0,
    max_temperature: float = 35.0,
    cycle_count: float = 500.0,
    has_service_history: bool = True,
) -> str:
    """Assign a second-life grade (A/B/C/D) based on SOH and safety factors.

    Grades: A(≥85%), B(70-85%), C(60-70%), D(<60%).
    Safety overrides may downgrade for high resistance, thermal events, etc.

    Args:
        soh: Predicted State of Health percentage
        internal_resistance: Internal resistance in milliOhms
        max_temperature: Maximum recorded temperature in Celsius
        cycle_count: Number of charge/discharge cycles
        has_service_history: Whether service records exist
    """
    result = calculate_grade(soh, internal_resistance, max_temperature, cycle_count, has_service_history)
    return json.dumps(result, indent=2)


@tool
def generate_material_passport(
    component_id: str,
    manufacturer: str = "EV Motors",
    model: str = "ElectraX 75",
    chemistry: str = "NMC",
    rated_capacity_kwh: float = 75.0,
    predicted_soh: float = 0.0,
) -> str:
    """Generate an EU Regulation 2023/1542 compliant material passport.

    Creates a digital record with identity, specs, health, materials,
    lifecycle, sustainability, and traceability. Includes QR code.

    Args:
        component_id: Unique component identifier
        manufacturer: Manufacturer name
        model: Battery/vehicle model
        chemistry: Battery chemistry (NMC, LFP, NCA)
        rated_capacity_kwh: Rated capacity in kWh
        predicted_soh: SOH if already predicted (0 if not)
    """
    passport_input = {
        "component_id": component_id,
        "battery_id": f"BAT-{component_id}",
        "vehicle_id": f"VEH-{component_id}",
        "manufacturer": manufacturer,
        "model": model,
        "chemistry": chemistry,
        "rated_capacity_kwh": rated_capacity_kwh,
        "voltage": 400.0,
        "module_count": 16,
        "cell_count": 192,
        "manufacturing_date": "2022-01-15",
    }

    soh_data = None
    grade_data = None
    if predicted_soh > 0:
        soh_data = {"predicted_soh": predicted_soh, "rul_cycles": 1200, "confidence": "Medium"}
        grade_data = calculate_grade(soh=predicted_soh)

    result = generate_passport(passport_input, soh_data=soh_data, grade_data=grade_data)

    qr_url = f"http://localhost:8000/passport/{component_id}"
    qr_b64 = generate_qr_code(qr_url, component_id)
    result["qr_code_data_url"] = f"data:image/png;base64,{qr_b64}"

    return json.dumps(result, indent=2, default=str)


@tool
def plan_recovery(
    component_id: str,
    grade: str,
    soh: float,
    chemistry: str = "NMC",
    module_count: int = 16,
) -> str:
    """Generate disassembly and material recovery plan.

    Produces step-by-step disassembly, material recovery, carbon impact,
    and economic value analysis.

    Args:
        component_id: Component identifier
        grade: Battery grade (A, B, C, or D)
        soh: Predicted SOH percentage
        chemistry: Battery chemistry
        module_count: Number of modules
    """
    result = generate_recovery_plan(grade=grade, soh=soh, chemistry=chemistry, module_count=module_count)
    return json.dumps(result, indent=2)


@tool
def calculate_circularity(
    component_id: str,
    soh: float,
    grade: str,
    materials_recovered_pct: float = 80.0,
    carbon_avoided_kg: float = 420.0,
) -> str:
    """Calculate overall circularity score (0-100) for a component.

    Args:
        component_id: Component identifier
        soh: SOH percentage
        grade: Battery grade
        materials_recovered_pct: Material recovery percentage
        carbon_avoided_kg: CO2 avoided in kg
    """
    result = calculate_circularity_score({
        "soh": soh, "grade": grade,
        "materials_recovered_pct": materials_recovered_pct,
        "carbon_avoided_kg": carbon_avoided_kg,
        "second_life_potential": grade in ["A", "B"],
    })
    result["component_id"] = component_id
    return json.dumps(result, indent=2)


@tool
def analyze_recyclability(
    component_name: str = "EV Battery Pack",
    fastener_count: int = 42,
    adhesive_use: str = "High",
    material_mix: str = "Aluminum, Plastic, Steel, Copper",
    labeling_quality: str = "Poor",
    modularity: str = "Low",
    hazard_separation: str = "Difficult",
) -> str:
    """Analyze component design for recyclability and suggest improvements.

    Args:
        component_name: Component name
        fastener_count: Number of fasteners
        adhesive_use: Low/Medium/High
        material_mix: Comma-separated material list
        labeling_quality: Poor/Fair/Good/Excellent
        modularity: Low/Medium/High
        hazard_separation: Easy/Moderate/Difficult
    """
    result = calculate_recyclability_score({
        "component_name": component_name,
        "fastener_count": fastener_count,
        "adhesive_use": adhesive_use,
        "material_mix": [m.strip() for m in material_mix.split(",")],
        "labeling_quality": labeling_quality,
        "modularity": modularity,
        "hazard_separation": hazard_separation,
    })
    return json.dumps(result, indent=2)


@tool
def get_sample_battery(scenario: str = "moderate") -> str:
    """Get sample battery data for demo. Scenario: healthy, moderate, or degraded.

    Args:
        scenario: One of 'healthy', 'moderate', or 'degraded'
    """
    samples = {
        "healthy": {
            "name": "Healthy Retired Battery", "cycle_count": 450, "voltage": 3.85,
            "current": 2.1, "temperature": 28.0, "charge_capacity": 92.0,
            "discharge_capacity": 89.5, "internal_resistance": 35.0,
            "rated_capacity": 95.0, "depth_of_discharge": 75.0,
            "max_temperature": 38.0, "energy_throughput": 680.0,
        },
        "moderate": {
            "name": "Moderate Battery", "cycle_count": 1200, "voltage": 3.72,
            "current": 2.8, "temperature": 32.0, "charge_capacity": 78.0,
            "discharge_capacity": 74.0, "internal_resistance": 65.0,
            "rated_capacity": 95.0, "depth_of_discharge": 85.0,
            "max_temperature": 42.0, "energy_throughput": 1800.0,
        },
        "degraded": {
            "name": "Degraded/Risky Battery", "cycle_count": 2500, "voltage": 3.45,
            "current": 3.5, "temperature": 38.0, "charge_capacity": 55.0,
            "discharge_capacity": 50.0, "internal_resistance": 120.0,
            "rated_capacity": 95.0, "depth_of_discharge": 90.0,
            "max_temperature": 52.0, "energy_throughput": 4200.0,
        },
    }
    return json.dumps(samples.get(scenario.lower(), samples["moderate"]), indent=2)


# ─── Tool Registry ───────────────────────────────────────────────

ALL_TOOLS = [
    predict_battery_soh,
    grade_battery,
    generate_material_passport,
    plan_recovery,
    calculate_circularity,
    analyze_recyclability,
    get_sample_battery,
]


# ─── LangGraph Agent Factory ─────────────────────────────────────

def create_circularity_agent(model=None):
    """Create the LangGraph ReAct agent for circularity intelligence.

    If no model is provided, uses the configured LLM provider:
      - LLM_PROVIDER=target  → Target internal API (corporate wrapper)
      - LLM_PROVIDER=openai  → OpenAI / compatible endpoint (direct)
    """
    if model is None:
        from llm_provider import get_llm
        model = get_llm()

    agent = create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    return agent

