"""
MCP (Model Context Protocol) Server for CircularDrive AI.

Exposes all circularity intelligence tools as MCP tools that can be
discovered and invoked by any MCP-compatible agent or client.

Protocol: https://modelcontextprotocol.io/
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from models.soh_predictor import predict_soh, get_model
from models.grading_engine import calculate_grade
from models.passport_generator import generate_passport
from models.recovery_optimizer import generate_recovery_plan
from models.recyclability_advisor import calculate_recyclability_score
from models.carbon_calculator import calculate_circularity_score
from utils.qr_generator import generate_qr_code

# Initialize model at import time
get_model()

mcp = FastMCP(
    "CircularDrive AI MCP Server",
    instructions="""You are the CircularDrive AI tool server.
You provide tools for EV battery circularity intelligence:
- Battery State of Health (SOH) prediction
- Second-life grading (A/B/C/D)
- Material passport generation (EU Reg 2023/1542)
- Disassembly & recovery optimization
- Circularity score calculation
- Design-for-recyclability analysis

Always use these tools when the user asks about battery health,
second-life decisions, material passports, recycling, or recovery planning.""",
)


@mcp.tool()
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
    """Predict battery State of Health (SOH) percentage and Remaining Useful Life (RUL).

    Use this tool when the user wants to assess a battery's health, predict its remaining
    life, or determine if it's suitable for second-life applications.

    Args:
        cycle_count: Number of charge/discharge cycles completed
        voltage: Current voltage in Volts (typical range: 3.0-4.2V)
        current: Current in Amperes
        temperature: Average operating temperature in Celsius
        charge_capacity: Current charge capacity in Ah
        discharge_capacity: Current discharge capacity in Ah
        internal_resistance: Internal resistance in milliOhms
        rated_capacity: Original rated capacity in Ah
        depth_of_discharge: Typical depth of discharge percentage (0-100)
        max_temperature: Maximum recorded temperature in Celsius
        energy_throughput: Total energy throughput in kWh

    Returns:
        JSON string with predicted SOH, RUL, confidence, and feature importances
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


@mcp.tool()
def grade_battery(
    soh: float,
    internal_resistance: float = 50.0,
    max_temperature: float = 35.0,
    cycle_count: float = 500.0,
    has_service_history: bool = True,
) -> str:
    """Assign a second-life grade (A/B/C/D) to a battery based on its health and safety factors.

    Grading logic:
      A (>=85%): Continue EV use or premium second-life
      B (70-85%): Stationary energy storage
      C (60-70%): Module-level refurbishment
      D (<60%): Direct recycling / material recovery

    Safety overrides may downgrade the battery if high resistance, thermal events,
    rapid capacity fade, or missing service history are detected.

    Args:
        soh: Predicted State of Health percentage (0-100)
        internal_resistance: Internal resistance in milliOhms
        max_temperature: Maximum recorded temperature in Celsius
        cycle_count: Number of charge/discharge cycles
        has_service_history: Whether service history records are available

    Returns:
        JSON string with grade, recommendation, risk flags, and confidence
    """
    result = calculate_grade(soh, internal_resistance, max_temperature, cycle_count, has_service_history)
    return json.dumps(result, indent=2)


@mcp.tool()
def generate_material_passport(
    component_id: str,
    manufacturer: str = "EV Motors",
    model: str = "ElectraX 75",
    chemistry: str = "NMC (Nickel Manganese Cobalt)",
    rated_capacity_kwh: float = 75.0,
    voltage: float = 400.0,
    module_count: int = 16,
    cell_count: int = 192,
    manufacturing_date: str = "2022-01-15",
    predicted_soh: float = 0.0,
) -> str:
    """Generate an EU Regulation 2023/1542 compliant material passport for a battery.

    Creates a comprehensive digital record with identity, technical specs, health data,
    material composition, lifecycle info, sustainability metrics, and traceability.
    Includes a QR code URL for digital access.

    Args:
        component_id: Unique identifier for the component (e.g., BAT-IND-2026-001)
        manufacturer: Battery/vehicle manufacturer name
        model: Battery/vehicle model name
        chemistry: Battery chemistry type (e.g., NMC, LFP, NCA)
        rated_capacity_kwh: Rated capacity in kilowatt-hours
        voltage: Nominal voltage in Volts
        module_count: Number of modules in the battery pack
        cell_count: Number of cells in the battery pack
        manufacturing_date: Manufacturing date (YYYY-MM-DD)
        predicted_soh: Predicted SOH if available (0 if not yet predicted)

    Returns:
        JSON string with full passport data including all EU-mandated sections
    """
    passport_input = {
        "component_id": component_id,
        "battery_id": f"BAT-{component_id}",
        "vehicle_id": f"VEH-{component_id}",
        "manufacturer": manufacturer,
        "model": model,
        "chemistry": chemistry,
        "rated_capacity_kwh": rated_capacity_kwh,
        "voltage": voltage,
        "module_count": module_count,
        "cell_count": cell_count,
        "manufacturing_date": manufacturing_date,
    }

    soh_data = None
    grade_data = None
    if predicted_soh > 0:
        soh_data = {"predicted_soh": predicted_soh, "rul_cycles": 1200, "confidence": "Medium"}
        grade_data = calculate_grade(soh=predicted_soh)

    result = generate_passport(passport_input, soh_data=soh_data, grade_data=grade_data)

    # Generate QR code
    qr_url = f"http://localhost:8000/passport/{component_id}"
    qr_base64 = generate_qr_code(qr_url, component_id)
    result["qr_code_data_url"] = f"data:image/png;base64,{qr_base64}"

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def plan_recovery(
    component_id: str,
    grade: str,
    soh: float,
    chemistry: str = "NMC",
    module_count: int = 16,
) -> str:
    """Generate an intelligent disassembly and material recovery plan for a battery.

    Produces a step-by-step disassembly sequence, material recovery estimates,
    carbon impact analysis, and economic value calculation.

    Args:
        component_id: Component identifier
        grade: Battery grade (A, B, C, or D)
        soh: Predicted State of Health percentage
        chemistry: Battery chemistry type
        module_count: Number of modules in the pack

    Returns:
        JSON string with recovery plan steps, material recovery, carbon impact, and economics
    """
    result = generate_recovery_plan(
        grade=grade, soh=soh, chemistry=chemistry, module_count=module_count
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def calculate_circularity(
    component_id: str,
    soh: float,
    grade: str,
    materials_recovered_pct: float = 80.0,
    carbon_avoided_kg: float = 420.0,
    second_life_potential: bool = True,
) -> str:
    """Calculate the overall circularity score for a component (0-100).

    Combines material recovery, lifetime extension, carbon avoidance,
    and value retention into a single circularity metric.

    Args:
        component_id: Component identifier
        soh: Predicted State of Health percentage
        grade: Battery grade (A, B, C, or D)
        materials_recovered_pct: Percentage of materials that can be recovered
        carbon_avoided_kg: Estimated CO2 avoided in kg
        second_life_potential: Whether the battery has second-life potential

    Returns:
        JSON string with circularity score and breakdown
    """
    result = calculate_circularity_score({
        "soh": soh,
        "grade": grade,
        "materials_recovered_pct": materials_recovered_pct,
        "carbon_avoided_kg": carbon_avoided_kg,
        "second_life_potential": second_life_potential,
    })
    result["component_id"] = component_id
    return json.dumps(result, indent=2)


@mcp.tool()
def analyze_recyclability(
    component_name: str = "EV Battery Pack",
    fastener_count: int = 42,
    adhesive_use: str = "High",
    material_mix: str = "Aluminum, Plastic, Steel, Copper",
    labeling_quality: str = "Poor",
    modularity: str = "Low",
    hazard_separation: str = "Difficult",
) -> str:
    """Analyze a component's design for recyclability and suggest improvements.

    Evaluates design attributes and generates a recyclability score (0-100)
    with prioritized suggestions for improving end-of-life processing.

    Args:
        component_name: Name of the component being analyzed
        fastener_count: Number of fasteners used
        adhesive_use: Level of adhesive usage (Low, Medium, High)
        material_mix: Comma-separated list of materials used
        labeling_quality: Quality of material labels (Poor, Fair, Good, Excellent)
        modularity: Level of modularity (Low, Medium, High)
        hazard_separation: Difficulty of hazard separation (Easy, Moderate, Difficult)

    Returns:
        JSON string with recyclability score, suggestions, and priority actions
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


@mcp.tool()
def get_sample_battery_data(scenario: str = "moderate") -> str:
    """Get sample battery data for demonstration purposes.

    Args:
        scenario: Which scenario to load - "healthy", "moderate", or "degraded"

    Returns:
        JSON string with sample battery parameters
    """
    samples = {
        "healthy": {
            "name": "Healthy Retired Battery",
            "cycle_count": 450, "voltage": 3.85, "current": 2.1,
            "temperature": 28.0, "charge_capacity": 92.0,
            "discharge_capacity": 89.5, "internal_resistance": 35.0,
            "rated_capacity": 95.0, "depth_of_discharge": 75.0,
            "max_temperature": 38.0, "energy_throughput": 680.0,
        },
        "moderate": {
            "name": "Moderate Battery",
            "cycle_count": 1200, "voltage": 3.72, "current": 2.8,
            "temperature": 32.0, "charge_capacity": 78.0,
            "discharge_capacity": 74.0, "internal_resistance": 65.0,
            "rated_capacity": 95.0, "depth_of_discharge": 85.0,
            "max_temperature": 42.0, "energy_throughput": 1800.0,
        },
        "degraded": {
            "name": "Degraded/Risky Battery",
            "cycle_count": 2500, "voltage": 3.45, "current": 3.5,
            "temperature": 38.0, "charge_capacity": 55.0,
            "discharge_capacity": 50.0, "internal_resistance": 120.0,
            "rated_capacity": 95.0, "depth_of_discharge": 90.0,
            "max_temperature": 52.0, "energy_throughput": 4200.0,
        },
    }
    data = samples.get(scenario.lower(), samples["moderate"])
    return json.dumps(data, indent=2)


# MCP Resources - provide context to agents
@mcp.resource("circulardrive://grading-rules")
def grading_rules() -> str:
    """EU-aligned battery grading rules used by CircularDrive AI."""
    return """
# CircularDrive AI - Battery Second-Life Grading Rules

## Grade Definitions
- **Grade A (SOH >= 85%)**: Continue EV use or premium second-life stationary storage
- **Grade B (SOH 70-85%)**: Second-life stationary energy storage (solar/grid)
- **Grade C (SOH 60-70%)**: Module-level refurbishment or limited backup use
- **Grade D (SOH < 60%)**: Direct recycling and material recovery

## Safety Override Rules
1. Critical internal resistance (>150 mΩ) → Downgrade 2 levels + safety inspection
2. High internal resistance (>100 mΩ) → Downgrade 1 level
3. Thermal event (max temp >55°C) → Downgrade 2 levels + safety inspection
4. Elevated temperature (max temp >45°C) → Downgrade 1 level
5. Rapid capacity fade (>2% per 100 cycles) → Downgrade 1 level
6. Missing service history → Confidence penalty

## EU Regulation 2023/1542 Compliance
Battery passports must include identity, technical, health, materials,
lifecycle, sustainability, end-of-life, and traceability information.
Accessible via QR code, machine-readable, and interoperable.

**DISCLAIMER**: This is a prototype. Not a certified safety assessment.
"""


@mcp.resource("circulardrive://material-composition")
def material_composition() -> str:
    """Default NMC battery material composition data."""
    return json.dumps({
        "chemistry": "NMC (Nickel Manganese Cobalt)",
        "materials": {
            "lithium": {"mass_kg": 8.5, "recyclable": True, "value_per_kg_usd": 42.0},
            "nickel": {"mass_kg": 35.0, "recyclable": True, "value_per_kg_usd": 16.5},
            "cobalt": {"mass_kg": 12.0, "recyclable": True, "value_per_kg_usd": 33.0},
            "manganese": {"mass_kg": 18.0, "recyclable": True, "value_per_kg_usd": 3.5},
            "graphite": {"mass_kg": 50.0, "recyclable": True, "value_per_kg_usd": 1.2},
            "copper": {"mass_kg": 22.0, "recyclable": True, "value_per_kg_usd": 8.5},
            "aluminum": {"mass_kg": 45.0, "recyclable": True, "value_per_kg_usd": 2.3},
            "plastics": {"mass_kg": 15.0, "recyclable": False, "value_per_kg_usd": 0.3},
            "steel": {"mass_kg": 30.0, "recyclable": True, "value_per_kg_usd": 0.5},
        },
    }, indent=2)


if __name__ == "__main__":
    mcp.run()

