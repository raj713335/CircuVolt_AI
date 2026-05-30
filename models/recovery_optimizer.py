"""
Intelligent Disassembly and Recovery Optimizer
Recommends optimal disassembly sequence and material recovery paths.
"""
from typing import Optional


# Component graph representing battery pack structure
BATTERY_PACK_GRAPH = {
    "battery_pack": {
        "children": ["enclosure", "modules", "bms", "connectors", "cooling_system"],
        "material": "mixed",
        "mass_kg": 0,  # Sum of children
    },
    "enclosure": {
        "children": [],
        "material": "aluminum",
        "mass_kg": 45.0,
        "disassembly_time_min": 15,
        "tools_required": ["socket_wrench", "torx_driver"],
        "safety_risk": "low",
        "recovery_method": "shredding_and_melting",
        "value_per_kg": 2.3,
    },
    "modules": {
        "children": ["cells", "busbars", "cooling_plates", "module_housing"],
        "material": "mixed",
        "mass_kg": 0,
        "disassembly_time_min": 45,
        "tools_required": ["insulated_tools", "voltage_tester"],
        "safety_risk": "medium",
    },
    "cells": {
        "children": [],
        "material": "lithium_nmc",
        "mass_kg": 120.0,
        "disassembly_time_min": 60,
        "tools_required": ["insulated_tools", "cell_handler"],
        "safety_risk": "high",
        "recovery_method": "hydrometallurgical_recycling",
        "value_per_kg": 15.0,
    },
    "busbars": {
        "children": [],
        "material": "copper",
        "mass_kg": 22.0,
        "disassembly_time_min": 20,
        "tools_required": ["torque_wrench"],
        "safety_risk": "medium",
        "recovery_method": "direct_reuse_or_melting",
        "value_per_kg": 8.5,
    },
    "cooling_plates": {
        "children": [],
        "material": "aluminum",
        "mass_kg": 18.0,
        "disassembly_time_min": 25,
        "tools_required": ["socket_wrench"],
        "safety_risk": "low",
        "recovery_method": "melting_and_recast",
        "value_per_kg": 2.3,
    },
    "module_housing": {
        "children": [],
        "material": "aluminum_plastic",
        "mass_kg": 12.0,
        "disassembly_time_min": 10,
        "tools_required": ["basic_tools"],
        "safety_risk": "low",
        "recovery_method": "material_separation",
        "value_per_kg": 1.5,
    },
    "bms": {
        "children": [],
        "material": "electronics",
        "mass_kg": 3.0,
        "disassembly_time_min": 10,
        "tools_required": ["screwdriver", "connector_tool"],
        "safety_risk": "low",
        "recovery_method": "e_waste_recycling",
        "value_per_kg": 5.0,
    },
    "connectors": {
        "children": [],
        "material": "copper_plastic",
        "mass_kg": 5.0,
        "disassembly_time_min": 15,
        "tools_required": ["connector_tool", "wire_cutter"],
        "safety_risk": "low",
        "recovery_method": "copper_separation",
        "value_per_kg": 4.0,
    },
    "cooling_system": {
        "children": [],
        "material": "aluminum_rubber",
        "mass_kg": 8.0,
        "disassembly_time_min": 20,
        "tools_required": ["hose_clamp_tool", "wrench"],
        "safety_risk": "low",
        "recovery_method": "material_separation",
        "value_per_kg": 1.8,
    },
}

# Carbon avoidance factors (kgCO2e per kg of material recovered vs. virgin production)
CARBON_FACTORS = {
    "lithium_nmc": 12.0,
    "copper": 3.5,
    "aluminum": 8.0,
    "electronics": 15.0,
    "aluminum_plastic": 5.0,
    "copper_plastic": 3.0,
    "aluminum_rubber": 4.0,
}

# Labor cost per minute (USD)
LABOR_COST_PER_MIN = 1.5


def generate_recovery_plan(grade: str, soh: float, chemistry: str = "NMC",
                           module_count: int = 16, materials: Optional[dict] = None) -> dict:
    """
    Generate optimal disassembly sequence and recovery recommendations.
    """
    # Determine primary recovery strategy based on grade
    if grade == "A":
        strategy = "minimal_disassembly"
        primary_action = "Reuse as complete pack or with minimal refurbishment"
    elif grade == "B":
        strategy = "module_level_assessment"
        primary_action = "Test individual modules for second-life grading"
    elif grade == "C":
        strategy = "selective_recovery"
        primary_action = "Identify reusable modules, recycle degraded ones"
    else:
        strategy = "full_recycling"
        primary_action = "Complete disassembly for maximum material recovery"

    # Generate step-by-step disassembly plan
    recovery_plan = _generate_disassembly_steps(grade, strategy, module_count)

    # Calculate material recovery estimates
    material_recovery = _calculate_material_recovery(grade, materials)

    # Calculate carbon impact
    carbon_impact = _calculate_carbon_impact(material_recovery, grade, soh)

    # Calculate economic value
    economic_value = _calculate_economic_value(material_recovery, grade, soh)

    # Overall recovery score
    recovery_score = _calculate_recovery_score(
        material_recovery, carbon_impact, economic_value, grade
    )

    return {
        "strategy": strategy,
        "primary_action": primary_action,
        "recovery_plan": recovery_plan,
        "material_recovery": material_recovery,
        "carbon_impact": carbon_impact,
        "economic_value": economic_value,
        "recovery_score": recovery_score,
    }


def _generate_disassembly_steps(grade: str, strategy: str, module_count: int) -> list:
    """Generate ordered disassembly steps based on strategy."""
    steps = []

    # Safety first - always
    steps.append({
        "step": 1,
        "action": "Isolate and discharge battery pack to safe voltage",
        "reason": "Safety protocol - prevent electrical hazard",
        "duration_min": 30,
        "safety_level": "critical",
        "tools": ["voltage_tester", "discharge_equipment", "PPE"],
    })

    if strategy == "minimal_disassembly":
        steps.append({
            "step": 2,
            "action": "Perform visual inspection and diagnostic test",
            "reason": "Verify pack integrity for direct reuse",
            "duration_min": 20,
            "safety_level": "medium",
            "tools": ["diagnostic_equipment", "visual_inspection"],
        })
        steps.append({
            "step": 3,
            "action": "Update BMS firmware and recalibrate",
            "reason": "Prepare for second-life application",
            "duration_min": 15,
            "safety_level": "low",
            "tools": ["BMS_programmer"],
        })
        steps.append({
            "step": 4,
            "action": "Certify and label for second-life deployment",
            "reason": "Quality assurance for reuse market",
            "duration_min": 10,
            "safety_level": "low",
            "tools": ["label_printer", "certification_docs"],
        })

    elif strategy == "module_level_assessment":
        steps.append({
            "step": 2,
            "action": "Remove enclosure and access modules",
            "reason": "Access individual modules for testing",
            "duration_min": 15,
            "safety_level": "medium",
            "tools": ["socket_wrench", "torx_driver"],
        })
        steps.append({
            "step": 3,
            "action": f"Disconnect and test {module_count} modules individually",
            "reason": "Identify reusable modules vs. degraded ones",
            "duration_min": module_count * 5,
            "safety_level": "medium",
            "tools": ["insulated_tools", "module_tester"],
        })
        steps.append({
            "step": 4,
            "action": "Sort modules by health grade (A/B/C/D)",
            "reason": "Maximize value of each module",
            "duration_min": 10,
            "safety_level": "low",
            "tools": ["sorting_station"],
        })
        steps.append({
            "step": 5,
            "action": "Route Grade A/B modules to second-life storage",
            "reason": "Higher value than recycling",
            "duration_min": 15,
            "safety_level": "low",
            "tools": ["transport_equipment"],
        })
        steps.append({
            "step": 6,
            "action": "Send Grade C/D modules to recycling facility",
            "reason": "Material recovery from degraded modules",
            "duration_min": 10,
            "safety_level": "low",
            "tools": ["transport_equipment"],
        })

    elif strategy == "selective_recovery":
        steps.append({
            "step": 2,
            "action": "Remove enclosure for material recovery",
            "reason": "Recover aluminum enclosure material",
            "duration_min": 15,
            "safety_level": "medium",
            "tools": ["socket_wrench", "torx_driver"],
        })
        steps.append({
            "step": 3,
            "action": "Disconnect BMS and wiring harness",
            "reason": "Separate electronics for e-waste recycling",
            "duration_min": 10,
            "safety_level": "medium",
            "tools": ["insulated_tools", "connector_tool"],
        })
        steps.append({
            "step": 4,
            "action": f"Test {module_count} modules - identify any reusable ones",
            "reason": "Salvage value from partially healthy modules",
            "duration_min": module_count * 5,
            "safety_level": "medium",
            "tools": ["module_tester", "insulated_tools"],
        })
        steps.append({
            "step": 5,
            "action": "Separate copper busbars and cooling plates",
            "reason": "High-value material recovery",
            "duration_min": 25,
            "safety_level": "low",
            "tools": ["torque_wrench", "separation_tools"],
        })
        steps.append({
            "step": 6,
            "action": "Send cells to hydrometallurgical recycling",
            "reason": "Recover lithium, nickel, cobalt, manganese",
            "duration_min": 10,
            "safety_level": "medium",
            "tools": ["cell_container", "transport"],
        })

    else:  # full_recycling
        steps.append({
            "step": 2,
            "action": "Remove enclosure (aluminum recovery)",
            "reason": "Recover high-grade aluminum",
            "duration_min": 15,
            "safety_level": "medium",
            "tools": ["socket_wrench", "torx_driver"],
        })
        steps.append({
            "step": 3,
            "action": "Remove BMS, connectors, and wiring",
            "reason": "Separate electronics and copper wiring",
            "duration_min": 15,
            "safety_level": "medium",
            "tools": ["insulated_tools", "wire_cutter"],
        })
        steps.append({
            "step": 4,
            "action": "Remove cooling system components",
            "reason": "Recover aluminum and prepare for cell extraction",
            "duration_min": 20,
            "safety_level": "low",
            "tools": ["hose_clamp_tool", "wrench"],
        })
        steps.append({
            "step": 5,
            "action": "Extract all modules and separate cells",
            "reason": "Prepare cells for recycling process",
            "duration_min": 60,
            "safety_level": "high",
            "tools": ["insulated_tools", "cell_handler", "PPE"],
        })
        steps.append({
            "step": 6,
            "action": "Sort materials: copper, aluminum, plastics, cells",
            "reason": "Maximize recovery yield by material type",
            "duration_min": 30,
            "safety_level": "medium",
            "tools": ["sorting_station", "material_bins"],
        })
        steps.append({
            "step": 7,
            "action": "Send cells to hydrometallurgical processing",
            "reason": "Recover Li, Ni, Co, Mn for new battery production",
            "duration_min": 10,
            "safety_level": "low",
            "tools": ["certified_transport"],
        })

    return steps


def _calculate_material_recovery(grade: str, materials: Optional[dict] = None) -> dict:
    """Calculate estimated material recovery quantities."""
    if materials is None:
        materials = {
            "lithium": {"mass_kg": 8.5, "recovery_rate": 0.92},
            "nickel": {"mass_kg": 35.0, "recovery_rate": 0.95},
            "cobalt": {"mass_kg": 12.0, "recovery_rate": 0.95},
            "manganese": {"mass_kg": 18.0, "recovery_rate": 0.90},
            "graphite": {"mass_kg": 50.0, "recovery_rate": 0.60},
            "copper": {"mass_kg": 22.0, "recovery_rate": 0.98},
            "aluminum": {"mass_kg": 63.0, "recovery_rate": 0.95},
            "plastics": {"mass_kg": 15.0, "recovery_rate": 0.30},
            "steel": {"mass_kg": 30.0, "recovery_rate": 0.95},
        }

    # Adjust recovery rates based on grade
    grade_multiplier = {"A": 0.3, "B": 0.6, "C": 0.8, "D": 1.0}  # More recycling for lower grades
    multiplier = grade_multiplier.get(grade, 1.0)

    recovery = {}
    total_recovered = 0
    total_mass = 0

    for material, info in materials.items():
        mass = info['mass_kg']
        rate = info['recovery_rate'] * multiplier
        recovered = mass * rate
        recovery[material] = {
            "total_mass_kg": mass,
            "recovered_kg": round(recovered, 1),
            "recovery_rate_pct": round(rate * 100, 1),
        }
        total_recovered += recovered
        total_mass += mass

    recovery["summary"] = {
        "total_mass_kg": round(total_mass, 1),
        "total_recovered_kg": round(total_recovered, 1),
        "overall_recovery_pct": round((total_recovered / total_mass) * 100, 1) if total_mass > 0 else 0,
    }

    return recovery


def _calculate_carbon_impact(material_recovery: dict, grade: str, soh: float) -> dict:
    """Calculate carbon avoidance from material recovery and second-life use."""
    # Carbon avoided from material recovery (vs virgin material production)
    material_carbon_saved = 0
    material_breakdown = {}

    carbon_per_kg = {
        "lithium": 15.0,
        "nickel": 10.0,
        "cobalt": 25.0,
        "manganese": 5.0,
        "graphite": 3.0,
        "copper": 3.5,
        "aluminum": 8.0,
        "plastics": 2.0,
        "steel": 1.8,
    }

    for material, info in material_recovery.items():
        if material == "summary":
            continue
        factor = carbon_per_kg.get(material, 2.0)
        saved = info['recovered_kg'] * factor
        material_carbon_saved += saved
        material_breakdown[material] = round(saved, 1)

    # Carbon avoided from second-life use (displacing new battery/energy storage)
    second_life_carbon = 0
    if grade in ["A", "B"]:
        # Second-life battery displaces new battery manufacturing
        # Average: ~100 kgCO2e/kWh for new battery
        displaced_capacity_kwh = 75.0 * (soh / 100) * 0.7  # Usable capacity
        second_life_carbon = displaced_capacity_kwh * 50  # 50% credit vs new

    total_carbon_avoided = material_carbon_saved + second_life_carbon

    return {
        "material_recovery_kgco2e": round(material_carbon_saved, 0),
        "second_life_displacement_kgco2e": round(second_life_carbon, 0),
        "total_carbon_avoided_kgco2e": round(total_carbon_avoided, 0),
        "material_breakdown_kgco2e": material_breakdown,
        "equivalent_trees_year": round(total_carbon_avoided / 22, 0),  # ~22kg CO2/tree/year
        "equivalent_km_driving": round(total_carbon_avoided / 0.12, 0),  # ~120g CO2/km
    }


def _calculate_economic_value(material_recovery: dict, grade: str, soh: float) -> dict:
    """Calculate economic value of recovery."""
    material_prices = {
        "lithium": 42.0,
        "nickel": 16.5,
        "cobalt": 33.0,
        "manganese": 3.5,
        "graphite": 1.2,
        "copper": 8.5,
        "aluminum": 2.3,
        "plastics": 0.3,
        "steel": 0.5,
    }

    material_value = 0
    value_breakdown = {}

    for material, info in material_recovery.items():
        if material == "summary":
            continue
        price = material_prices.get(material, 1.0)
        value = info['recovered_kg'] * price
        material_value += value
        value_breakdown[material] = round(value, 2)

    # Second-life value
    second_life_value = 0
    if grade == "A":
        second_life_value = 75.0 * (soh / 100) * 80  # ~$80/kWh for premium second-life
    elif grade == "B":
        second_life_value = 75.0 * (soh / 100) * 50  # ~$50/kWh for standard second-life

    # Labor cost estimate
    total_time_min = sum([
        step.get('disassembly_time_min', 0)
        for step in BATTERY_PACK_GRAPH.values()
        if isinstance(step, dict) and 'disassembly_time_min' in step
    ])
    labor_cost = total_time_min * LABOR_COST_PER_MIN

    return {
        "material_value_usd": round(material_value, 2),
        "second_life_value_usd": round(second_life_value, 2),
        "total_gross_value_usd": round(material_value + second_life_value, 2),
        "estimated_labor_cost_usd": round(labor_cost, 2),
        "net_value_usd": round(material_value + second_life_value - labor_cost, 2),
        "value_breakdown": value_breakdown,
    }


def _calculate_recovery_score(material_recovery: dict, carbon_impact: dict,
                              economic_value: dict, grade: str) -> float:
    """Calculate overall recovery score (0-100)."""
    # Weight factors
    material_weight = 0.3
    carbon_weight = 0.3
    economic_weight = 0.25
    safety_weight = 0.15

    # Material score (based on recovery percentage)
    material_pct = material_recovery.get("summary", {}).get("overall_recovery_pct", 0)
    material_score = min(material_pct, 100)

    # Carbon score (normalized, max ~5000 kgCO2e is excellent)
    carbon_avoided = carbon_impact.get("total_carbon_avoided_kgco2e", 0)
    carbon_score = min((carbon_avoided / 5000) * 100, 100)

    # Economic score (positive net value = good)
    net_value = economic_value.get("net_value_usd", 0)
    economic_score = min(max((net_value / 5000) * 100, 0), 100)

    # Safety score (lower grade = safer to handle but less reuse value)
    safety_scores = {"A": 90, "B": 85, "C": 75, "D": 70}
    safety_score = safety_scores.get(grade, 70)

    total = (material_score * material_weight +
             carbon_score * carbon_weight +
             economic_score * economic_weight +
             safety_score * safety_weight)

    return round(total, 1)

