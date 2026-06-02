"""
Intelligent Disassembly and Recovery Optimizer
Recommends optimal disassembly sequence and material recovery paths for various EV components.
"""
from typing import Optional

# Component graph representing battery pack structure
BATTERY_PACK_GRAPH = {
    "battery_pack": {"children": ["enclosure", "modules", "bms", "connectors", "cooling_system"], "material": "mixed", "mass_kg": 0},
    "enclosure": {"children": [], "material": "aluminum", "mass_kg": 45.0, "disassembly_time_min": 15, "tools_required": ["socket_wrench", "torx_driver"], "safety_risk": "low"},
    "modules": {"children": ["cells", "busbars", "cooling_plates", "module_housing"], "material": "mixed", "mass_kg": 0, "disassembly_time_min": 45, "tools_required": ["insulated_tools", "voltage_tester"], "safety_risk": "medium"},
    "cells": {"children": [], "material": "lithium_nmc", "mass_kg": 120.0, "disassembly_time_min": 60, "tools_required": ["insulated_tools", "cell_handler"], "safety_risk": "high"},
    "busbars": {"children": [], "material": "copper", "mass_kg": 22.0, "disassembly_time_min": 20, "tools_required": ["torque_wrench"], "safety_risk": "medium"},
    "cooling_plates": {"children": [], "material": "aluminum", "mass_kg": 18.0, "disassembly_time_min": 25, "tools_required": ["socket_wrench"], "safety_risk": "low"},
    "module_housing": {"children": [], "material": "aluminum_plastic", "mass_kg": 12.0, "disassembly_time_min": 10, "tools_required": ["basic_tools"], "safety_risk": "low"},
    "bms": {"children": [], "material": "electronics", "mass_kg": 3.0, "disassembly_time_min": 10, "tools_required": ["screwdriver", "connector_tool"], "safety_risk": "low"},
    "connectors": {"children": [], "material": "copper_plastic", "mass_kg": 5.0, "disassembly_time_min": 15, "tools_required": ["connector_tool", "wire_cutter"], "safety_risk": "low"},
    "cooling_system": {"children": [], "material": "aluminum_rubber", "mass_kg": 8.0, "disassembly_time_min": 20, "tools_required": ["hose_clamp_tool", "wrench"], "safety_risk": "low"},
}

ELECTRIC_MOTOR_GRAPH = {
    "electric_motor": {"children": ["housing", "stator", "rotor", "bearings"], "material": "mixed", "mass_kg": 0},
    "housing": {"children": [], "material": "aluminum", "mass_kg": 20.0, "disassembly_time_min": 20, "safety_risk": "low"},
    "stator": {"children": ["windings", "stator_core"], "material": "mixed", "mass_kg": 0, "disassembly_time_min": 35, "safety_risk": "medium"},
    "rotor": {"children": ["magnets", "rotor_core"], "material": "mixed", "mass_kg": 0, "disassembly_time_min": 25, "safety_risk": "high"},
    "windings": {"children": [], "material": "copper", "mass_kg": 15.0, "disassembly_time_min": 30, "safety_risk": "low"},
    "magnets": {"children": [], "material": "neodymium", "mass_kg": 2.5, "disassembly_time_min": 40, "safety_risk": "high"},
    "stator_core": {"children": [], "material": "steel", "mass_kg": 18.0, "disassembly_time_min": 10, "safety_risk": "low"},
    "rotor_core": {"children": [], "material": "steel", "mass_kg": 12.0, "disassembly_time_min": 10, "safety_risk": "low"},
    "bearings": {"children": [], "material": "steel", "mass_kg": 1.5, "disassembly_time_min": 5, "safety_risk": "low"},
}

POWER_ELECTRONICS_GRAPH = {
    "inverter": {"children": ["housing", "pcb_boards", "cooling_plates", "busbars", "capacitors"], "material": "mixed", "mass_kg": 0},
    "housing": {"children": [], "material": "aluminum", "mass_kg": 8.0, "disassembly_time_min": 15, "safety_risk": "low"},
    "pcb_boards": {"children": ["semiconductors", "precious_metals"], "material": "mixed", "mass_kg": 3.0, "disassembly_time_min": 25, "safety_risk": "medium"},
    "cooling_plates": {"children": [], "material": "aluminum", "mass_kg": 4.0, "disassembly_time_min": 10, "safety_risk": "low"},
    "busbars": {"children": [], "material": "copper", "mass_kg": 2.5, "disassembly_time_min": 10, "safety_risk": "low"},
    "capacitors": {"children": [], "material": "electronics", "mass_kg": 1.5, "disassembly_time_min": 15, "safety_risk": "high"},
    "semiconductors": {"children": [], "material": "silicon_carbide", "mass_kg": 0.5, "disassembly_time_min": 20, "safety_risk": "medium"},
    "precious_metals": {"children": [], "material": "gold_silver", "mass_kg": 0.1, "disassembly_time_min": 30, "safety_risk": "high"},
}

# Labor cost per minute (USD)
LABOR_COST_PER_MIN = 1.5

def generate_recovery_plan(grade: str, soh: float, chemistry: str = "NMC",
                           module_count: int = 16, component_type: str = "EV Battery Pack",
                           motor_type: str = "PMSM", semiconductor_type: str = "Silicon Carbide (SiC)",
                           materials: Optional[dict] = None) -> dict:
    """Generate optimal disassembly sequence and recovery recommendations."""
    
    if component_type == "Electric Drive Motor":
        strategy = "shredding_and_magnetic_separation" if grade in ["C", "D"] else "non_destructive_disassembly"
        primary_action = "Recover high-value Rare Earth magnets and Copper windings" if grade in ["C", "D"] else "Refurbish rotor and rewind stator"
        recovery_plan = _generate_motor_steps(grade, motor_type)
        material_recovery = _calculate_motor_materials(grade, motor_type, materials)
    elif component_type == "Power Electronics (Inverter)":
        strategy = "chemical_leaching_and_smelting" if grade in ["C", "D"] else "component_level_refurbishment"
        primary_action = "Recover precious metals and SiC dies" if grade in ["C", "D"] else "Replace capacitors and refurbish PCB"
        recovery_plan = _generate_inverter_steps(grade, semiconductor_type)
        material_recovery = _calculate_inverter_materials(grade, semiconductor_type, materials)
    else: # Battery Pack
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
        recovery_plan = _generate_battery_steps(grade, strategy, module_count)
        material_recovery = _calculate_battery_materials(grade, chemistry, materials)

    carbon_impact = _calculate_carbon_impact(material_recovery, grade, soh, component_type)
    economic_value = _calculate_economic_value(material_recovery, grade, soh, component_type)
    recovery_score = _calculate_recovery_score(material_recovery, carbon_impact, economic_value, grade)

    return {
        "strategy": strategy,
        "primary_action": primary_action,
        "recovery_plan": recovery_plan,
        "material_recovery": material_recovery,
        "carbon_impact": carbon_impact,
        "economic_value": economic_value,
        "recovery_score": recovery_score,
    }

def _generate_battery_steps(grade: str, strategy: str, module_count: int) -> list:
    steps = [{
        "step": 1, "action": "Isolate and discharge battery pack to safe voltage",
        "reason": "Safety protocol - prevent electrical hazard", "duration_min": 30, "safety_level": "critical"
    }]
    if strategy == "minimal_disassembly":
        steps += [
            {"step": 2, "action": "Perform visual inspection and diagnostic test", "reason": "Verify pack integrity", "duration_min": 20, "safety_level": "medium"},
            {"step": 3, "action": "Update BMS firmware and recalibrate", "reason": "Prepare for second-life", "duration_min": 15, "safety_level": "low"},
        ]
    elif strategy == "module_level_assessment":
        steps += [
            {"step": 2, "action": "Remove enclosure and access modules", "reason": "Access for testing", "duration_min": 15, "safety_level": "medium"},
            {"step": 3, "action": f"Disconnect and test {module_count} modules individually", "reason": "Grade health", "duration_min": module_count * 5, "safety_level": "medium"},
            {"step": 4, "action": "Sort modules by health grade", "reason": "Maximize value", "duration_min": 10, "safety_level": "low"},
        ]
    else:
        steps += [
            {"step": 2, "action": "Remove enclosure (aluminum recovery)", "reason": "Recover high-grade aluminum", "duration_min": 15, "safety_level": "medium"},
            {"step": 3, "action": "Remove BMS, connectors, and wiring", "reason": "Separate electronics", "duration_min": 15, "safety_level": "medium"},
            {"step": 4, "action": "Extract all modules and separate cells", "reason": "Prepare cells for recycling", "duration_min": 60, "safety_level": "high"},
            {"step": 5, "action": "Send cells to hydrometallurgical processing", "reason": "Recover Li, Ni, Co, Mn", "duration_min": 10, "safety_level": "low"},
        ]
    return steps

def _generate_motor_steps(grade: str, motor_type: str) -> list:
    steps = [
        {"step": 1, "action": "Drain cooling fluids and secure shaft", "reason": "Safety protocol", "duration_min": 15, "safety_level": "medium"},
        {"step": 2, "action": "Unbolt and remove aluminum outer housing", "reason": "Recover structural aluminum", "duration_min": 25, "safety_level": "medium"},
        {"step": 3, "action": "Extract stator and rotor core assembly", "reason": "Separate core magnetic components", "duration_min": 40, "safety_level": "high"},
    ]
    if motor_type == "PMSM":
        steps.append({"step": 4, "action": "Thermal demagnetization of rotor", "reason": "Safely extract Rare Earth magnets (NdFeB)", "duration_min": 45, "safety_level": "high"})
    steps.append({"step": 5, "action": "Shred and magnetically separate copper windings from steel core", "reason": "Recover high-purity copper", "duration_min": 30, "safety_level": "medium"})
    return steps

def _generate_inverter_steps(grade: str, semiconductor_type: str) -> list:
    return [
        {"step": 1, "action": "Discharge high-voltage capacitors", "reason": "Critical safety protocol", "duration_min": 20, "safety_level": "critical"},
        {"step": 2, "action": "Remove aluminum cold plates and thermal paste", "reason": "Recover heat sinks", "duration_min": 15, "safety_level": "low"},
        {"step": 3, "action": "Extract internal copper busbars", "reason": "High purity copper recovery", "duration_min": 10, "safety_level": "medium"},
        {"step": 4, "action": "De-solder main PCB boards", "reason": "Separate microelectronics", "duration_min": 25, "safety_level": "medium"},
        {"step": 5, "action": f"Chemical leaching of {semiconductor_type} dies and precious metals", "reason": "Recover Gold, Silver, and Semiconductor material", "duration_min": 60, "safety_level": "high"},
    ]

def _calculate_battery_materials(grade: str, chemistry: str, materials: Optional[dict]) -> dict:
    if chemistry.upper() == "LFP":
        m = {"lithium": 6.5, "iron": 32.0, "phosphate": 45.0, "graphite": 48.0, "copper": 20.0, "aluminum": 60.0}
    elif chemistry.upper() == "NCA":
        m = {"lithium": 8.0, "nickel": 40.0, "cobalt": 8.0, "aluminum_active": 4.0, "graphite": 50.0, "copper": 22.0, "aluminum": 63.0}
    else:
        m = {"lithium": 8.5, "nickel": 35.0, "cobalt": 12.0, "manganese": 18.0, "graphite": 50.0, "copper": 22.0, "aluminum": 63.0}
    return _build_material_dict(m, grade)

def _calculate_motor_materials(grade: str, motor_type: str, materials: Optional[dict]) -> dict:
    m = {"copper": 15.0, "aluminum": 20.0, "steel": 30.0, "plastics": 2.0}
    if motor_type == "PMSM":
        m["neodymium"] = 2.5
        m["dysprosium"] = 0.5
    return _build_material_dict(m, grade)

def _calculate_inverter_materials(grade: str, semiconductor_type: str, materials: Optional[dict]) -> dict:
    m = {"aluminum": 12.0, "copper": 3.5, "gold": 0.05, "silver": 0.1, "plastics": 2.0}
    if "SiC" in semiconductor_type:
        m["silicon_carbide"] = 0.5
    else:
        m["silicon"] = 0.8
    return _build_material_dict(m, grade)

def _build_material_dict(raw_materials: dict, grade: str) -> dict:
    grade_multiplier = {"A": 0.3, "B": 0.6, "C": 0.8, "D": 1.0}.get(grade, 1.0)
    recovery = {}
    total_recovered, total_mass = 0, 0
    for mat, mass in raw_materials.items():
        rate = 0.95 * grade_multiplier
        recovered = mass * rate
        recovery[mat] = {"total_mass_kg": mass, "recovered_kg": round(recovered, 2), "recovery_rate_pct": round(rate*100, 1)}
        total_mass += mass
        total_recovered += recovered
    recovery["summary"] = {
        "total_mass_kg": round(total_mass, 1),
        "total_recovered_kg": round(total_recovered, 1),
        "overall_recovery_pct": round((total_recovered/total_mass)*100, 1) if total_mass > 0 else 0
    }
    return recovery

def _calculate_carbon_impact(material_recovery: dict, grade: str, soh: float, component_type: str) -> dict:
    carbon_per_kg = {
        "lithium": 15.0, "nickel": 10.0, "cobalt": 25.0, "manganese": 5.0, "iron": 1.5, "phosphate": 2.5,
        "neodymium": 35.0, "dysprosium": 40.0, "silicon_carbide": 20.0, "gold": 12500.0, "silver": 150.0,
        "graphite": 3.0, "copper": 3.5, "aluminum": 8.0, "steel": 1.8, "plastics": 2.0, "silicon": 15.0
    }
    saved = sum(info['recovered_kg'] * carbon_per_kg.get(m, 2.0) for m, info in material_recovery.items() if m != "summary")
    
    sl_carbon = 0
    if grade in ["A", "B"]:
        if component_type == "EV Battery Pack":
            sl_carbon = 75.0 * (soh/100) * 0.7 * 50
        else:
            sl_carbon = 1500  # Generic component reuse carbon credit
            
    total = saved + sl_carbon
    return {
        "total_carbon_avoided_kgco2e": round(total, 0),
        "equivalent_trees_year": round(total / 22, 0),
        "equivalent_km_driving": round(total / 0.12, 0),
    }

def _calculate_economic_value(material_recovery: dict, grade: str, soh: float, component_type: str) -> dict:
    prices = {
        "lithium": 42.0, "nickel": 16.5, "cobalt": 33.0, "manganese": 3.5, "iron": 0.3, "phosphate": 0.6,
        "neodymium": 70.0, "dysprosium": 250.0, "silicon_carbide": 45.0, "gold": 65000.0, "silver": 800.0,
        "graphite": 1.2, "copper": 8.5, "aluminum": 2.3, "steel": 0.5, "plastics": 0.3, "silicon": 5.0
    }
    mat_val = sum(info['recovered_kg'] * prices.get(m, 1.0) for m, info in material_recovery.items() if m != "summary")
    
    sl_val = 0
    if grade in ["A", "B"]:
        if component_type == "EV Battery Pack":
            sl_val = 75.0 * (soh/100) * (80 if grade == "A" else 50)
        elif component_type == "Electric Drive Motor":
            sl_val = 1200 * (soh/100)
        else:
            sl_val = 800 * (soh/100)
            
    labor = 60 * 1.5  # Approx 60 mins avg
    return {
        "net_value_usd": round(mat_val + sl_val - labor, 2),
    }

def _calculate_recovery_score(material_recovery: dict, carbon_impact: dict, economic_value: dict, grade: str) -> float:
    mat_pct = material_recovery.get("summary", {}).get("overall_recovery_pct", 0)
    co2 = min((carbon_impact.get("total_carbon_avoided_kgco2e", 0) / 5000) * 100, 100)
    econ = min(max((economic_value.get("net_value_usd", 0) / 5000) * 100, 0), 100)
    safe = {"A": 90, "B": 85, "C": 75, "D": 70}.get(grade, 70)
    return round(mat_pct*0.3 + co2*0.3 + econ*0.25 + safe*0.15, 1)
