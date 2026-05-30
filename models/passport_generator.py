"""
AI Material Passport Generator
Creates EU Battery Regulation compliant material passport records.
"""
import json
import uuid
from datetime import datetime
from typing import Optional


DEFAULT_MATERIALS = {
    "lithium": {"mass_kg": 8.5, "recyclable": True, "value_per_kg": 42.0},
    "nickel": {"mass_kg": 35.0, "recyclable": True, "value_per_kg": 16.5},
    "cobalt": {"mass_kg": 12.0, "recyclable": True, "value_per_kg": 33.0},
    "manganese": {"mass_kg": 18.0, "recyclable": True, "value_per_kg": 3.5},
    "graphite": {"mass_kg": 50.0, "recyclable": True, "value_per_kg": 1.2},
    "copper": {"mass_kg": 22.0, "recyclable": True, "value_per_kg": 8.5},
    "aluminum": {"mass_kg": 45.0, "recyclable": True, "value_per_kg": 2.3},
    "plastics": {"mass_kg": 15.0, "recyclable": False, "value_per_kg": 0.3},
    "steel": {"mass_kg": 30.0, "recyclable": True, "value_per_kg": 0.5},
    "electrolyte": {"mass_kg": 12.0, "recyclable": False, "value_per_kg": 0.0},
}


def generate_passport(passport_input: dict, soh_data: Optional[dict] = None,
                      grade_data: Optional[dict] = None) -> dict:
    """
    Generate a comprehensive material passport for a battery component.

    Follows EU Regulation 2023/1542 battery passport structure.
    """
    component_id = passport_input.get('component_id', f"BAT-IND-{datetime.now().year}-{uuid.uuid4().hex[:6].upper()}")
    materials = passport_input.get('materials') or DEFAULT_MATERIALS

    # Calculate totals
    total_mass = sum(m['mass_kg'] for m in materials.values())
    recyclable_mass = sum(m['mass_kg'] for m in materials.values() if m['recyclable'])
    total_value = sum(m['mass_kg'] * m['value_per_kg'] for m in materials.values())

    # Embodied carbon estimate (simplified: ~100 kgCO2e per kWh for NMC batteries)
    rated_capacity = passport_input.get('rated_capacity_kwh', 75.0)
    embodied_carbon = rated_capacity * 100  # kgCO2e

    # Recycled content estimate
    recycled_content_pct = 12.0  # Industry average for new batteries

    # Recovery score
    recovery_score = (recyclable_mass / total_mass) * 100 if total_mass > 0 else 0

    # Completeness score
    completeness = _calculate_completeness(passport_input, soh_data, grade_data)

    passport = {
        "identity": {
            "component_id": component_id,
            "battery_id": passport_input.get('battery_id', f"BAT-{uuid.uuid4().hex[:8].upper()}"),
            "vehicle_id": passport_input.get('vehicle_id', "N/A"),
            "manufacturer": passport_input.get('manufacturer', 'Unknown'),
            "model": passport_input.get('model', 'Unknown'),
            "unique_identifier": str(uuid.uuid4()),
            "passport_version": "1.0",
            "created_date": datetime.now().isoformat(),
        },
        "technical": {
            "chemistry": passport_input.get('chemistry', 'NMC'),
            "rated_capacity_kwh": rated_capacity,
            "nominal_voltage": passport_input.get('voltage', 400.0),
            "module_count": passport_input.get('module_count', 16),
            "cell_count": passport_input.get('cell_count', 192),
            "total_mass_kg": round(total_mass, 1),
        },
        "health": {
            "predicted_soh": soh_data.get('predicted_soh', None) if soh_data else None,
            "rul_cycles": soh_data.get('rul_cycles', None) if soh_data else None,
            "confidence_score": soh_data.get('confidence', None) if soh_data else None,
            "grade": grade_data.get('grade', None) if grade_data else None,
            "recommendation": grade_data.get('recommendation', None) if grade_data else None,
            "risk_flags": grade_data.get('risk_flags', []) if grade_data else [],
            "assessment_date": datetime.now().isoformat(),
        },
        "materials": {
            name: {
                "mass_kg": info['mass_kg'],
                "recyclable": info['recyclable'],
                "estimated_value_usd": round(info['mass_kg'] * info['value_per_kg'], 2),
            }
            for name, info in materials.items()
        },
        "lifecycle": {
            "manufacturing_date": passport_input.get('manufacturing_date', '2020-01-01'),
            "service_history": passport_input.get('service_history', []),
            "ownership_transfers": 1,
            "total_energy_delivered_mwh": round(rated_capacity * 0.8 * 500 / 1000, 1),  # Estimate
        },
        "sustainability": {
            "embodied_carbon_kgco2e": round(embodied_carbon, 0),
            "recycled_content_pct": recycled_content_pct,
            "recovery_score_pct": round(recovery_score, 1),
            "total_material_value_usd": round(total_value, 2),
            "recyclable_mass_kg": round(recyclable_mass, 1),
            "recyclable_mass_pct": round((recyclable_mass / total_mass) * 100, 1) if total_mass > 0 else 0,
        },
        "end_of_life": {
            "recommendation": grade_data.get('recommendation', 'Assessment pending') if grade_data else 'Assessment pending',
            "disassembly_complexity": "Medium",
            "safety_warnings": grade_data.get('risk_flags', []) if grade_data else [],
            "estimated_recovery_value_usd": round(total_value * 0.7, 2),  # 70% recovery efficiency
        },
        "traceability": {
            "qr_code_url": f"/passport/{component_id}/qr",
            "digital_record_url": f"/passport/{component_id}",
            "last_updated": datetime.now().isoformat(),
            "update_log": [
                {"date": datetime.now().isoformat(), "action": "Passport created", "source": "CircularDrive AI"}
            ],
            "regulation_compliance": "EU Regulation 2023/1542 (prototype)"
        },
        "completeness_score": completeness,
    }

    return passport


def _calculate_completeness(passport_input: dict, soh_data: Optional[dict],
                            grade_data: Optional[dict]) -> float:
    """Calculate passport data completeness percentage."""
    total_fields = 10
    filled = 0

    if passport_input.get('component_id'):
        filled += 1
    if passport_input.get('manufacturer') and passport_input['manufacturer'] != 'Unknown':
        filled += 1
    if passport_input.get('chemistry'):
        filled += 1
    if passport_input.get('rated_capacity_kwh'):
        filled += 1
    if passport_input.get('manufacturing_date'):
        filled += 1
    if passport_input.get('materials'):
        filled += 1
    else:
        filled += 0.5  # Using defaults
    if soh_data:
        filled += 2  # SOH + RUL
    if grade_data:
        filled += 1.5  # Grade + recommendation
    if passport_input.get('service_history'):
        filled += 0.5

    return round(min((filled / total_fields) * 100, 100), 1)

