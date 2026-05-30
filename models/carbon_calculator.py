"""
Carbon Impact Calculator
Estimates carbon footprint avoidance from circular economy decisions.
"""


def calculate_circularity_score(input_data: dict) -> dict:
    """
    Calculate overall circularity score combining multiple factors.
    """
    soh = input_data.get('soh', 75.0)
    grade = input_data.get('grade', 'B')
    materials_recovered_pct = input_data.get('materials_recovered_pct', 80.0)
    carbon_avoided_kg = input_data.get('carbon_avoided_kg', 0)
    second_life_potential = input_data.get('second_life_potential', True)

    # Factor 1: Material Circularity (0-30 points)
    material_score = min(materials_recovered_pct * 0.3, 30)

    # Factor 2: Lifetime Extension (0-25 points)
    if second_life_potential and grade in ['A', 'B']:
        lifetime_score = 25 * (soh / 100)
    elif second_life_potential and grade == 'C':
        lifetime_score = 15 * (soh / 100)
    else:
        lifetime_score = 5

    # Factor 3: Carbon Avoidance (0-25 points)
    # Normalize: 1000 kgCO2e avoided = full score
    carbon_score = min((carbon_avoided_kg / 1000) * 25, 25)

    # Factor 4: Value Retention (0-20 points)
    value_retention_factors = {'A': 20, 'B': 15, 'C': 10, 'D': 5}
    value_score = value_retention_factors.get(grade, 5)

    total_score = material_score + lifetime_score + carbon_score + value_score

    return {
        "circularity_score": round(total_score, 1),
        "max_possible": 100,
        "breakdown": {
            "material_circularity": {"score": round(material_score, 1), "max": 30,
                                     "description": "Material recovery and recyclability"},
            "lifetime_extension": {"score": round(lifetime_score, 1), "max": 25,
                                   "description": "Second-life and reuse potential"},
            "carbon_avoidance": {"score": round(carbon_score, 1), "max": 25,
                                 "description": "CO₂ emissions prevented"},
            "value_retention": {"score": round(value_score, 1), "max": 20,
                                "description": "Economic value preserved"},
        },
        "rating": _get_rating(total_score),
        "comparison": {
            "vs_landfill": f"{round(total_score, 0)}% better than landfill disposal",
            "vs_basic_recycling": f"{round(max(total_score - 30, 0), 0)}% better than basic shredding",
        }
    }


def _get_rating(score: float) -> str:
    if score >= 80:
        return "Excellent Circularity"
    elif score >= 60:
        return "Good Circularity"
    elif score >= 40:
        return "Moderate Circularity"
    else:
        return "Low Circularity - Improvement Needed"

