"""
Carbon Impact Calculator & Circularity Score Engine
Estimates carbon footprint avoidance and calculates the 100-point circularity index.
"""


def calculate_circularity_score(input_data: dict) -> dict:
    """
    Calculate overall circularity score combining 6 factors (100 pts total).
    """
    soh = input_data.get('soh', 75.0)
    grade = input_data.get('grade', 'B')
    materials_recovered_pct = input_data.get('materials_recovered_pct', 80.0)
    carbon_avoided_kg = input_data.get('carbon_avoided_kg', 0)
    second_life_potential = input_data.get('second_life_potential', True)
    recycled_content_pct = input_data.get('recycled_content_pct', 15.0)
    dfd_rating = input_data.get('dfd_rating', 5.0)
    origin = input_data.get('origin', 'Local')

    # Factor 1: Material Circularity (0-25 points)
    material_score = min(materials_recovered_pct * 0.25, 25)

    # Factor 2: Lifetime Extension (0-20 points)
    if second_life_potential and grade in ['A', 'B']:
        lifetime_score = 20 * (soh / 100)
    elif second_life_potential and grade == 'C':
        lifetime_score = 10 * (soh / 100)
    else:
        lifetime_score = 5

    # Factor 3: Carbon Avoidance (0-20 points)
    # Normalize: 1000 kgCO2e avoided = full score
    carbon_score = min((carbon_avoided_kg / 1000) * 20, 20)
    if origin == "Imported":
        carbon_score *= 0.8  # Penalty for transport emissions

    # Factor 4: Value Retention (0-15 points)
    value_retention_factors = {'A': 15, 'B': 10, 'C': 5, 'D': 2}
    value_score = value_retention_factors.get(grade, 2)

    # Factor 5: Manufacturing Sustainability (0-10 points)
    # 50% recycled content = full 10 points
    mfg_score = min((recycled_content_pct / 50.0) * 10, 10)

    # Factor 6: Design for Disassembly (0-10 points)
    # dfd_rating is 0-10
    dfd_score = min(dfd_rating, 10)

    total_score = material_score + lifetime_score + carbon_score + value_score + mfg_score + dfd_score

    return {
        "circularity_score": round(total_score, 1),
        "max_possible": 100,
        "breakdown": {
            "material_circularity": {"score": round(material_score, 1), "max": 25,
                                     "description": "Material recovery and recyclability"},
            "lifetime_extension": {"score": round(lifetime_score, 1), "max": 20,
                                   "description": "Second-life and reuse potential"},
            "carbon_avoidance": {"score": round(carbon_score, 1), "max": 20,
                                 "description": "CO₂ emissions prevented"},
            "value_retention": {"score": round(value_score, 1), "max": 15,
                                "description": "Economic value preserved"},
            "manufacturing_sustainability": {"score": round(mfg_score, 1), "max": 10,
                                "description": "Recycled content and sourcing"},
            "design_for_disassembly": {"score": round(dfd_score, 1), "max": 10,
                                "description": "Modularity and ease of separation"},
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
