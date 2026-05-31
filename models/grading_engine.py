"""
Second-Life Grading Engine
Assigns A/B/C/D grade based on SOH prediction and safety rules.
"""
from typing import List, Tuple


def calculate_grade(soh: float, internal_resistance: float = 50.0,
                    max_temperature: float = 35.0, cycle_count: float = 500,
                    has_service_history: bool = True) -> dict:
    """
    Assign second-life grade based on SOH and safety factors.

    Grade Logic:
        A (≥85%): Continue EV use / premium second-life
        B (70-85%): Stationary energy storage
        C (60-70%): Module-level refurbishment or limited use
        D (<60%): Direct recycling / material recovery
    """
    # Base grade from SOH
    if soh >= 85:
        grade = "A"
        recommendation = "Continue EV use or premium second-life stationary storage"
    elif soh >= 70:
        grade = "B"
        recommendation = "Second-life stationary energy storage (solar/grid)"
    elif soh >= 60:
        grade = "C"
        recommendation = "Module-level refurbishment or limited backup use"
    else:
        grade = "D"
        recommendation = "Direct recycling and material recovery"

    # Risk assessment and safety overrides
    risk_flags = []
    confidence_penalty = 0

    # High internal resistance check
    if internal_resistance > 150:
        risk_flags.append("critical_internal_resistance")
        grade = _downgrade(grade, 2)
        recommendation = "Safety inspection required before any reuse"
    elif internal_resistance > 100:
        risk_flags.append("high_internal_resistance_growth")
        grade = _downgrade(grade, 1)

    # Temperature history check
    if max_temperature > 55:
        risk_flags.append("thermal_event_detected")
        grade = _downgrade(grade, 2)
        recommendation = "Safety inspection required - thermal event history"
    elif max_temperature > 45:
        risk_flags.append("elevated_temperature_history")
        grade = _downgrade(grade, 1)

    # Rapid capacity fade check (high cycles with moderate SOH loss)
    if cycle_count > 0:
        fade_rate = (100 - soh) / cycle_count
        if fade_rate > 0.02:
            risk_flags.append("rapid_capacity_fade")
            if grade in ["A", "B"]:
                grade = _downgrade(grade, 1)
                risk_flags.append("second_life_caution")

    # Missing service history
    if not has_service_history:
        risk_flags.append("missing_service_history")
        confidence_penalty += 1

    # Confidence level
    if confidence_penalty == 0 and len(risk_flags) == 0:
        confidence = "High"
    elif confidence_penalty <= 1 and len(risk_flags) <= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Updated recommendation based on final grade
    grade_recommendations = {
        "A": "Continue EV use or premium second-life stationary storage",
        "B": "Second-life stationary energy storage (solar/grid)",
        "C": "Module-level refurbishment or limited backup use",
        "D": "Direct recycling and material recovery"
    }

    if "Safety inspection" not in recommendation:
        recommendation = grade_recommendations.get(grade, recommendation)

    return {
        "grade": grade,
        "recommendation": recommendation,
        "risk_flags": risk_flags,
        "confidence": confidence,
        "soh_range": _get_soh_range(grade),
        "grade_explanation": _get_grade_explanation(grade, risk_flags, soh)
    }


def _downgrade(grade: str, levels: int) -> str:
    """Downgrade a grade by specified levels."""
    grades = ["A", "B", "C", "D"]
    current_idx = grades.index(grade)
    new_idx = min(current_idx + levels, 3)
    return grades[new_idx]


def _get_soh_range(grade: str) -> str:
    ranges = {
        "A": "≥ 85%",
        "B": "70-85%",
        "C": "60-70%",
        "D": "< 60%"
    }
    return ranges.get(grade, "Unknown")


def _get_grade_explanation(grade: str, risk_flags: List[str], soh: float) -> str:
    """Generate human-readable explanation for the grade."""
    explanations = []

    if grade == "A":
        explanations.append(f"Battery SOH of {soh:.1f}% indicates excellent health suitable for continued EV use or premium second-life applications.")
    elif grade == "B":
        explanations.append(f"Battery SOH of {soh:.1f}% indicates good health suitable for stationary energy storage applications.")
    elif grade == "C":
        explanations.append(f"Battery SOH of {soh:.1f}% indicates moderate degradation. Module-level assessment recommended.")
    else:
        explanations.append(f"Battery SOH of {soh:.1f}% indicates significant degradation. Recycling recommended for material recovery.")

    if "critical_internal_resistance" in risk_flags:
        explanations.append("Critical internal resistance detected - safety inspection mandatory.")
    if "thermal_event_detected" in risk_flags:
        explanations.append("Thermal event history detected - requires safety assessment.")
    if "rapid_capacity_fade" in risk_flags:
        explanations.append("Rapid capacity fade observed - accelerated aging may continue.")
    if "missing_service_history" in risk_flags:
        explanations.append("Missing service history reduces confidence in assessment.")

    return " ".join(explanations)

