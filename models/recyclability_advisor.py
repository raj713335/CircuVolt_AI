"""
Design for Recyclability Advisor
Generates suggestions to improve component recyclability.
"""
from typing import List


def calculate_recyclability_score(design_input: dict) -> dict:
    """
    Calculate recyclability score (0-100) and generate improvement suggestions.
    """
    score = 100.0
    suggestions = []
    penalties = {}

    # Fastener assessment
    fastener_count = design_input.get('fastener_count', 42)
    if fastener_count > 60:
        penalty = 15
        score -= penalty
        penalties['fasteners'] = penalty
        suggestions.append({
            "category": "Fasteners",
            "issue": f"High fastener count ({fastener_count})",
            "suggestion": "Reduce fastener count by using snap-fit designs and standardized clips",
            "impact": "Reduces disassembly time by ~30%",
            "priority": "high",
            "effort": "medium",
        })
    elif fastener_count > 40:
        penalty = 8
        score -= penalty
        penalties['fasteners'] = penalty
        suggestions.append({
            "category": "Fasteners",
            "issue": f"Moderate fastener count ({fastener_count})",
            "suggestion": "Standardize to 2-3 screw types maximum for faster disassembly",
            "impact": "Reduces tool changes and disassembly time by ~15%",
            "priority": "medium",
            "effort": "low",
        })

    # Adhesive assessment
    adhesive_use = design_input.get('adhesive_use', 'High').lower()
    if adhesive_use == 'high':
        penalty = 20
        score -= penalty
        penalties['adhesives'] = penalty
        suggestions.append({
            "category": "Adhesives",
            "issue": "High adhesive use prevents non-destructive disassembly",
            "suggestion": "Replace permanent adhesives with thermally-debondable or mechanically-removable fasteners",
            "impact": "Enables non-destructive module separation, increasing reuse potential by 40%",
            "priority": "high",
            "effort": "high",
        })
    elif adhesive_use == 'medium':
        penalty = 10
        score -= penalty
        penalties['adhesives'] = penalty
        suggestions.append({
            "category": "Adhesives",
            "issue": "Moderate adhesive use limits disassembly options",
            "suggestion": "Use debondable adhesives (heat-activated release) for critical joints",
            "impact": "Easier module separation for second-life applications",
            "priority": "medium",
            "effort": "medium",
        })

    # Material mix assessment
    material_mix = design_input.get('material_mix', [])
    if len(material_mix) > 4:
        penalty = 12
        score -= penalty
        penalties['material_mix'] = penalty
        suggestions.append({
            "category": "Material Complexity",
            "issue": f"High material diversity ({len(material_mix)} types: {', '.join(material_mix)})",
            "suggestion": "Reduce material types or ensure easy separation points between different materials",
            "impact": "Improves sorting efficiency and material purity in recycling",
            "priority": "medium",
            "effort": "high",
        })
    elif len(material_mix) > 3:
        penalty = 6
        score -= penalty
        penalties['material_mix'] = penalty
        suggestions.append({
            "category": "Material Complexity",
            "issue": f"Moderate material diversity ({len(material_mix)} types)",
            "suggestion": "Add clear separation interfaces between material zones",
            "impact": "Better material sorting during end-of-life processing",
            "priority": "low",
            "effort": "medium",
        })

    # Labeling assessment
    labeling = design_input.get('labeling_quality', 'Poor').lower()
    if labeling == 'poor':
        penalty = 15
        score -= penalty
        penalties['labeling'] = penalty
        suggestions.append({
            "category": "Material Labeling",
            "issue": "Poor material identification labels on components",
            "suggestion": "Add QR/RFID material labels on each module and major component with ISO material codes",
            "impact": "Enables automated sorting and 95%+ correct material identification",
            "priority": "high",
            "effort": "low",
        })
    elif labeling == 'fair':
        penalty = 7
        score -= penalty
        penalties['labeling'] = penalty
        suggestions.append({
            "category": "Material Labeling",
            "issue": "Incomplete material labeling",
            "suggestion": "Extend labels to all sub-components using embedded RFID tags",
            "impact": "Supports automated recycling facility sorting",
            "priority": "medium",
            "effort": "low",
        })

    # Modularity assessment
    modularity = design_input.get('modularity', 'Low').lower()
    if modularity == 'low':
        penalty = 18
        score -= penalty
        penalties['modularity'] = penalty
        suggestions.append({
            "category": "Modularity",
            "issue": "Low modularity prevents selective component replacement",
            "suggestion": "Redesign with standardized module interfaces and quick-disconnect fittings",
            "impact": "Enables individual module replacement/reuse, extending pack lifetime by 50%",
            "priority": "high",
            "effort": "high",
        })
    elif modularity == 'medium':
        penalty = 8
        score -= penalty
        penalties['modularity'] = penalty
        suggestions.append({
            "category": "Modularity",
            "issue": "Moderate modularity - some components not independently serviceable",
            "suggestion": "Add quick-release connectors for cooling and electrical interfaces",
            "impact": "Faster module-level service and second-life preparation",
            "priority": "medium",
            "effort": "medium",
        })

    # Hazard separation assessment
    hazard_sep = design_input.get('hazard_separation', 'Difficult').lower()
    if hazard_sep == 'difficult':
        penalty = 15
        score -= penalty
        penalties['hazard_separation'] = penalty
        suggestions.append({
            "category": "Hazard Separation",
            "issue": "Difficult to isolate hazardous materials during disassembly",
            "suggestion": "Design dedicated hazard zones with clear visual indicators and physical barriers",
            "impact": "Reduces worker safety risk and enables faster safe disassembly",
            "priority": "high",
            "effort": "medium",
        })
    elif hazard_sep == 'moderate':
        penalty = 7
        score -= penalty
        penalties['hazard_separation'] = penalty
        suggestions.append({
            "category": "Hazard Separation",
            "issue": "Moderate difficulty in hazard isolation",
            "suggestion": "Add color-coded safety zones and single-step isolation mechanisms",
            "impact": "Clearer disassembly procedures for recycling facilities",
            "priority": "medium",
            "effort": "low",
        })

    # Ensure score doesn't go below 0
    score = max(score, 5.0)

    # Priority actions (top 3 by priority)
    high_priority = [s for s in suggestions if s['priority'] == 'high']
    medium_priority = [s for s in suggestions if s['priority'] == 'medium']
    priority_actions = [s['suggestion'] for s in (high_priority + medium_priority)[:3]]

    return {
        "recyclability_score": round(score, 1),
        "suggestions": suggestions,
        "priority_actions": priority_actions,
        "score_penalties": penalties,
        "grade": _score_to_grade(score),
    }


def _score_to_grade(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Poor"

