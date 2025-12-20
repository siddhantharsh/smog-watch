"""
AirAware Health & Policy Advisory Engine
-----------------------------------------
Features:
1. Group-specific actionable health protocols:
   - Children & Schools (sports, morning assembly, school timing, pediatric masks)
   - Asthma & COPD Patients (inhalers, peak flow, morning walk warnings, ER red flags)
   - Outdoor Workers (traffic police, delivery agents, construction, shift limits, N95/FFP2)
   - General Public & Commuters (ventilation windows, car AC recirculation, indoor care)
2. Graded Response Action Plan (GRAP Stages I-IV) Policy Engine for Delhi-NCR & North India.
3. Personalized Air Vulnerability Index (AVI) Assessment Tool.
4. Exportable Daily Air Quality & Health Bulletin generator.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


# -----------------------------------------------------------------------------
# 1. CPCB AQI CATEGORIES & CLINICAL HEALTH DEFINITIONS
# -----------------------------------------------------------------------------
AQI_CLINICAL_DEFINITIONS = {
    "Good": {
        "range": "0 - 50",
        "color": "#00B050",
        "badge": "🟢 Good",
        "general_health": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "hospital_impact": "Negligible increase in emergency respiratory presentations."
    },
    "Satisfactory": {
        "range": "51 - 100",
        "color": "#92D050",
        "badge": "🟡 Satisfactory",
        "general_health": "Minor breathing discomfort may be experienced by hyper-sensitive individuals.",
        "hospital_impact": "Occasional mild allergy or throat irritation visits."
    },
    "Moderate": {
        "range": "101 - 200",
        "color": "#FFC000",
        "badge": "🟠 Moderate",
        "general_health": "Breathing discomfort to people with asthma, lungs, and heart diseases, and children/elderly.",
        "hospital_impact": "5-10% increase in outpatient bronchodilator refill requests."
    },
    "Poor": {
        "range": "201 - 300",
        "color": "#FF7C80",
        "badge": "🔴 Poor",
        "general_health": "Breathing discomfort to most people on prolonged exposure. Triggers acute asthma attacks.",
        "hospital_impact": "15-25% surge in respiratory OPD cases; increased nebulization needs."
    },
    "Very Poor": {
        "range": "301 - 400",
        "color": "#FF0000",
        "badge": "🚨 Very Poor",
        "general_health": "Respiratory illness on prolonged exposure. Significant cardiovascular stress in vulnerable adults.",
        "hospital_impact": "30-50% increase in emergency room acute bronchospasm and COPD exacerbations."
    },
    "Severe": {
        "range": "401 - 500",
        "color": "#C00000",
        "badge": "☣️ Severe",
        "general_health": "Affects healthy individuals and seriously impacts those with existing respiratory or cardiac diseases.",
        "hospital_impact": "Public health emergency: ICU admissions for hypoxemia, acute bronchitis, and cardiac arrhythmias rise sharply."
    },
    "Severe+": {
        "range": "> 500",
        "color": "#7030A0",
        "badge": "☠️ Severe+ (Emergency)",
        "general_health": "Severe health hazard. Equivalent to smoking 25-35 cigarettes a day for every resident in open air.",
        "hospital_impact": "Crisis levels: Pediatric and geriatric pulmonology wards at maximum capacity."
    }
}


# -----------------------------------------------------------------------------
# 2. GRADED RESPONSE ACTION PLAN (GRAP) POLICY ENGINE
# -----------------------------------------------------------------------------
def get_grap_stage(aqi: int) -> Dict[str, Any]:
    """
    Maps current/forecast AQI to Commission for Air Quality Management (CAQM)
    Graded Response Action Plan (GRAP) stage and mandatory regulatory directives.
    """
    if aqi <= 200:
        return {
            "stage": "Pre-GRAP / Normal",
            "roman": "Normal",
            "trigger": "AQI ≤ 200 (Moderate or better)",
            "status": "Green",
            "measures": [
                "Routine mechanized road sweeping and municipal dust cleaning.",
                "Enforcement of PUC (Pollution Under Control) certification at fuel stations.",
                "Standard municipal solid waste anti-burning inspections."
            ],
            "vehicle_restrictions": "None",
            "school_status": "Normal physical operations.",
            "work_advisory": "Standard in-office working."
        }
    elif 201 <= aqi <= 300:
        return {
            "stage": "GRAP Stage I: 'Poor'",
            "roman": "Stage I",
            "trigger": "AQI 201 - 300",
            "status": "Warning",
            "measures": [
                "Intensify mechanized sweeping and water sprinkling on major arterial roads.",
                "Strict anti-dust guidelines at construction and demolition sites (>500 sq m).",
                "Strict ban on open burning of biomass, leaves, and municipal solid waste.",
                "Synchronize traffic movements at bottlenecks to reduce idle vehicle idling."
            ],
            "vehicle_restrictions": "Strict checks on visibly polluting vehicles and outdated PUC certificates.",
            "school_status": "Avoid intense outdoor physical activities in early mornings.",
            "work_advisory": "Promote carpooling and public transit."
        }
    elif 301 <= aqi <= 400:
        return {
            "stage": "GRAP Stage II: 'Very Poor'",
            "roman": "Stage II",
            "trigger": "AQI 301 - 400",
            "status": "High Alert",
            "measures": [
                "Daily mechanical sweeping and water sprinkling along with dust suppressants.",
                "Uninterrupted power supply to minimize private diesel generator (DG) usage.",
                "Enforce differential parking fee hikes to discourage personal car usage.",
                "Increase frequency of Metro and CNG public bus transit services."
            ],
            "vehicle_restrictions": "Ban on Diesel Generator sets except for emergency healthcare and railway services.",
            "school_status": "Cancel outdoor sports and morning assemblies.",
            "work_advisory": "Encourage hybrid working and staggered office hours to disperse peak traffic."
        }
    elif 401 <= aqi <= 450:
        return {
            "stage": "GRAP Stage III: 'Severe'",
            "roman": "Stage III",
            "trigger": "AQI 401 - 450",
            "status": "Severe Crisis",
            "measures": [
                "Strict ban on all non-essential construction and demolition activities.",
                "Halt operations of stone crushers and brick kilns not on clean fuel.",
                "Intensify water sprinkling using anti-smog mist cannons on key corridors.",
                "Enhanced public transit fare discounts or frequency surge."
            ],
            "vehicle_restrictions": "Strict ban on BS-III Petrol and BS-IV Diesel 4-wheelers in NCR.",
            "school_status": "Discontinue physical classes for primary schools (Nursery to Class V); shift to online mode.",
            "work_advisory": "Public and private offices advised to operate at 50% capacity with mandatory WFH for remaining staff."
        }
    else:
        return {
            "stage": "GRAP Stage IV: 'Severe+' (Emergency)",
            "roman": "Stage IV",
            "trigger": "AQI > 450",
            "status": "Public Health Emergency",
            "measures": [
                "Complete stop on all construction, highway, and municipal infrastructure works.",
                "Entry ban on non-Delhi commercial trucks (except essential goods and LNG/CNG/Electric).",
                "Deployment of state-wide anti-smog guns and emergency industrial shutdowns.",
                "Emergency odd-even vehicle rationing scheme prepared for immediate rollout."
            ],
            "vehicle_restrictions": "Ban on medium/heavy diesel goods vehicles; BS-IV and below diesel cars strictly prohibited.",
            "school_status": "Discontinue physical classes for all students up to Class IX and Class XI; complete online schooling.",
            "work_advisory": "Mandatory 50% Work-From-Home for government & private sector; flexible emergency hours."
        }


# -----------------------------------------------------------------------------
# 3. GROUP-SPECIFIC ACTIONABLE ADVISORY ENGINE
# -----------------------------------------------------------------------------
def get_group_specific_advice(aqi: int, dominant_pollutant: str = "PM2.5") -> Dict[str, Any]:
    """
    Returns rich, actionable, group-specific guidance tailored to four distinct personas:
    1. Children & School Administration
    2. Asthma, COPD & Cardiovascular Patients
    3. Outdoor Workers (Traffic Police, Delivery Executives, Laborers)
    4. General Public & Commuters
    """
    if aqi <= 50:
        return _advice_good()
    elif aqi <= 100:
        return _advice_satisfactory()
    elif aqi <= 200:
        return _advice_moderate()
    elif aqi <= 300:
        return _advice_poor()
    elif aqi <= 400:
        return _advice_very_poor()
    elif aqi <= 500:
        return _advice_severe()
    else:
        return _advice_severe_plus()


def _advice_good():
    return {
        "schools": {
            "title": "Children & Schools: All Outdoor Activities Permitted",
            "outdoor_play": "Full outdoor playtime, competitive sports, and athletics encouraged.",
            "assembly": "Outdoor morning prayer assemblies fully safe.",
            "school_timings": "Regular scheduled school hours.",
            "mask_mandate": "No masks required.",
            "classroom_ventilation": "Open windows for fresh cross-ventilation throughout the day."
        },
        "patients": {
            "title": "Asthma & COPD: Optimal Outdoor Breathing Conditions",
            "inhaler_alert": "Maintain standard preventative medication as prescribed.",
            "morning_walks": "Morning and evening walks are completely safe and beneficial.",
            "air_purifier": "Natural ventilation is sufficient; air purifiers not required.",
            "warning_signs": "Routine symptom monitoring."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Standard Working Conditions",
            "mask_mandate": "No specialized respirator required.",
            "exposure_limit": "No special outdoor shift restrictions.",
            "hydration": "Standard hydration (2.5L/day).",
            "rest_breaks": "Standard occupational break schedules."
        },
        "general_public": {
            "title": "General Public: Ideal Atmospheric Quality",
            "commute_mode": "Cycling, walking, and open-air transit encouraged.",
            "ventilation_window": "Open windows freely all day.",
            "car_ac_mode": "Fresh Air Mode or Recirculate.",
            "indoor_plants": "Keep indoor greens healthy with standard watering."
        }
    }


def _advice_satisfactory():
    return {
        "schools": {
            "title": "Children & Schools: Standard Operations with Minor Precautions",
            "outdoor_play": "Outdoor sports and playground breaks allowed.",
            "assembly": "Outdoor assemblies permitted.",
            "school_timings": "Normal schedules.",
            "mask_mandate": "Optional for hypersensitive or allergic children.",
            "classroom_ventilation": "Open windows during sunny midday hours."
        },
        "patients": {
            "title": "Asthma & COPD: Mild Sensitivity Precautions",
            "inhaler_alert": "Keep rescue bronchodilator inhaler handy during outdoor travel.",
            "morning_walks": "Safe, but sensitive patients should avoid brisk cold jogging before sunrise.",
            "air_purifier": "Optional in bedrooms for sensitive individuals.",
            "warning_signs": "Mild throat tickle or dry cough on extended exposure."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Minimal Impact",
            "mask_mandate": "Cloth or surgical mask optional for dusty worksites.",
            "exposure_limit": "Normal shift lengths.",
            "hydration": "Drink 3 liters of water; keep nasal passages clean.",
            "rest_breaks": "Standard breaks."
        },
        "general_public": {
            "title": "General Public: Good Everyday Conditions",
            "commute_mode": "Public transit and personal vehicles normal.",
            "ventilation_window": "Open windows between 10 AM and 4 PM.",
            "car_ac_mode": "Fresh air mode acceptable; switch to recirculate in heavy traffic tunnels.",
            "indoor_plants": "Areca Palm, Snake Plant, and Money Plant help maintain indoor oxygen."
        }
    }


def _advice_moderate():
    return {
        "schools": {
            "title": "Children & Schools: Limit High-Intensity Outdoor Sports",
            "outdoor_play": "Replace endurance running or intense cardio with moderate low-exertion play.",
            "assembly": "Limit outdoor morning assembly to 10-15 minutes.",
            "school_timings": "Standard timings.",
            "mask_mandate": "Recommended for children with a history of asthma or bronchitis.",
            "classroom_ventilation": "Keep windows closed before 9 AM; ventilate between 12 PM - 3 PM."
        },
        "patients": {
            "title": "Asthma & COPD: Moderate Inversion Alert",
            "inhaler_alert": "Verify rescue inhaler dosage counter. Keep emergency spacer accessible.",
            "morning_walks": "Shift morning walks from 6 AM to 8:30 AM after the sun breaks the ground inversion.",
            "air_purifier": "Run air purifier on auto mode in bedrooms at night.",
            "warning_signs": "Increased coughing, nocturnal chest tightness, slight wheeze."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Dust & Particulate Protection",
            "mask_mandate": "Wear 3-ply surgical mask or N95 during high-traffic road hours.",
            "exposure_limit": "Cap continuous unmasked road duty to 4 hours.",
            "hydration": "Drink frequent sips of lukewarm water to lubricate pharyngeal mucosa.",
            "rest_breaks": "15-minute clean air indoor respite every 3 hours."
        },
        "general_public": {
            "title": "General Public: Conscious Exposure Reduction",
            "commute_mode": "Prefer AC buses or Metro; avoid open two-wheeler rides during rush hours.",
            "ventilation_window": "Open windows strictly between 12:00 PM and 3:30 PM only.",
            "car_ac_mode": "Switch Car AC to RECIRCULATE mode in traffic jams.",
            "indoor_plants": "Wipe dust off indoor plant leaves with a damp cloth to maintain transpiration."
        }
    }


def _advice_poor():
    return {
        "schools": {
            "title": "Children & Schools: Restrict Outdoor Sports & Morning Assembly",
            "outdoor_play": "Cancel competitive outdoor matches and heavy PE classes; move physical activities indoors.",
            "assembly": "Conduct morning assembly inside classrooms over the public address system.",
            "school_timings": "Consider delaying school start by 30-45 minutes to avoid peak morning smog.",
            "mask_mandate": "Mandatory pediatric N95 or well-fitting KF94 masks for all outdoor school transit.",
            "classroom_ventilation": "Keep doors and windows sealed. Use air purifiers if available in school halls."
        },
        "patients": {
            "title": "Asthma & COPD: High Respiratory Irritation Risk",
            "inhaler_alert": "Take preventative maintenance corticosteroid inhalers strictly as prescribed. Keep rescue inhaler in pocket.",
            "morning_walks": "STRICT BAN on early morning walks (5 AM - 8:30 AM) and evening strolls (7 PM - 10 PM). Exercise indoors.",
            "air_purifier": "Run True HEPA air purifier continuously on Medium/High in occupied rooms.",
            "warning_signs": "Peak Expiratory Flow (PEF) dropping below 80% of personal best; frequent nocturnal awakenings."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: N95 Respirator Mandate",
            "mask_mandate": "MANDATORY N95 or FFP2 respirator. Ensure snug seal over nose bridge and chin.",
            "exposure_limit": "Max continuous outdoor shift capped at 3 hours; rotate staff into indoor booths.",
            "hydration": "Drink 3.5 liters of fluids; perform warm saline gargles after ending shift.",
            "rest_breaks": "Mandatory 20-minute rest break in air-filtered rest rooms every 2.5 hours."
        },
        "general_public": {
            "title": "General Public: Actionable Protective Measures",
            "commute_mode": "Avoid non-essential two-wheeler or auto-rickshaw commuting. Travel via Metro.",
            "ventilation_window": "DO NOT open windows in early morning or late night. Brief 20-min ventilation at 1:30 PM.",
            "car_ac_mode": "ALWAYS keep car AC on RECIRCULATE mode to prevent sucking particulate matter into cabin.",
            "indoor_plants": "Avoid lighting incense sticks, dhoop, mosquito coils, or candles indoors."
        }
    }


def _advice_very_poor():
    return {
        "schools": {
            "title": "Children & Schools: Total Ban on Outdoor Play; GRAP Stage II Directives",
            "outdoor_play": "ABSOLUTE BAN on all outdoor sports, playground breaks, and physical training.",
            "assembly": "NO outdoor gatherings allowed. Conduct all briefings in closed classrooms.",
            "school_timings": "Schools should evaluate 1-hour delay in reporting time or hybrid options.",
            "mask_mandate": "MANDATORY certified N95 or KF94 mask for every child during travel.",
            "classroom_ventilation": "Seal windows tightly. Wet-mop classrooms twice daily; never dry-sweep."
        },
        "patients": {
            "title": "Asthma & COPD: Severe Bronchospasm Warning",
            "inhaler_alert": "High risk of acute asthma exacerbation. Check that spacer is clean and rescue inhaler has >20 doses.",
            "morning_walks": "ZERO OUTDOOR EXPOSURE. Do not step onto balconies or terraces during early hours.",
            "air_purifier": "Run HEPA air purifiers at maximum CADR setting. Seal door thresholds with draft stoppers.",
            "warning_signs": "Breathlessness while talking, persistent cough unresponsive to 2 puffs of rescue inhaler, chest tightness."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Hazardous Exposure Limits",
            "mask_mandate": "MANDATORY FFP2 / N95 respirator. Replace mask every 8 hours or if breathing resistance increases.",
            "exposure_limit": "Cap continuous exposure to 2 hours maximum. Employers must provide respiratory PPE.",
            "hydration": "Nasal saline spray / rinses after shift; drink warm fluids and ginger-honey decoction.",
            "rest_breaks": "Mandatory 30-minute indoor clean air rest every 2 hours."
        },
        "general_public": {
            "title": "General Public: Smog Alert Protocols",
            "commute_mode": "Use Delhi Metro / AC trains exclusively. Keep car windows rolled up 100% of the time.",
            "ventilation_window": "Keep windows CLOSED 24/7. Air exchange only via HEPA filter systems.",
            "car_ac_mode": "Car AC strictly on RECIRCULATE. Replace cabin pollen/HEPA filter if older than 6 months.",
            "indoor_plants": "Wet-mop floors with damp cloth; avoid any frying or cooking without high-power chimney."
        }
    }


def _advice_severe():
    return {
        "schools": {
            "title": "Children & Schools: Emergency Primary School Closure (GRAP Stage III)",
            "outdoor_play": "CRISIS: Complete shutdown of all outdoor and sports facilities.",
            "assembly": "Strict ban on any student gathering.",
            "school_timings": "Primary schools (Classes Nursery to V) must close physical operations and switch to online schooling under GRAP Stage III.",
            "mask_mandate": "Mandatory fitted N95 respirator without exhalation valve for all staff and students.",
            "classroom_ventilation": "Air filtration units in all indoor facilities; complete window lockdown."
        },
        "patients": {
            "title": "Asthma, COPD & Cardiac: Severe Clinical Threat",
            "inhaler_alert": "Keep oral corticosteroids / emergency action plan ready as advised by your pulmonologist.",
            "morning_walks": "STRICTLY FORBIDDEN. Stay in a sealed room with active HEPA filtration.",
            "air_purifier": "Operate purifiers on maximum boost continuously. Create a clean-air sanctuary room.",
            "warning_signs": "RED FLAG EMERGENCY: Cyanosis (blue lips/fingers), inability to speak full sentences, confusion, resting pulse > 115 bpm. Rush to hospital ER immediately."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Mandatory Work Restrictions & Respirators",
            "mask_mandate": "MANDATORY NIOSH N95 / FFP3 respirator with tight facial seal. Reusable half-face elastomeric respirator recommended for traffic police.",
            "exposure_limit": "Cut continuous outdoor duty to 1.5 hours max. Mandate indoor rotation.",
            "hydration": "Provide clean drinking water, ORS, and warm saline gargling stations at depots.",
            "rest_breaks": "Provide dedicated pressurized clean-air respite booths."
        },
        "general_public": {
            "title": "General Public: Toxic Atmospheric Smog Emergency",
            "commute_mode": "Cancel all non-essential travel. Do not venture outdoors for leisure or recreation.",
            "ventilation_window": "DO NOT OPEN ANY WINDOWS OR BALCONY DOORS. Tape window crevices if drafty.",
            "car_ac_mode": "Car AC strictly RECIRCULATE. Never roll down windows for tolls (use FASTag).",
            "indoor_plants": "Avoid lighting stoves without ventilation. Do not perform heavy physical cardio indoors."
        }
    }


def _advice_severe_plus():
    return {
        "schools": {
            "title": "Children & Schools: State-Wide School Closure (GRAP Stage IV Emergency)",
            "outdoor_play": "PUBLIC HEALTH EMERGENCY: All schools closed up to Class IX and Class XI. 100% online learning.",
            "assembly": "Schools completely closed for physical attendance.",
            "school_timings": "Remote learning mode activated across district.",
            "mask_mandate": "Children must remain indoors. If stepping out for medical needs, high-grade pediatric N95 mandatory.",
            "classroom_ventilation": "Facilities sealed."
        },
        "patients": {
            "title": "Asthma & COPD: Critical Health Emergency",
            "inhaler_alert": "Verify access to home nebulizer machine, oxygen saturation pulse oximeter, and emergency ambulance contacts.",
            "morning_walks": "STAY INDOORS. Ambient air is toxic and equivalent to smoking 30+ cigarettes.",
            "air_purifier": "Seal bedroom doors with towels at base. Run dual HEPA filters if available.",
            "warning_signs": "SPO2 dropping below 92%, severe intercostal retractions, unremitting chest pain. Call emergency ambulance (108/102) immediately."
        },
        "outdoor_workers": {
            "title": "Outdoor Workers: Emergency Labor Cease Mandate",
            "mask_mandate": "FFP3 / N99 or elastomeric respirators mandatory for essential security/police personnel.",
            "exposure_limit": "Complete suspension of all outdoor construction, manual earthwork, and non-essential delivery work.",
            "hydration": "Emergency medical monitoring of arterial oxygen saturation and respiratory health.",
            "rest_breaks": "Essential personnel must spend 60% of shift inside sealed air-conditioned stations."
        },
        "general_public": {
            "title": "General Public: Environmental Disaster Lockdown",
            "commute_mode": "Absolute stay-at-home advisory. Work From Home (WFH) activated across corporate and government offices.",
            "ventilation_window": "AIR LOCKDOWN: Keep all windows and exterior vents sealed shut.",
            "car_ac_mode": "Avoid driving. If driving for medical emergencies, car air recirculation mandatory.",
            "indoor_plants": "Maintain calm, avoid strenuous indoor exertion, drink warm herbal fluids."
        }
    }


# -----------------------------------------------------------------------------
# 4. PERSONAL AIR VULNERABILITY INDEX (AVI) ASSESSMENT
# -----------------------------------------------------------------------------
def calculate_personal_vulnerability_index(
    age: int,
    conditions: List[str],
    daily_outdoor_hours: float,
    activity_level: str,
    current_aqi: int
) -> Dict[str, Any]:
    """
    Computes a personalized 1-100 Air Vulnerability Index (AVI) score.
    Provides customized exposure budgets, required mask specifications, and warnings.
    """
    # 1. Age sensitivity factor (0 to 30)
    if age < 6:
        age_score = 28.0  # Underdeveloped respiratory airways
    elif age < 14:
        age_score = 22.0  # High respiratory rate per kg bodyweight
    elif age > 70:
        age_score = 30.0  # Reduced alveolar elasticity & cardiac reserve
    elif age > 60:
        age_score = 24.0
    else:
        age_score = 10.0

    # 2. Medical conditions factor (0 to 35)
    condition_weights = {
        "Asthma": 24.0,
        "COPD / Chronic Bronchitis": 30.0,
        "Heart Disease / Hypertension": 20.0,
        "Diabetes": 10.0,
        "Pregnancy": 22.0,
        "Allergies / Sinusitis": 12.0
    }
    cond_score = sum(condition_weights.get(c, 0.0) for c in conditions)
    cond_score = min(35.0, cond_score)

    # 3. Outdoor exposure factor (0 to 20)
    exp_score = min(20.0, daily_outdoor_hours * 2.8)

    # 4. Physical exertion multiplier
    activity_multiplier = {
        "Sedentary / Desk": 1.0,
        "Moderate (Walking / Light Chores)": 1.25,
        "Vigorous (Jogging / Heavy Manual Labor)": 1.6
    }.get(activity_level, 1.0)

    # Base personal sensitivity (0 to 85)
    base_sensitivity = (age_score + cond_score + exp_score) * activity_multiplier
    base_sensitivity = min(85.0, base_sensitivity)

    # Ambient AQI Hazard Multiplier (0 to 1.0)
    aqi_hazard = min(1.0, current_aqi / 400.0)

    # Final AVI Score (1 to 100)
    avi_score = int(round(min(100.0, max(5.0, (base_sensitivity * 0.5) + (aqi_hazard * 50.0)))))

    # Risk Tier classification
    if avi_score < 30:
        risk_tier = "Low Vulnerability"
        tier_color = "#00B050"
        outdoor_budget_mins = 240
        mask_recommendation = "Cloth or Surgical Mask optional in dusty areas."
    elif avi_score < 55:
        risk_tier = "Moderate Vulnerability"
        tier_color = "#FFC000"
        outdoor_budget_mins = 90
        mask_recommendation = "Standard 3-ply mask or N95 during road transit."
    elif avi_score < 75:
        risk_tier = "High Vulnerability"
        tier_color = "#FF7C80"
        outdoor_budget_mins = 35
        mask_recommendation = "Certified N95 / FFP2 respirator with tight nasal seal."
    else:
        risk_tier = "Critical Vulnerability"
        tier_color = "#C00000"
        outdoor_budget_mins = 10
        mask_recommendation = "NIOSH N95 / FFP3 mask mandatory; remain indoors in clean-air room."

    return {
        "avi_score": avi_score,
        "risk_tier": risk_tier,
        "tier_color": tier_color,
        "safe_outdoor_budget_minutes": outdoor_budget_mins,
        "prescribed_mask": mask_recommendation,
        "age_component": round(age_score, 1),
        "condition_component": round(cond_score, 1),
        "exposure_component": round(exp_score, 1)
    }


# -----------------------------------------------------------------------------
# 5. EXPORTABLE DAILY AIR HEALTH BULLETIN GENERATOR
# -----------------------------------------------------------------------------
def generate_daily_bulletin_text(
    city_name: str,
    current_aqi: int,
    aqi_category: str,
    dominant_pollutant: str,
    stubble_metrics: Dict[str, Any],
    forecast_df: Optional[Any] = None
) -> str:
    """Generates a comprehensive Markdown Daily Health Bulletin for export and sharing."""
    now_str = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    grap = get_grap_stage(current_aqi)
    advice = get_group_specific_advice(current_aqi, dominant_pollutant)

    # 24h and 48h forecast estimates
    fc_24 = current_aqi
    fc_48 = current_aqi
    if forecast_df is not None and not forecast_df.empty:
        if len(forecast_df) >= 24:
            fc_24 = int(forecast_df.iloc[23].get("predicted_aqi", current_aqi))
        if len(forecast_df) >= 48:
            fc_48 = int(forecast_df.iloc[47].get("predicted_aqi", current_aqi))

    report = f"""================================================================================
AIRAWARE DAILY POLLUTION SEASON & HEALTH BULLETIN
Location: {city_name}, India
Generated: {now_str}
================================================================================

1. CURRENT AIR QUALITY STATUS
--------------------------------------------------------------------------------
* Current AQI: {current_aqi} ({aqi_category})
* Dominant Toxic Pollutant: {dominant_pollutant}
* GRAP Regulatory Directive: {grap['stage']}
* Agricultural Stubble Fire Index: {stubble_metrics.get('stubble_fire_index', 'N/A')}/100 ({stubble_metrics.get('plume_risk', 'N/A')})
* Active Hotspots (350 km radius): {stubble_metrics.get('fire_count_350km', 0)} clusters

2. 24-72 HOUR FORECAST TRAJECTORY
--------------------------------------------------------------------------------
* Next 24 Hours Projected AQI: {fc_24}
* Next 48 Hours Projected AQI: {fc_48}
* Atmospheric Dispersion: Morning and nocturnal temperature inversions will
  significantly trap fine particulates near surface level.

3. GROUP-SPECIFIC ACTIONABLE PROTOCOLS
--------------------------------------------------------------------------------
A. CHILDREN & SCHOOLS:
   - Outdoor Play / Sports: {advice['schools']['outdoor_play']}
   - Morning Prayer Assembly: {advice['schools']['assembly']}
   - School Operational Status: {advice['schools']['school_timings']}
   - Mask Requirement: {advice['schools']['mask_mandate']}

B. ASTHMA, COPD & CARDIAC PATIENTS:
   - Medication Readiness: {advice['patients']['inhaler_alert']}
   - Morning / Evening Walks: {advice['patients']['morning_walks']}
   - Air Purifier Setting: {advice['patients']['air_purifier']}
   - Red Flag Emergency Signs: {advice['patients']['warning_signs']}

C. OUTDOOR WORKERS (POLICE / GIG WORKERS / LABOR):
   - Respiratory PPE: {advice['outdoor_workers']['mask_mandate']}
   - Shift Continuous Exposure Limit: {advice['outdoor_workers']['exposure_limit']}
   - Hydration & Respite: {advice['outdoor_workers']['hydration']} | {advice['outdoor_workers']['rest_breaks']}

D. GENERAL PUBLIC & COMMUTERS:
   - Transit Mode: {advice['general_public']['commute_mode']}
   - Car Air Conditioning: {advice['general_public']['car_ac_mode']}
   - Home Window Ventilation Window: {advice['general_public']['ventilation_window']}

4. GOVERNMENT REGULATORY COMPLIANCE (GRAP {grap['roman']})
--------------------------------------------------------------------------------
{chr(10).join(['* ' + m for m in grap['measures']])}
* Vehicle Restrictions: {grap['vehicle_restrictions']}
* Work Policy: {grap['work_advisory']}

================================================================================
Sustainable Development Goals: SDG 3 (Good Health) | SDG 11 (Sustainable Cities) | SDG 13 (Climate Action)
AirAware AI System - Open Source Environmental Intelligence
================================================================================
"""
    return report
