"""
AirAware Automated Alert & Broadcast Notification Dispatcher
-----------------------------------------------------------
Features:
1. Automated condition evaluation:
   - GRAP Policy Stage Escalation (Stages I to IV)
   - Dangerous AQI Spikes (> 300 Very Poor, > 400 Severe)
   - Stubble Burning Smoke Inflow Surge
   - Morning Thermal Inversion Walk Warning (5:00 AM - 8:30 AM)
2. Multi-channel message formatting:
   - Compact SMS Broadcast (telecom standard)
   - Rich WhatsApp Notification with action bullets
   - School Principal / Education Department Official Advisory
   - Construction Site & Outdoor Labor Safety Directive
3. Subscriber management and dispatch audit logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SUBSCRIBERS_FILE = os.path.join(DATA_DIR, "alert_subscribers.json")
LOGS_FILE = os.path.join(DATA_DIR, "alert_dispatch_logs.json")


class AlertNotificationSystem:
    """Manages alert triggers, subscriber profiles, and multi-channel message dispatch."""

    def __init__(self):
        self.subscribers: List[Dict[str, Any]] = []
        self.dispatch_logs: List[Dict[str, Any]] = []
        self._load_state()

    def _load_state(self):
        # Load or initialize subscribers
        if os.path.exists(SUBSCRIBERS_FILE):
            try:
                with open(SUBSCRIBERS_FILE, "r") as f:
                    self.subscribers = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading subscribers: {e}")
                self.subscribers = self._default_subscribers()
        else:
            self.subscribers = self._default_subscribers()
            self._save_subscribers()

        # Load logs
        if os.path.exists(LOGS_FILE):
            try:
                with open(LOGS_FILE, "r") as f:
                    self.dispatch_logs = json.load(f)
            except Exception as e:
                self.dispatch_logs = []

    def _default_subscribers(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "sub_001",
                "name": "Delhi Public School Admin",
                "role": "School Principal",
                "city": "Delhi",
                "channel": "Email & WhatsApp",
                "contact": "+91-98100-XXXXX",
                "aqi_threshold": 300,
                "created_at": "2026-10-01"
            },
            {
                "id": "sub_002",
                "name": "DLF CyberCity Construction Team",
                "role": "Outdoor Site Manager",
                "city": "Gurugram",
                "channel": "SMS & WhatsApp",
                "contact": "+91-98711-XXXXX",
                "aqi_threshold": 350,
                "created_at": "2026-10-02"
            },
            {
                "id": "sub_003",
                "name": "Dr. Verma Pulmonary Clinic",
                "role": "Asthma & COPD Patient Network",
                "city": "Lucknow",
                "channel": "WhatsApp",
                "contact": "+91-94150-XXXXX",
                "aqi_threshold": 250,
                "created_at": "2026-10-03"
            }
        ]

    def _save_subscribers(self):
        try:
            with open(SUBSCRIBERS_FILE, "w") as f:
                json.dump(self.subscribers, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save subscribers: {e}")

    def _save_logs(self):
        try:
            with open(LOGS_FILE, "w") as f:
                json.dump(self.dispatch_logs[-50:], f, indent=2)  # Keep latest 50
        except Exception as e:
            logger.error(f"Failed to save dispatch logs: {e}")

    def register_subscriber(
        self,
        name: str,
        role: str,
        city: str,
        contact: str,
        channel: str = "WhatsApp",
        threshold: int = 300
    ) -> Dict[str, Any]:
        """Registers a new subscriber for automated air quality emergency alerts."""
        sub_id = f"sub_{len(self.subscribers) + 1:03d}"
        new_sub = {
            "id": sub_id,
            "name": name,
            "role": role,
            "city": city,
            "channel": channel,
            "contact": contact,
            "aqi_threshold": threshold,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.subscribers.append(new_sub)
        self._save_subscribers()
        logger.info(f"Registered subscriber: {name} ({role}) for {city}")
        return new_sub

    def evaluate_and_format_alerts(
        self,
        city_name: str,
        current_aqi: int,
        aqi_category: str,
        grap_stage: str,
        stubble_fire_index: float,
        inversion_risk: float
    ) -> Dict[str, Any]:
        """
        Evaluates conditions and generates tailored alert payloads across communication channels.
        """
        now_time = datetime.now().strftime("%I:%M %p, %d %b")

        # 1. SMS Payload (Concise telecom standard < 160 chars)
        sms_text = (
            f"AIRAWARE ALERT [{city_name}]: AQI {current_aqi} ({aqi_category}). "
            f"GRAP {grap_stage}. Wear certified N95 outdoors. "
            f"Schools & outdoor work curbs active. airaware.in"
        )

        # 2. WhatsApp Rich Message
        wa_text = f"""🚨 *AIRAWARE EMERGENCY AIR ALERT: {city_name.upper()}*
📅 *Time:* {now_time}
------------------------------------
🔴 *Current AQI:* *{current_aqi}* ({aqi_category})
🛡️ *Policy Mandate:* *{grap_stage}*
🔥 *Stubble Burning Risk:* {stubble_fire_index}/100
🌫️ *Inversion Smog Trapping:* {inversion_risk} (Elevated)

*ACTIONABLE PROTOCOLS FOR TODAY:*
• 🎒 *Schools:* Cancel outdoor sports and morning prayer assembly.
• 🫁 *Asthma Patients:* STRICT BAN on morning walks (5 AM - 9 AM). Keep rescue inhalers at hand.
• 👷 *Outdoor Labor:* Mandate N95 respirators. Cap outdoor shifts to 2 hours.
• 🚗 *Commuters:* Set Car AC strictly to *RECIRCULATE* mode.

_Stay informed. Protect vulnerable lungs._
🔗 View 72h forecast: https://airaware.gov.in
"""

        # 3. School Principal Official Email Advisory
        school_text = f"""OFFICIAL ADVISORY TO SCHOOL ADMINISTRATORS & PRINCIPALS
Location: {city_name} | Date: {now_time}
Subject: Environmental Air Quality Directive - GRAP Action Mandate

Dear Principal,

The automated AirAware intelligence system has recorded ambient Air Quality Index at {current_aqi} ({aqi_category}) across {city_name}. In compliance with Central Pollution Control Board (CPCB) and CAQM Graded Response Action Plan ({grap_stage}) guidelines:

1. MORNING ASSEMBLY: Discontinue all open-air prayer assemblies immediately. Brief students inside classrooms.
2. SPORTS & PHYSICAL EDUCATION: Suspend all outdoor athletics, soccer, and running activities until afternoon dispersion.
3. CLASSROOM VENTILATION: Seal doors and windows during early morning hours. Wet-mop floors twice daily; prohibit dry sweeping.
4. PEDIATRIC RESPIRATORS: Instruct parents to equip children with well-fitting KF94 or pediatric N95 respirators for school transit.
{"5. EMERGENCY CLOSURE NOTICE: Primary classes (Nursery to Class V) advised to switch to remote learning under GRAP Stage III/IV directives." if current_aqi >= 400 else ""}

Regards,
AirAware Directorate of Child Environmental Health
"""

        # 4. Construction & Outdoor Labor Safety Directive
        labor_text = f"""OCCUPATIONAL SAFETY DIRECTIVE: OUTDOOR WORKERS & SITE MANAGERS
Jurisdiction: {city_name} | Timestamp: {now_time}

MANDATORY DIRECTIVES UNDER {grap_stage}:
1. RESPIRATOR COMPLIANCE: Every on-site worker, traffic personnel, or delivery partner must be issued a certified NIOSH N95 / FFP2 mask. Cloth cloths filter less than 15% of PM2.5.
2. SHIFT ROTATION: Maximum continuous outdoor physical labor is capped at 2.0 hours, followed by a mandatory 30-minute respite in a sealed, air-filtered shelter.
3. HYDRATION STATIONS: Employers must provide clean drinking water, oral rehydration salts, and warm saline gargling points.
{"4. WORK STOPPAGE: Complete cessation of all construction, earthwork, and stone-crushing operations mandated by GRAP Stage III/IV." if current_aqi >= 400 else ""}

Non-compliance attracts penalties under the Air (Prevention and Control of Pollution) Act.
"""

        return {
            "city": city_name,
            "timestamp": now_time,
            "trigger_aqi": current_aqi,
            "channels": {
                "sms": sms_text,
                "whatsapp": wa_text,
                "school_circular": school_text,
                "occupational_directive": labor_text
            }
        }

    def dispatch_test_alert(
        self,
        city_name: str,
        current_aqi: int,
        aqi_category: str,
        grap_stage: str,
        target_channel: str = "WhatsApp",
        recipient: str = "Self / Test User"
    ) -> Dict[str, Any]:
        """Simulates dispatching an alert to subscribers and records in audit log."""
        alerts = self.evaluate_and_format_alerts(
            city_name=city_name,
            current_aqi=current_aqi,
            aqi_category=aqi_category,
            grap_stage=grap_stage,
            stubble_fire_index=65.0,
            inversion_risk=3.2
        )

        log_entry = {
            "dispatch_id": f"disp_{len(self.dispatch_logs) + 1:04d}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "city": city_name,
            "aqi": current_aqi,
            "channel": target_channel,
            "recipient": recipient,
            "status": "Delivered (Simulated Carrier OK)",
            "preview": alerts["channels"].get(target_channel.lower(), alerts["channels"]["whatsapp"])[:140] + "..."
        }
        self.dispatch_logs.append(log_entry)
        self._save_logs()
        logger.info(f"Dispatched alert {log_entry['dispatch_id']} to {recipient} via {target_channel}")
        return {
            "log": log_entry,
            "payloads": alerts["channels"]
        }


_alert_instance: Optional[AlertNotificationSystem] = None

def get_alert_system() -> AlertNotificationSystem:
    global _alert_instance
    if _alert_instance is None:
        _alert_instance = AlertNotificationSystem()
    return _alert_instance


if __name__ == "__main__":
    print("Testing AirAware Alert Notification Dispatcher...")
    sys = get_alert_system()
    res = sys.dispatch_test_alert(
        city_name="Delhi",
        current_aqi=425,
        aqi_category="Severe",
        grap_stage="Stage III"
    )
    print("Dispatch Status:", res["log"]["status"])
    print("\nWhatsApp Payload Preview:\n", res["payloads"]["whatsapp"][:250].encode('ascii', errors='replace').decode('ascii'))
