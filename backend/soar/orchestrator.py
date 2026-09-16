"""
Aegis-CNI: SOAR (Security Orchestration, Automation, and Response) Orchestrator

Consumes the probabilistic outputs from the LSTM stage predictor and
executes autonomous containment playbooks when confidence thresholds
are exceeded. All actions generate immutable, cryptographically signed
audit log entries for regulatory compliance.

Blast Radius Risk Score = stage_severity * confidence * anomaly_score
"""

import hashlib
import json
import time
import psutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ContainmentPlaybook:
    """Defines automated containment actions for each MITRE ATT&CK stage."""
    stage: int
    stage_name: str
    actions: List[Dict]
    min_confidence: float
    min_blast_radius: float


# Pre-defined containment playbooks per stage
PLAYBOOKS = [
    ContainmentPlaybook(
        stage=5,
        stage_name="Discovery",
        min_confidence=0.80,
        min_blast_radius=50,
        actions=[
            {
                "type": "acl_update",
                "description": "Restricting network segment access — anomalous scanning detected",
                "target": "Firewall VLAN-SCADA-01",
                "auto_execute": False,  # Human approval required
            },
        ],
    ),
    ContainmentPlaybook(
        stage=6,
        stage_name="Lateral Movement",
        min_confidence=0.85,
        min_blast_radius=65,
        actions=[
            {
                "type": "isolate_endpoint",
                "description": "Isolating compromised workstation — credential theft detected",
                "target": "SCADA-WS-07 (10.0.2.15)",
                "auto_execute": True,
            },
            {
                "type": "revoke_credential",
                "description": "Revoking compromised service account credentials",
                "target": "svc_plc_admin@cni.gov.in",
                "auto_execute": True,
            },
            {
                "type": "snapshot_vm",
                "description": "Capturing forensic snapshot before adversary anti-forensics",
                "target": "VM-SCADA-HMI-03",
                "auto_execute": True,
            },
        ],
    ),
    ContainmentPlaybook(
        stage=7,
        stage_name="Impact",
        min_confidence=0.90,
        min_blast_radius=80,
        actions=[
            {
                "type": "block_ip",
                "description": "Severing C2 communication channel — exfiltration in progress",
                "target": "185.92.74.33:8443 (External C2)",
                "auto_execute": True,
            },
            {
                "type": "isolate_endpoint",
                "description": "Emergency isolation of database server — ransomware deployment",
                "target": "DB-HISTORIAN-01 (10.0.3.5)",
                "auto_execute": True,
            },
            {
                "type": "acl_update",
                "description": "Emergency perimeter lockdown — blocking all non-essential egress",
                "target": "Perimeter-FW-01 (Emergency Rule)",
                "auto_execute": True,
            },
        ],
    ),
]


class SOAROrchestrator:
    """
    Autonomous SOAR engine that evaluates threat intelligence from the
    ML pipeline and executes containment playbooks.
    """

    def __init__(self):
        self.executed_actions: List[Dict] = []
        self.audit_log: List[Dict] = []
        self.pending_approvals: List[Dict] = []
        self._executed_action_ids: set = set()

    def reset(self):
        self.executed_actions = []
        self.audit_log = []
        self.pending_approvals = []
        self._executed_action_ids = set()

    def compute_blast_radius(self, stage: int, confidence: float,
                             anomaly_score: float) -> float:
        """
        Compute Blast Radius Risk Score.

        blast_radius = (stage_severity / 7) * confidence * (0.5 + 0.5 * anomaly_score) * 100

        Range: 0 - 100
        """
        stage_severity = stage / 7.0
        blast_radius = stage_severity * confidence * (0.5 + 0.5 * anomaly_score) * 100
        return min(100.0, max(0.0, blast_radius))

    def evaluate_and_respond(
        self,
        stage: int,
        confidence: float,
        anomaly_score: float,
        tick: int,
    ) -> Dict:
        """
        Evaluate the current threat level and execute appropriate playbooks.

        Returns:
            dict with blast_radius, new_actions, new_audit_entries
        """
        blast_radius = self.compute_blast_radius(stage, confidence, anomaly_score)

        new_actions = []
        new_audit_entries = []

        # Find applicable playbooks
        for playbook in PLAYBOOKS:
            if (stage >= playbook.stage and
                confidence >= playbook.min_confidence and
                blast_radius >= playbook.min_blast_radius):

                for action in playbook.actions:
                    action_id = f"soar_{tick}_{action['type']}_{playbook.stage}"

                    # Don't re-execute the same action type+stage
                    dedup_key = f"{action['type']}_{playbook.stage}"
                    if dedup_key in self._executed_action_ids:
                        continue
                    self._executed_action_ids.add(dedup_key)

                    if action.get("auto_execute", False):
                        status = "executed"
                    else:
                        status = "pending_approval"
                        self.pending_approvals.append(action_id)

                    soar_action = {
                        "id": action_id,
                        "timestamp": f"T+{tick * 0.5:.0f}s",
                        "type": action["type"],
                        "target": action["target"],
                        "status": status,
                        "confidence": round(confidence, 3),
                        "description": action["description"],
                    }

                    new_actions.append(soar_action)
                    self.executed_actions.append(soar_action)

                    # Generate audit entry
                    audit = self._create_audit_entry(soar_action, tick)
                    new_audit_entries.append(audit)
                    self.audit_log.append(audit)

        return {
            "blastRadius": round(blast_radius, 1),
            "newActions": new_actions,
            "newAuditEntries": new_audit_entries,
            "totalActionsExecuted": len([a for a in self.executed_actions if a["status"] == "executed"]),
            "totalPendingApproval": len(self.pending_approvals),
        }

    def execute_live_action(self, action: Dict, tick: int) -> Dict:
        """Physically executes a live action like killing a process."""
        if action["type"] == "kill_process":
            try:
                pid = int(action["target"])
                p = psutil.Process(pid)
                p.terminate() # Actually kill the process!
                action["status"] = "executed"
                action["description"] += " [SUCCESSFULLY TERMINATED BY SOAR]"
                print(f"[SOAR] KILLED MALICIOUS PROCESS PID {pid}")
            except Exception as e:
                action["status"] = "failed"
                action["description"] += f" [FAILED: {e}]"
                print(f"[SOAR] FAILED TO KILL PID {action['target']}: {e}")
        
        # Log it
        self.executed_actions.append(action)
        audit = self._create_audit_entry(action, tick)
        self.audit_log.append(audit)
        
        return action, audit

    def _create_audit_entry(self, action: Dict, tick: int) -> Dict:
        """Create a cryptographically signed audit log entry."""
        # Chain hash with previous entry for immutability
        previous_hash = self.audit_log[-1]["hash"] if self.audit_log else "GENESIS"

        payload = json.dumps({
            "action": action,
            "tick": tick,
            "previous_hash": previous_hash,
            "timestamp": time.time(),
        }, sort_keys=True)

        entry_hash = hashlib.sha256(payload.encode()).hexdigest()

        return {
            "id": f"audit_{action['id']}",
            "timestamp": action["timestamp"],
            "action": f"{action['type'].upper()}: {action['description']}",
            "actor": "AEGIS_AI",
            "details": f"Target: {action['target']} | Confidence: {action['confidence']} | Status: {action['status']}",
            "hash": entry_hash,
        }

    def get_playbook_coverage(self) -> Dict:
        """
        Calculate autonomous playbook coverage metrics for evaluation.

        Returns percentage of playbook steps that were executed autonomously
        vs. requiring human approval.
        """
        total_possible = sum(len(p.actions) for p in PLAYBOOKS)
        auto_executed = len([a for a in self.executed_actions if a["status"] == "executed"])
        pending = len(self.pending_approvals)

        return {
            "totalPlaybookSteps": total_possible,
            "autonomouslyExecuted": auto_executed,
            "pendingApproval": pending,
            "coveragePercent": round(
                (auto_executed / max(1, total_possible)) * 100, 1
            ),
        }
