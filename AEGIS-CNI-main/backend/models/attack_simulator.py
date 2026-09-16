"""
Aegis-CNI: APT Attack Simulator

Simulates a realistic Advanced Persistent Threat campaign progressing through
MITRE ATT&CK stages over time. Generates the provenance graph snapshots that
feed into the GAE + LSTM pipeline for the live demo.

The demo scenario models an AIIMS-Delhi-2022-style attack:
    1. Phishing email → initial foothold (Week 1)
    2. PowerShell execution → persistence (Week 1-2)
    3. Discovery/enumeration of hospital network (Week 2)
    4. Lateral movement to critical database servers (Week 3)
    5. Data exfiltration + ransomware deployment (Week 3-4)

Compressed into a ~120-tick simulation for the hackathon demo.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from models.provenance_graph import ProvenanceGraphGenerator, ProvenanceNode, ProvenanceEdge


@dataclass
class SimulationConfig:
    """Configuration for the attack simulation."""
    total_ticks: int = 120
    tick_interval_ms: int = 500  # Time between ticks sent to frontend

    # Graph size parameters
    base_processes: int = 25
    base_files: int = 12
    base_sockets: int = 6

    # Attack progression timeline (tick ranges for each stage)
    # Stage transitions happen gradually, not instantly
    stage_timeline: Dict[int, Tuple[int, int]] = field(default_factory=lambda: {
        0: (0, 19),      # Normal Operations (ticks 0-19)
        1: (20, 34),     # Initial Access (ticks 20-34)
        2: (35, 49),     # Execution (ticks 35-49)
        3: (50, 59),     # Persistence (ticks 50-59)
        4: (60, 69),     # Evasion (ticks 60-69)
        5: (70, 84),     # Discovery (ticks 70-84)
        6: (85, 104),    # Lateral Movement (ticks 85-104)
        7: (105, 119),   # Impact (ticks 105-119)
    })

    # When SOAR should trigger autonomous containment
    soar_trigger_tick: int = 95  # During lateral movement phase
    soar_confidence_threshold: float = 0.85


class AttackSimulator:
    """
    Drives the complete APT simulation for the Aegis-CNI demo.

    Each tick:
      1. Generates a baseline normal provenance graph
      2. Injects attack subgraph for the current MITRE stage (if applicable)
      3. Returns the complete graph + metadata for the pipeline
    """

    def __init__(self, config: Optional[SimulationConfig] = None, seed: int = 42):
        self.config = config or SimulationConfig()
        self.graph_gen = ProvenanceGraphGenerator(seed=seed)
        self.current_tick = 0
        self.is_running = False

        # Persistent attack nodes that carry across ticks
        self._persistent_attack_nodes: List[ProvenanceNode] = []
        self._persistent_attack_edges: List[ProvenanceEdge] = []

        # Alert history
        self.alerts: List[Dict] = []
        self.soar_actions: List[Dict] = []
        self.audit_log: List[Dict] = []

    def reset(self):
        """Reset the simulation to initial state."""
        self.current_tick = 0
        self.is_running = False
        self._persistent_attack_nodes = []
        self._persistent_attack_edges = []
        self.alerts = []
        self.soar_actions = []
        self.audit_log = []

    def get_current_stage(self) -> int:
        """Get the MITRE ATT&CK stage for the current tick."""
        for stage, (start, end) in self.config.stage_timeline.items():
            if start <= self.current_tick <= end:
                return stage
        return 0

    def get_attack_intensity(self) -> float:
        """
        Get the attack intensity for the current tick.
        Ramps up gradually within each stage to model realistic progression.
        """
        stage = self.get_current_stage()
        if stage == 0:
            return 0.0

        start, end = self.config.stage_timeline[stage]
        progress = (self.current_tick - start) / max(1, end - start)

        # Sigmoid ramp-up within each stage
        intensity = 1 / (1 + math.exp(-8 * (progress - 0.3)))

        # Scale by stage severity (later stages are more intense)
        stage_multiplier = 0.3 + (stage / 7) * 0.7
        return min(1.0, intensity * stage_multiplier)

    def tick(self) -> Dict:
        """
        Advance the simulation by one tick.

        Returns a complete state snapshot including:
            - Provenance graph (nodes + edges)
            - Current MITRE stage
            - Attack intensity
            - Any new alerts or SOAR actions
        """
        if self.current_tick >= self.config.total_ticks:
            self.is_running = False
            return self._build_final_state()

        self.is_running = True
        timestamp = float(self.current_tick)
        stage = self.get_current_stage()
        intensity = self.get_attack_intensity()

        # ── Generate base normal graph ──
        nodes, edges = self.graph_gen.generate_normal_graph(
            num_processes=self.config.base_processes + self.current_tick % 5,
            num_files=self.config.base_files,
            num_sockets=self.config.base_sockets,
            timestamp=timestamp,
        )

        # ── Add persistent attack nodes from previous ticks ──
        nodes.extend(self._persistent_attack_nodes)
        edges.extend(self._persistent_attack_edges)

        # ── Inject new attack subgraph for current stage ──
        new_alerts = []
        new_soar_actions = []

        if stage > 0 and intensity > 0.1:
            pre_count = len(nodes)
            nodes, edges = self.graph_gen.inject_attack_subgraph(
                nodes, edges, stage=stage, intensity=intensity, timestamp=timestamp
            )
            post_count = len(nodes)

            # Track new attack nodes as persistent
            new_attack_nodes = nodes[pre_count:]
            for n in new_attack_nodes:
                if n not in self._persistent_attack_nodes:
                    self._persistent_attack_nodes.append(n)

            # Generate alerts for significant anomalies
            if intensity > 0.4:
                alert = self._generate_alert(stage, intensity, timestamp, new_attack_nodes)
                if alert:
                    new_alerts.append(alert)
                    self.alerts.append(alert)

        # ── SOAR Autonomous Actions ──
        if self.current_tick >= self.config.soar_trigger_tick and stage >= 6:
            soar_action = self._generate_soar_action(stage, intensity, timestamp)
            if soar_action and soar_action["id"] not in [a["id"] for a in self.soar_actions]:
                new_soar_actions.append(soar_action)
                self.soar_actions.append(soar_action)

                # Audit log entry
                audit = self._generate_audit_entry(soar_action, timestamp)
                self.audit_log.append(audit)

        # ── Build state snapshot ──
        state = {
            "tick": self.current_tick,
            "totalTicks": self.config.total_ticks,
            "timestamp": f"T+{self.current_tick * (self.config.tick_interval_ms / 1000):.1f}s",
            "currentStage": stage,
            "attackIntensity": intensity,
            "graph": {
                "nodes": [n.to_dict() for n in nodes],
                "links": [e.to_dict() for e in edges],
            },
            "newAlerts": new_alerts,
            "newSOARActions": new_soar_actions,
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
        }

        self.current_tick += 1
        return state

    def _generate_alert(self, stage: int, intensity: float,
                        timestamp: float, new_nodes: List[ProvenanceNode]) -> Optional[Dict]:
        """Generate an anomaly alert for the current attack stage."""
        stage_alerts = {
            1: "Suspicious outbound connection to unknown external IP detected",
            2: "Anomalous PowerShell execution with encoded payload observed",
            3: "Unauthorized registry modification — persistence mechanism detected",
            4: "Audit log tampering detected — potential evidence destruction",
            5: "Unusual network enumeration activity — port scanning detected",
            6: "Credential dump detected — lateral movement in progress",
            7: "CRITICAL: Mass data exfiltration to external C2 server",
        }

        stage_names = [
            "Normal", "Initial Access", "Execution", "Persistence",
            "Evasion", "Discovery", "Lateral Movement", "Impact"
        ]

        severity_map = {1: "low", 2: "medium", 3: "medium", 4: "high",
                        5: "high", 6: "critical", 7: "critical"}

        message = stage_alerts.get(stage, "Unknown anomaly")
        node_id = new_nodes[0].id if new_nodes else "unknown"

        return {
            "id": f"alert_{self.current_tick}_{stage}",
            "timestamp": f"T+{timestamp * 0.5:.0f}s",
            "severity": severity_map.get(stage, "medium"),
            "message": message,
            "nodeId": node_id,
            "mitreStage": stage_names[stage],
            "anomalyScore": round(intensity * 0.9, 2),
        }

    def _generate_soar_action(self, stage: int, intensity: float,
                              timestamp: float) -> Optional[Dict]:
        """Generate autonomous SOAR containment actions."""
        actions_by_stage = {
            6: {
                "type": "isolate_endpoint",
                "target": "SCADA-WS-07 (10.0.2.15)",
                "description": "Isolating compromised endpoint — lateral movement detected",
            },
            7: {
                "type": "block_ip",
                "target": "185.92.74.33 (External C2)",
                "description": "Blocking C2 communication — data exfiltration in progress",
            },
        }

        extra_actions = [
            {
                "type": "revoke_credential",
                "target": "plc_admin@domain.local",
                "description": "Revoking compromised credentials — unauthorized auth detected",
            },
            {
                "type": "snapshot_vm",
                "target": "VM-DB-HISTORIAN-01",
                "description": "Snapshotting VM state — preserving forensic artifacts",
            },
            {
                "type": "acl_update",
                "target": "Firewall Rule #847",
                "description": "Injecting emergency ACL — severing C2 egress channel",
            },
        ]

        action_base = actions_by_stage.get(stage)
        if not action_base:
            return None

        # Add extra actions at later ticks
        if self.current_tick > self.config.soar_trigger_tick + 5:
            extra = extra_actions[min(
                self.current_tick - self.config.soar_trigger_tick - 6,
                len(extra_actions) - 1
            )]
            action_base = extra

        return {
            "id": f"soar_{self.current_tick}_{action_base['type']}",
            "timestamp": f"T+{timestamp * 0.5:.0f}s",
            "type": action_base["type"],
            "target": action_base["target"],
            "status": "executed",
            "confidence": round(min(0.99, intensity * 1.1), 2),
            "description": action_base["description"],
        }

    def _generate_audit_entry(self, soar_action: Dict, timestamp: float) -> Dict:
        """Generate an immutable audit log entry for a SOAR action."""
        import hashlib
        import json

        # Create deterministic hash for audit integrity
        payload = json.dumps(soar_action, sort_keys=True)
        action_hash = hashlib.sha256(payload.encode()).hexdigest()

        return {
            "id": f"audit_{soar_action['id']}",
            "timestamp": f"T+{timestamp * 0.5:.0f}s",
            "action": f"SOAR: {soar_action['description']}",
            "actor": "AEGIS_AI",
            "details": f"Target: {soar_action['target']} | Confidence: {soar_action['confidence']}",
            "hash": action_hash,
        }

    def _build_final_state(self) -> Dict:
        """Build the final state when simulation completes."""
        return {
            "tick": self.config.total_ticks,
            "totalTicks": self.config.total_ticks,
            "timestamp": "SIMULATION COMPLETE",
            "currentStage": 7,
            "attackIntensity": 1.0,
            "graph": {"nodes": [], "links": []},
            "newAlerts": [],
            "newSOARActions": [],
            "nodeCount": 0,
            "edgeCount": 0,
            "complete": True,
        }
