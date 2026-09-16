import math
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

# We mimic the data structures from provenance_graph so inference_engine works seamlessly
from models.provenance_graph import ProvenanceNode, ProvenanceEdge

@dataclass
class LiveConfig:
    total_ticks: int = 1000
    tick_interval_ms: int = 2000  # Scans every 2 seconds to save CPU
    malware_target: str = "notepad.exe"  # The process we want to detect and kill!

class LiveTelemetrySimulator:
    """
    Scans the actual Windows laptop using `psutil` to extract running processes.
    Builds a real Provenance Graph of the live system.
    """
    def __init__(self, config: Optional[LiveConfig] = None):
        self.config = config or LiveConfig()
        self.current_tick = 0
        self.is_running = False
        
        # We need a stable list of nodes to draw the graph nicely
        # But processes come and go, so we track a snapshot
        self.alerts = []
        self.soar_actions = []
        self.audit_log = []
        
        # Simulated MITRE Stage for the demo
        self.simulated_stage = 0
        self.simulated_intensity = 0.0

    def reset(self):
        self.current_tick = 0
        self.is_running = False
        self.alerts = []
        self.soar_actions = []
        self.audit_log = []
        self.simulated_stage = 0
        self.simulated_intensity = 0.0

    def get_attack_intensity(self, stage: int) -> float:
        if stage == 0: return 0.0
        return min(1.0, 0.4 + (stage * 0.1))

    def tick(self) -> Dict:
        self.is_running = True
        timestamp = float(self.current_tick)
        
        nodes = []
        edges = []
        
        # 1. Fetch REAL laptop processes!
        # Limit to ~40 random benign processes to not overwhelm the UI
        # But ALWAYS look for the malware_target
        count = 0
        target_found = False
        target_pid = None
        target_node_id = None
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'ppid', 'username']):
                try:
                    # Skip the System Idle Process and System
                    if proc.info['pid'] in [0, 4]:
                        continue
                        
                    proc_name = (proc.info['name'] or 'unknown').lower()
                    
                    # Check if it's our designated "Malware" (e.g. notepad.exe)
                    if self.config.malware_target in proc_name:
                        target_found = True
                        target_pid = proc.info['pid']
                        target_node_id = f"proc_{target_pid}"
                        n = ProvenanceNode(
                            id=target_node_id,
                            label=proc_name,
                            node_type="process",
                            features=np.random.normal(0, 0.5, 16),
                            anomaly_score=1.0, # Spike the anomaly score!
                            timestamp=timestamp
                        )
                        nodes.append(n)
                    elif count < 40:
                        # Benign process
                        n = ProvenanceNode(
                            id=f"proc_{proc.info['pid']}",
                            label=proc_name[:15], # limit length
                            node_type="process",
                            features=np.zeros(16),
                            anomaly_score=np.random.uniform(0.01, 0.05),
                            timestamp=timestamp
                        )
                        nodes.append(n)
                        
                        # Connect to parent if it's also in our list (just a hacky graph builder for the demo)
                        if proc.info['ppid']:
                            edges.append(ProvenanceEdge(
                                source=f"proc_{proc.info['ppid']}",
                                target=f"proc_{proc.info['pid']}",
                                edge_type="fork",
                                timestamp=timestamp
                            ))
                        count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            print(f"[Live Agent] Warning: psutil iterator error: {e}")

        # Ensure edges point to valid nodes
        valid_node_ids = {n.id for n in nodes}
        valid_edges = [e for e in edges if e.source in valid_node_ids and e.target in valid_node_ids]
        
        # 2. Progress the "Attack" if target is found
        new_alerts = []
        new_soar_actions = []
        
        if target_found:
            self.simulated_stage = min(7, self.simulated_stage + 1)
            self.simulated_intensity = self.get_attack_intensity(self.simulated_stage)
            
            # Generate Alert
            alert = {
                "id": f"alert_live_{self.current_tick}",
                "timestamp": f"T+{self.current_tick * 2}s",
                "severity": "critical",
                "message": f"CRITICAL: Malicious process detected on actual host: {self.config.malware_target}",
                "nodeId": target_node_id,
                "mitreStage": "Impact",
                "anomalyScore": 0.99
            }
            new_alerts.append(alert)
            
            # Generate REAL SOAR Action to kill it
            if self.simulated_stage >= 4:  # Give it a few ticks to show on the graph
                soar_action = {
                    "id": f"soar_live_kill_{target_pid}_{self.current_tick}",
                    "timestamp": f"T+{self.current_tick * 2}s",
                    "type": "kill_process",  # Special type for orchestrator!
                    "target": str(target_pid), # Target is the exact PID
                    "status": "pending_execution", # Let Orchestrator actually kill it
                    "confidence": 0.99,
                    "description": f"Autonomous termination of {self.config.malware_target} (PID: {target_pid})"
                }
                new_soar_actions.append(soar_action)
        else:
            # Cooldown if closed
            self.simulated_stage = max(0, self.simulated_stage - 1)
            self.simulated_intensity = self.get_attack_intensity(self.simulated_stage)

        # 3. Build state snapshot
        state = {
            "tick": self.current_tick,
            "totalTicks": self.config.total_ticks,
            "timestamp": f"LIVE T+{self.current_tick * 2}s",
            "currentStage": self.simulated_stage,
            "attackIntensity": self.simulated_intensity,
            "graph": {
                "nodes": [n.to_dict() for n in nodes],
                "links": [e.to_dict() for e in valid_edges],
            },
            "newAlerts": new_alerts,
            "newSOARActions": new_soar_actions,
            "nodeCount": len(nodes),
            "edgeCount": len(valid_edges),
            "is_live_mode": True # Flag to tell inference engine
        }

        self.current_tick += 1
        return state
