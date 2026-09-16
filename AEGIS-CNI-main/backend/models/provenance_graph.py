"""
Aegis-CNI: Synthetic Provenance Graph Generator

Generates realistic system provenance graphs that model host-level activity
in a cyber-physical infrastructure environment. Each graph represents a
time-windowed snapshot of processes, files, sockets, users and their
causal relationships — modeled after the W3C PROV-DM standard used by the
ProvICS dataset.

The generator creates both normal (benign) operational graphs and graphs
containing injected attack subgraphs corresponding to specific MITRE
ATT&CK stages.
"""

import random
import uuid
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np


@dataclass
class ProvenanceNode:
    """A node in the provenance graph — represents a system entity."""
    id: str
    label: str
    node_type: str  # process, file, socket, user, registry, alert
    features: np.ndarray = field(default_factory=lambda: np.zeros(16))
    anomaly_score: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type,
            "anomalyScore": self.anomaly_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ProvenanceEdge:
    """An edge in the provenance graph — represents a causal relationship."""
    source: str
    target: str
    edge_type: str  # fork, exec, read, write, connect, recv, send, auth, etc.
    timestamp: float = 0.0
    features: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def to_dict(self):
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "timestamp": self.timestamp,
        }


# ─── Normal Operation Templates ─────────────────────────────────────────────

NORMAL_PROCESSES = [
    "sshd", "crond", "systemd", "rsyslogd", "httpd", "nginx",
    "postgres", "redis-server", "dockerd", "containerd",
    "node_exporter", "prometheus", "grafana-server",
    "modbus_slave", "plc_controller", "scada_hmi",
    "opcua_server", "mqtt_broker", "historian_db",
    "ntp_sync", "snmpd", "watchdog_service",
]

NORMAL_FILES = [
    "/var/log/syslog", "/var/log/auth.log", "/etc/passwd",
    "/etc/shadow", "/tmp/session_cache", "/var/run/pid",
    "/opt/scada/config.yml", "/opt/scada/data/readings.db",
    "/opt/plc/firmware/v2.3.bin", "/var/lib/historian/ts_data.db",
    "/etc/modbus/register_map.conf", "/var/log/audit/audit.log",
]

NORMAL_SOCKETS = [
    "tcp://10.0.1.10:502", "tcp://10.0.1.20:4840",
    "tcp://10.0.2.5:1883", "tcp://10.0.3.1:5432",
    "tcp://10.0.3.2:6379", "udp://10.0.1.1:123",
    "tcp://192.168.1.100:443", "tcp://10.0.4.1:9090",
]

NORMAL_USERS = ["root", "scada_operator", "plc_admin", "db_readonly", "monitoring"]


def _uid() -> str:
    return str(uuid.uuid4())[:8]


class ProvenanceGraphGenerator:
    """
    Generates synthetic provenance graphs for Aegis-CNI.

    Normal graphs model routine SCADA/ICS operations with deterministic
    process trees, file I/O, and network communication. Attack graphs
    inject adversarial substructures matching MITRE ATT&CK ICS stages.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_normal_graph(
        self,
        num_processes: int = 30,
        num_files: int = 15,
        num_sockets: int = 8,
        timestamp: float = 0.0,
    ) -> Tuple[List[ProvenanceNode], List[ProvenanceEdge]]:
        """Generate a normal operational provenance graph."""
        nodes: List[ProvenanceNode] = []
        edges: List[ProvenanceEdge] = []

        # ── Create process nodes ──
        process_ids = []
        for i in range(num_processes):
            proc_name = self.rng.choice(NORMAL_PROCESSES)
            pid = _uid()
            node = ProvenanceNode(
                id=f"proc_{pid}",
                label=f"{proc_name}",
                node_type="process",
                features=self._normal_process_features(),
                timestamp=timestamp + self.rng.uniform(0, 5),
                metadata={"pid": str(self.rng.randint(1000, 65535)), "name": proc_name},
            )
            nodes.append(node)
            process_ids.append(node.id)

        # ── Create file nodes ──
        file_ids = []
        for i in range(num_files):
            path = self.rng.choice(NORMAL_FILES)
            fid = _uid()
            node = ProvenanceNode(
                id=f"file_{fid}",
                label=path.split("/")[-1],
                node_type="file",
                features=self._normal_file_features(),
                timestamp=timestamp,
                metadata={"path": path},
            )
            nodes.append(node)
            file_ids.append(node.id)

        # ── Create socket nodes ──
        socket_ids = []
        for i in range(min(num_sockets, len(NORMAL_SOCKETS))):
            addr = NORMAL_SOCKETS[i]
            sid = _uid()
            node = ProvenanceNode(
                id=f"sock_{sid}",
                label=addr.split("://")[1],
                node_type="socket",
                features=self._normal_socket_features(),
                timestamp=timestamp,
                metadata={"address": addr},
            )
            nodes.append(node)
            socket_ids.append(node.id)

        # ── Create user nodes ──
        user_ids = []
        for uname in NORMAL_USERS:
            uid = _uid()
            node = ProvenanceNode(
                id=f"user_{uid}",
                label=uname,
                node_type="user",
                features=self._normal_user_features(),
                timestamp=timestamp,
                metadata={"username": uname},
            )
            nodes.append(node)
            user_ids.append(node.id)

        # ── Create edges (process trees + I/O) ──
        # Process fork trees (parent-child)
        for i in range(1, len(process_ids)):
            parent_idx = self.rng.randint(0, max(0, i - 1))
            edges.append(ProvenanceEdge(
                source=process_ids[parent_idx],
                target=process_ids[i],
                edge_type="fork",
                timestamp=timestamp + self.rng.uniform(0, 5),
                features=self._normal_edge_features(),
            ))

        # Process → file reads/writes
        for pid in process_ids:
            num_file_ops = self.rng.randint(0, 3)
            for _ in range(num_file_ops):
                fid = self.rng.choice(file_ids)
                op = self.rng.choice(["read", "write"])
                edges.append(ProvenanceEdge(
                    source=pid, target=fid, edge_type=op,
                    timestamp=timestamp + self.rng.uniform(0, 5),
                    features=self._normal_edge_features(),
                ))

        # Process → socket connections
        for pid in self.rng.sample(process_ids, min(6, len(process_ids))):
            sid = self.rng.choice(socket_ids)
            op = self.rng.choice(["connect", "send", "recv"])
            edges.append(ProvenanceEdge(
                source=pid, target=sid, edge_type=op,
                timestamp=timestamp + self.rng.uniform(0, 5),
                features=self._normal_edge_features(),
            ))

        # User → process auth
        for uid in user_ids:
            target_proc = self.rng.choice(process_ids)
            edges.append(ProvenanceEdge(
                source=uid, target=target_proc, edge_type="auth",
                timestamp=timestamp + self.rng.uniform(0, 2),
                features=self._normal_edge_features(),
            ))

        return nodes, edges

    def inject_attack_subgraph(
        self,
        nodes: List[ProvenanceNode],
        edges: List[ProvenanceEdge],
        stage: int,
        intensity: float = 0.5,
        timestamp: float = 0.0,
    ) -> Tuple[List[ProvenanceNode], List[ProvenanceEdge]]:
        """
        Inject adversarial subgraph patterns corresponding to MITRE ATT&CK stages.

        Stages:
            1 = Initial Access
            2 = Execution
            3 = Persistence
            4 = Evasion
            5 = Discovery
            6 = Lateral Movement
            7 = Impact
        """
        process_nodes = [n for n in nodes if n.node_type == "process"]
        file_nodes = [n for n in nodes if n.node_type == "file"]
        socket_nodes = [n for n in nodes if n.node_type == "socket"]

        if stage == 1:  # Initial Access — spear phishing → malicious process
            mal_proc = ProvenanceNode(
                id=f"proc_mal_{_uid()}", label="outlook_macro.exe",
                node_type="process", features=self._attack_process_features(intensity),
                anomaly_score=0.3 * intensity, timestamp=timestamp,
                metadata={"name": "outlook_macro.exe", "suspicious": "true"},
            )
            c2_socket = ProvenanceNode(
                id=f"sock_c2_{_uid()}", label="185.92.xx.xx:443",
                node_type="socket", features=self._attack_socket_features(intensity),
                anomaly_score=0.4 * intensity, timestamp=timestamp,
                metadata={"address": "tcp://185.92.74.33:443", "geo": "unknown"},
            )
            nodes.extend([mal_proc, c2_socket])
            # Email process spawns malicious child
            if process_nodes:
                edges.append(ProvenanceEdge(
                    source=self.rng.choice(process_nodes).id, target=mal_proc.id,
                    edge_type="exec", timestamp=timestamp,
                    features=self._attack_edge_features(intensity),
                ))
            # Malicious process connects to C2
            edges.append(ProvenanceEdge(
                source=mal_proc.id, target=c2_socket.id,
                edge_type="connect", timestamp=timestamp + 0.5,
                features=self._attack_edge_features(intensity),
            ))

        elif stage == 2:  # Execution — PowerShell/script execution
            ps_proc = ProvenanceNode(
                id=f"proc_ps_{_uid()}", label="powershell.exe",
                node_type="process", features=self._attack_process_features(intensity),
                anomaly_score=0.5 * intensity, timestamp=timestamp,
                metadata={"name": "powershell.exe", "args": "-enc BASE64PAYLOAD"},
            )
            script_file = ProvenanceNode(
                id=f"file_scr_{_uid()}", label="payload.ps1",
                node_type="file", features=self._attack_file_features(intensity),
                anomaly_score=0.45 * intensity, timestamp=timestamp,
                metadata={"path": "/tmp/.cache/payload.ps1"},
            )
            nodes.extend([ps_proc, script_file])
            # Malicious proc from stage 1 spawns PowerShell
            mal_procs = [n for n in nodes if "mal" in n.id or "ps" in n.id]
            parent = self.rng.choice(mal_procs) if mal_procs else self.rng.choice(process_nodes)
            edges.append(ProvenanceEdge(
                source=parent.id, target=ps_proc.id,
                edge_type="fork", timestamp=timestamp,
                features=self._attack_edge_features(intensity),
            ))
            edges.append(ProvenanceEdge(
                source=ps_proc.id, target=script_file.id,
                edge_type="write", timestamp=timestamp + 0.3,
                features=self._attack_edge_features(intensity),
            ))

        elif stage == 3:  # Persistence — registry/scheduled task
            reg_node = ProvenanceNode(
                id=f"reg_pers_{_uid()}", label="HKLM\\Run\\svc_update",
                node_type="registry", features=self._attack_process_features(intensity),
                anomaly_score=0.55 * intensity, timestamp=timestamp,
                metadata={"key": "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run"},
            )
            svc_proc = ProvenanceNode(
                id=f"proc_svc_{_uid()}", label="svc_update.exe",
                node_type="process", features=self._attack_process_features(intensity),
                anomaly_score=0.5 * intensity, timestamp=timestamp,
                metadata={"name": "svc_update.exe", "persistence": "true"},
            )
            nodes.extend([reg_node, svc_proc])
            attack_procs = [n for n in nodes if n.anomaly_score > 0.2]
            parent = self.rng.choice(attack_procs) if attack_procs else self.rng.choice(process_nodes)
            edges.append(ProvenanceEdge(
                source=parent.id, target=reg_node.id,
                edge_type="modify_reg", timestamp=timestamp,
                features=self._attack_edge_features(intensity),
            ))
            edges.append(ProvenanceEdge(
                source=reg_node.id, target=svc_proc.id,
                edge_type="exec", timestamp=timestamp + 1,
                features=self._attack_edge_features(intensity),
            ))

        elif stage == 4:  # Evasion — log tampering, masquerading
            tampered_log = ProvenanceNode(
                id=f"file_tamp_{_uid()}", label="audit.log.bak",
                node_type="file", features=self._attack_file_features(intensity),
                anomaly_score=0.6 * intensity, timestamp=timestamp,
                metadata={"path": "/var/log/audit/audit.log", "action": "truncated"},
            )
            nodes.append(tampered_log)
            attack_procs = [n for n in nodes if n.anomaly_score > 0.2 and n.node_type == "process"]
            if attack_procs:
                edges.append(ProvenanceEdge(
                    source=self.rng.choice(attack_procs).id, target=tampered_log.id,
                    edge_type="write", timestamp=timestamp,
                    features=self._attack_edge_features(intensity),
                ))

        elif stage == 5:  # Discovery — network enumeration
            for i in range(3):
                scan_proc = ProvenanceNode(
                    id=f"proc_scan_{_uid()}", label=self.rng.choice(["nmap", "arp-scan", "net_view"]),
                    node_type="process", features=self._attack_process_features(intensity),
                    anomaly_score=0.65 * intensity, timestamp=timestamp + i * 0.5,
                    metadata={"name": "network_scanner"},
                )
                scan_target = ProvenanceNode(
                    id=f"sock_scan_{_uid()}", label=f"10.0.{self.rng.randint(1,5)}.{self.rng.randint(1,254)}",
                    node_type="socket", features=self._attack_socket_features(intensity),
                    anomaly_score=0.3 * intensity, timestamp=timestamp + i * 0.5,
                )
                nodes.extend([scan_proc, scan_target])
                attack_procs = [n for n in nodes if n.anomaly_score > 0.3 and n.node_type == "process"]
                parent = self.rng.choice(attack_procs) if attack_procs else self.rng.choice(process_nodes)
                edges.append(ProvenanceEdge(
                    source=parent.id, target=scan_proc.id,
                    edge_type="fork", timestamp=timestamp + i * 0.5,
                    features=self._attack_edge_features(intensity),
                ))
                edges.append(ProvenanceEdge(
                    source=scan_proc.id, target=scan_target.id,
                    edge_type="connect", timestamp=timestamp + i * 0.5 + 0.1,
                    features=self._attack_edge_features(intensity),
                ))

        elif stage == 6:  # Lateral Movement — credential theft + remote execution
            cred_file = ProvenanceNode(
                id=f"file_cred_{_uid()}", label="lsass_dump.dmp",
                node_type="file", features=self._attack_file_features(intensity),
                anomaly_score=0.8 * intensity, timestamp=timestamp,
                metadata={"path": "/tmp/.cache/lsass_dump.dmp"},
            )
            remote_proc = ProvenanceNode(
                id=f"proc_lat_{_uid()}", label="psexec_svc.exe",
                node_type="process", features=self._attack_process_features(intensity),
                anomaly_score=0.75 * intensity, timestamp=timestamp + 1,
                metadata={"name": "psexec_svc.exe", "remote_host": "10.0.2.15"},
            )
            stolen_user = ProvenanceNode(
                id=f"user_stolen_{_uid()}", label="plc_admin (compromised)",
                node_type="user", features=self._attack_process_features(intensity),
                anomaly_score=0.7 * intensity, timestamp=timestamp,
                metadata={"username": "plc_admin", "compromised": "true"},
            )
            nodes.extend([cred_file, remote_proc, stolen_user])
            attack_procs = [n for n in nodes if n.anomaly_score > 0.3 and n.node_type == "process"]
            parent = self.rng.choice(attack_procs) if attack_procs else self.rng.choice(process_nodes)
            edges.append(ProvenanceEdge(
                source=parent.id, target=cred_file.id,
                edge_type="write", timestamp=timestamp,
                features=self._attack_edge_features(intensity),
            ))
            edges.append(ProvenanceEdge(
                source=stolen_user.id, target=remote_proc.id,
                edge_type="auth", timestamp=timestamp + 0.5,
                features=self._attack_edge_features(intensity),
            ))
            if socket_nodes:
                edges.append(ProvenanceEdge(
                    source=remote_proc.id, target=self.rng.choice(socket_nodes).id,
                    edge_type="connect", timestamp=timestamp + 1.5,
                    features=self._attack_edge_features(intensity),
                ))

        elif stage == 7:  # Impact — data exfiltration or process manipulation
            exfil_socket = ProvenanceNode(
                id=f"sock_exfil_{_uid()}", label="45.xx.xx.xx:8443",
                node_type="socket", features=self._attack_socket_features(intensity),
                anomaly_score=0.9 * intensity, timestamp=timestamp,
                metadata={"address": "tcp://45.33.91.12:8443", "data_size": "2.3GB"},
            )
            alert_node = ProvenanceNode(
                id=f"alert_{_uid()}", label="CRITICAL: Data Exfiltration",
                node_type="alert", features=self._attack_process_features(intensity),
                anomaly_score=0.95, timestamp=timestamp,
                metadata={"severity": "critical", "mitre": "T0882"},
            )
            nodes.extend([exfil_socket, alert_node])
            attack_procs = [n for n in nodes if n.anomaly_score > 0.5 and n.node_type == "process"]
            if attack_procs:
                edges.append(ProvenanceEdge(
                    source=self.rng.choice(attack_procs).id, target=exfil_socket.id,
                    edge_type="send", timestamp=timestamp,
                    features=self._attack_edge_features(intensity),
                ))
            edges.append(ProvenanceEdge(
                source=exfil_socket.id, target=alert_node.id,
                edge_type="connect", timestamp=timestamp + 0.1,
                features=self._attack_edge_features(intensity),
            ))

        return nodes, edges

    # ─── Feature Generators ──────────────────────────────────────────────

    def _normal_process_features(self) -> np.ndarray:
        """Normal process feature vector — low entropy, regular patterns."""
        return self.np_rng.normal(loc=0.0, scale=0.2, size=16).astype(np.float32)

    def _normal_file_features(self) -> np.ndarray:
        return self.np_rng.normal(loc=0.1, scale=0.15, size=16).astype(np.float32)

    def _normal_socket_features(self) -> np.ndarray:
        return self.np_rng.normal(loc=-0.1, scale=0.2, size=16).astype(np.float32)

    def _normal_user_features(self) -> np.ndarray:
        return self.np_rng.normal(loc=0.05, scale=0.1, size=16).astype(np.float32)

    def _normal_edge_features(self) -> np.ndarray:
        return self.np_rng.normal(loc=0.0, scale=0.15, size=8).astype(np.float32)

    def _attack_process_features(self, intensity: float) -> np.ndarray:
        """Attack process features — shifted distribution that the GAE should detect."""
        base = self.np_rng.normal(loc=0.0, scale=0.2, size=16).astype(np.float32)
        # Shift certain dimensions to create detectable anomaly
        base[0] += 1.5 * intensity   # Unusual syscall frequency
        base[3] += 1.2 * intensity   # High memory usage
        base[7] += 2.0 * intensity   # Unusual network I/O
        base[11] += 1.8 * intensity  # Abnormal child process count
        return base

    def _attack_file_features(self, intensity: float) -> np.ndarray:
        base = self.np_rng.normal(loc=0.1, scale=0.15, size=16).astype(np.float32)
        base[2] += 2.0 * intensity   # Large file write
        base[5] += 1.5 * intensity   # Unusual file path depth
        base[9] += 1.3 * intensity   # Entropy of file content
        return base

    def _attack_socket_features(self, intensity: float) -> np.ndarray:
        base = self.np_rng.normal(loc=-0.1, scale=0.2, size=16).astype(np.float32)
        base[1] += 2.5 * intensity   # Unusual destination port
        base[4] += 2.0 * intensity   # High outbound data volume
        base[8] += 1.7 * intensity   # Unknown destination geolocation
        return base

    def _attack_edge_features(self, intensity: float) -> np.ndarray:
        base = self.np_rng.normal(loc=0.0, scale=0.15, size=8).astype(np.float32)
        base[0] += 1.5 * intensity   # Unusual timing
        base[3] += 1.8 * intensity   # Anomalous payload size
        return base
