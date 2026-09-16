# AEGIS-CNI: AI-Powered Cyber Resilience Intelligence Platform

## 1. Project Overview
**Project Name:** AEGIS-CNI
**Theme:** Cyber Resilience for Critical National Infrastructure
**Objective:** To build an autonomous Endpoint Detection and Response (EDR) and Security Orchestration, Automation, and Response (SOAR) platform capable of detecting and containing Advanced Persistent Threats (APTs) in real-time.

AEGIS-CNI addresses the critical vulnerability of national infrastructure (power grids, hospitals, transport) to cyberattacks. Traditional signature-based antivirus solutions fail against novel, "Living off the Land" (LotL) tactics. AEGIS-CNI solves this by utilizing advanced AI models to analyze the **behavior** of processes rather than their signatures, identifying malicious intent and autonomously neutralizing the threat in milliseconds.

## 2. Problem Statement
**Problem Statement Addressed:** Autonomous Cyber Resilience and Threat Containment for Critical Infrastructure using AI/ML.

Modern cyberattacks on critical infrastructure (such as the AIIMS Delhi ransomware attack or the Colonial Pipeline attack) move faster than human security operators can respond. By the time a security analyst reviews an alert, the ransomware has often already encrypted critical databases. There is an urgent need for systems that can not only detect novel zero-day threats but autonomously contain them to minimize the "blast radius" without human intervention.

## 3. Solution & Architecture
AEGIS-CNI acts as a complete AI-driven security pipeline:

1. **Live Telemetry Ingestion (The Eyes):** A lightweight endpoint agent continuously monitors operating system processes, parent-child process spawns, and network connections.
2. **Provenance Graph Construction (The Context):** The raw telemetry is converted into W3C PROV-DM standard Directed Acyclic Graphs (DAGs). This provides deep context, linking a seemingly benign process to its true origin.
3. **Graph Attention Autoencoder (GAE) (The Anomaly Detector):** Trained on the massive DARPA Transparent Computing (TC) dataset, the GAE learns the baseline "normal" behavior of the network. When an anomalous graph shape is detected (e.g., Process Hollowing), the GAE outputs a high Reconstruction Error.
4. **LSTM Stage Predictor (The Threat Analyst):** A Long Short-Term Memory (LSTM) network evaluates the sequence of anomalies over time to accurately map the attack to the MITRE ATT&CK Kill Chain framework.
5. **Autonomous SOAR Orchestrator (The Hands):** Upon receiving high-confidence alerts from the AI, the SOAR engine executes immediate, real-time containment actions (e.g., physically terminating malicious processes like `notepad.exe` when hijacked) to protect the host.

## 4. Technical Stack
- **Backend Core:** Python, FastAPI, Uvicorn (Asynchronous API and WebSockets)
- **AI / Machine Learning:** PyTorch, PyTorch Geometric, Scikit-learn, NumPy, Pandas
- **Live Endpoint Agent:** `psutil` (for live cross-platform process monitoring and termination)
- **Frontend Dashboard:** Next.js, React, TypeScript, Tailwind CSS, D3.js (for live Graph visualization)
- **Cloud Training Environment:** Google Colab / Kaggle (used for heavy GPU training of the PyTorch models)

## 5. Innovation and Uniqueness
- **Behavioral over Signature-Based:** AEGIS-CNI ignores binary names. A hacker disguising ransomware as `notepad.exe` will bypass traditional AV, but AEGIS-CNI flags the abnormal graph structure and terminates it.
- **Live-Fire EDR Prototype:** Instead of a simple dashboard displaying mock data, the project features a functioning live agent that monitors the host's actual memory processes and physically executes termination commands via the SOAR engine.
- **Microsecond Response Time:** By automating containment, the Mean Time to Respond (MTTR) is reduced from hours/days to milliseconds, preventing lateral movement across the network.

## 6. Future Scope
If scaled, AEGIS-CNI can be deployed via lightweight kernel-level drivers (like eBPF on Linux or Sysmon on Windows) across thousands of nodes in a smart city or power grid. The federated AI models can share threat intelligence globally without exposing sensitive raw data, creating a collective defense network for National Infrastructure.
