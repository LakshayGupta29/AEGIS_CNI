# Aegis-CNI: Cyber Resilience Intelligence Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-15.1-black.svg)](https://nextjs.org/)

**Aegis-CNI** is an AI-powered Cyber Resilience platform designed for Critical National Infrastructure (CNI). It autonomously detects advanced persistent threats (APTs) using **Spatio-Temporal Graph Neural Networks (ST-GNNs)** without relying on legacy signature-based detection, and orchestrates containment actions in milliseconds using a built-in SOAR engine.

This project was built for the **ET AI Hackathon 2.0 (Problem Statement 7: Cyber Resilience for CNI)**.

---

## 🚀 Live Demo (Vercel Standalone Mode)

For the online hackathon submission, the frontend is deployed in a **Standalone Mock Mode**. It mathematically simulates the provenance graph generation directly in the browser to guarantee a flawless viewing experience without requiring the Python backend to be hosted online.

**View the Live Demo:** https://aegis-cni.vercel.app/

---

## 🧠 Core Architecture

Traditional SOCs rely on static log analysis and known signatures, resulting in an average Mean Time to Detect (MTTD) of 21 days for APTs. Aegis-CNI shifts the paradigm from log analysis to **behavioral graph analysis**.

### 1. W3C PROV-DM Telemetry Ingestion
Instead of isolated logs, the system builds real-time **Provenance Graphs** (modeling processes, files, sockets, and registry keys as nodes). This provides full causal context of system interactions.

### 2. Spatio-Temporal Graph Neural Network (ST-GNN)
- **Spatial Anomaly Detection (GAE):** A Graph Attention Autoencoder (GAE) is trained exclusively on benign network telemetry to establish a structural baseline. When an APT executes novel behavior (e.g., Lateral Movement), the GAE's reconstruction error ($L_{recon}$) spikes, flagging the anomaly without needing a malware signature.
- **Temporal Prediction (LSTM):** A Long Short-Term Memory network evaluates the sequence of these spatial anomalies over time. Using a custom Temporal Contrastive Loss function, it maps the attacker's progression against the **MITRE ATT&CK** framework, predicting their next move.

### 3. Autonomous SOAR Engine
The Security Orchestration, Automation, and Response (SOAR) layer calculates a dynamic **Blast Radius Risk Score**. When high-confidence thresholds are crossed, it executes autonomous playbooks (e.g., isolating endpoints, blocking IPs) and logs a cryptographically hashed immutable audit trail.

---

## 🛠️ Local Installation & Development

To run the full PyTorch ML backend and the Next.js frontend locally:

### Prerequisites
- Python 3.12+
- Node.js 18+

### 1. Start the ML Backend
```bash
cd backend
pip install -r requirements.txt
python api/main.py
```
*Note: The backend runs on `http://localhost:8080`. On startup, it performs micro-training on the GAE and LSTM models using a synthetic data generator to bypass the need for downloading 50GB benchmark datasets during the hackathon.*

### 2. Start the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
*The dashboard will be available at `http://localhost:3000`.*

---

## ☁️ Cloud Training Pipeline (For Real-World Data)
While the local backend uses a synthetic generator for the 48-hour hackathon demo, the repository includes a `colab_training_pipeline.ipynb` notebook. 

This notebook is designed to be executed on **Google Colab**. It ingests the massive **DARPA TC (Transparent Computing)** benchmark dataset, trains the PyTorch models using Cloud GPUs (T4), and exports the lightweight `.pt` weight files for Edge Inference on the local backend.

---

## 📜 MITRE ATT&CK Mapping
Our LSTM specifically predicts progression across the following stages:
1. Initial Access (TA0001)
2. Execution (TA0002)
3. Persistence (TA0003)
4. Evasion (TA0005)
5. Discovery (TA0007)
6. Lateral Movement (TA0008)
7. Impact (TA0040)

## ⚖️ License
MIT License
