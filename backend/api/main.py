"""
Aegis-CNI: FastAPI Backend Server

Provides WebSocket endpoint for real-time simulation streaming to the
Next.js dashboard, plus REST endpoints for system status and model info.
"""

import asyncio
import json
import sys
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inference_engine import InferenceEngine


# ─── Global State ────────────────────────────────────────────────────────────

engine: InferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the inference engine on startup."""
    global engine
    print("=" * 60)
    print("  AEGIS-CNI | Cyber Resilience Intelligence Platform")
    print("=" * 60)
    engine = InferenceEngine()
    engine.initialize(train_epochs=30)
    print("\n[Server] Ready. Connect frontend to ws://localhost:8000/ws/simulation")
    print("=" * 60)
    yield
    print("[Server] Shutting down...")


app = FastAPI(
    title="Aegis-CNI API",
    description="AI-Powered Cyber Resilience Platform for Critical National Infrastructure",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Aegis-CNI",
        "version": "1.0.0",
        "status": "online",
        "description": "AI-Powered Cyber Resilience Platform for Critical National Infrastructure",
    }


@app.get("/api/status")
async def get_status():
    return {
        "engine_initialized": engine.is_initialized if engine else False,
        "gae_trained": engine.gae.is_trained if engine else False,
        "lstm_trained": engine.stage_predictor.is_trained if engine else False,
        "baseline_recon_error": engine.gae.baseline_mean if engine else None,
    }


@app.get("/api/mitre-stages")
async def get_mitre_stages():
    """Return MITRE ATT&CK stage definitions."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mitre_attack_ics.json"
    )
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    return {"error": "MITRE data not found"}


@app.get("/api/playbook-coverage")
async def get_playbook_coverage():
    """Return SOAR playbook execution metrics."""
    if engine:
        return engine.soar.get_playbook_coverage()
    return {"error": "Engine not initialized"}


# ─── WebSocket Simulation Endpoint ───────────────────────────────────────────

@app.websocket("/ws/simulation")
async def simulation_websocket(websocket: WebSocket):
    """
    Main WebSocket endpoint for the live simulation.

    Protocol:
        Client sends: {"action": "start_simulation"}
        Server streams: SimulationTick JSON objects every 500ms
    """
    await websocket.accept()
    print(f"[WS] Client connected: {websocket.client}")

    try:
        while True:
            # Wait for client command
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("action") == "start_simulation":
                print("[WS] Starting simulation...")
                engine.enable_live_mode(False)
                engine.reset_simulation()
                
            elif data.get("action") == "start_live":
                print("[WS] Starting LIVE PSUTIL mode...")
                engine.enable_live_mode(True)
                engine.reset_simulation()

            if data.get("action") in ["start_simulation", "start_live"]:
                # Stream simulation ticks
                tick_count = 0
                start_time = time.time()
                
                # Determine loop bound based on mode
                is_live = engine.use_live_mode
                max_ticks = engine.live_simulator.config.total_ticks if is_live else engine.simulator.config.total_ticks
                
                current_tick_obj = engine.live_simulator if is_live else engine.simulator

                while current_tick_obj.current_tick < max_ticks:
                    # Process one tick through the full pipeline
                    tick_data = engine.process_tick()

                    # Send to client
                    await websocket.send_text(json.dumps(tick_data))

                    tick_count += 1

                    # Pace the simulation (500ms for demo, 2000ms for live telemetry)
                    sleep_time = 2.0 if is_live else 0.5
                    await asyncio.sleep(sleep_time)

                # Send completion message
                final = engine.process_tick()
                final["complete"] = True
                await websocket.send_text(json.dumps(final))

                elapsed = time.time() - start_time
                print(f"[WS] Simulation complete: {tick_count} ticks in {elapsed:.1f}s")

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
