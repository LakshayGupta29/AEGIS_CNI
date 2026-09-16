'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { SimulationTick, GraphNode, GraphLink, MitreStage, Alert, SOARAction, AuditEntry } from '@/lib/types';

const MOCK_INTERVAL_MS = 500;

export function useWebSocket(url: string) {
  const [data, setData] = useState<SimulationTick | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Standalone Mode State
  const [isStandaloneMode, setIsStandaloneMode] = useState(false);
  const mockIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mockStateRef = useRef({
    tick: 0,
    nodes: [] as GraphNode[],
    links: [] as GraphLink[],
    currentStage: 0,
    anomalyScore: 0.1,
    blastRadius: 0
  });

  const generateMockTick = () => {
    const s = mockStateRef.current;
    s.tick += 1;
    
    // Simulate progression
    if (s.tick % 15 === 0 && s.currentStage < 7) {
      s.currentStage += 1;
    }

    // Add nodes
    const isAnomaly = s.currentStage > 2;
    const newNode: GraphNode = {
      id: `node_${s.tick}`,
      label: isAnomaly ? `cmd_${s.tick}.exe` : `sys_${s.tick}.dll`,
      type: isAnomaly ? 'alert' : 'process',
      anomalyScore: isAnomaly ? Math.random() * 0.5 + 0.5 : Math.random() * 0.1,
      timestamp: Date.now()
    };
    s.nodes.push(newNode);
    
    if (s.nodes.length > 1) {
      s.links.push({
        source: s.nodes[s.nodes.length - 2].id,
        target: newNode.id,
        type: 'spawn',
        timestamp: Date.now()
      });
    }

    s.anomalyScore = isAnomaly ? Math.min(1.0, s.anomalyScore + 0.05) : 0.1;
    s.blastRadius = s.currentStage * 12 + (Math.random() * 5);

    const stages: MitreStage[] = [
      { id: 0, name: 'Normal', probability: s.currentStage === 0 ? 1 : 0, color: '#22c55e', active: s.currentStage >= 0 },
      { id: 1, name: 'Initial Access', probability: s.currentStage === 1 ? 0.9 : 0, color: '#eab308', active: s.currentStage >= 1 },
      { id: 2, name: 'Execution', probability: s.currentStage === 2 ? 0.8 : 0, color: '#f97316', active: s.currentStage >= 2 },
      { id: 3, name: 'Persistence', probability: s.currentStage === 3 ? 0.85 : 0, color: '#f97316', active: s.currentStage >= 3 },
      { id: 4, name: 'Evasion', probability: s.currentStage === 4 ? 0.75 : 0, color: '#ef4444', active: s.currentStage >= 4 },
      { id: 5, name: 'Discovery', probability: s.currentStage === 5 ? 0.9 : 0, color: '#ef4444', active: s.currentStage >= 5 },
      { id: 6, name: 'Lateral Movement', probability: s.currentStage === 6 ? 0.95 : 0, color: '#dc2626', active: s.currentStage >= 6 },
      { id: 7, name: 'Impact', probability: s.currentStage === 7 ? 0.99 : 0, color: '#991b1b', active: s.currentStage >= 7 },
    ];

    const alerts: Alert[] = isAnomaly && s.tick % 5 === 0 ? [{
      id: `alert_${s.tick}`,
      timestamp: new Date().toISOString(),
      severity: s.currentStage > 5 ? 'critical' : 'high',
      message: `Suspicious behavior detected in ${newNode.label}`,
      nodeId: newNode.id,
      mitreStage: stages[s.currentStage].name,
      anomalyScore: s.anomalyScore
    }] : [];

    const soarActions: SOARAction[] = s.blastRadius > 70 && s.tick % 10 === 0 ? [{
      id: `soar_${s.tick}`,
      timestamp: new Date().toISOString(),
      type: 'isolate_endpoint',
      target: newNode.id,
      status: 'executed',
      confidence: 0.95,
      description: 'Autonomous Isolation Triggered by high Blast Radius'
    }] : [];

    const auditLog: AuditEntry[] = soarActions.map(a => ({
      id: `audit_${s.tick}`,
      timestamp: a.timestamp,
      action: a.description,
      actor: 'AEGIS_AI',
      details: 'Automated containment',
      hash: '0x' + Math.random().toString(16).slice(2)
    }));

    const tickData: SimulationTick = {
      timestamp: new Date().toISOString(),
      tick: s.tick,
      totalTicks: 150,
      graph: { nodes: [...s.nodes], links: [...s.links] },
      anomalyScore: s.anomalyScore,
      blastRadiusScore: s.blastRadius,
      mitreStages: stages,
      currentStage: s.currentStage,
      alerts,
      soarActions,
      auditLog,
      metrics: {
        nodesAnalyzed: s.nodes.length,
        edgesAnalyzed: s.links.length,
        meanReconError: s.anomalyScore,
        stageConfidence: 0.92
      },
      complete: s.tick >= 150
    };

    setData(tickData);
  };

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setIsStandaloneMode(false);
        setError(null);
        console.log('[Aegis-CNI] WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const tick: SimulationTick = JSON.parse(event.data);
          setData(tick);
        } catch (e) {
          console.error('[Aegis-CNI] Failed to parse message:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('[Aegis-CNI] WebSocket disconnected. Falling back to Standalone Mode for Demo.');
        setIsStandaloneMode(true);
      };

      ws.onerror = () => {
        setIsStandaloneMode(true);
      };
    } catch {
      setIsStandaloneMode(true);
    }
  }, [url]);

  const sendMessage = useCallback((message: string) => {
    const parsed = JSON.parse(message);
    if (parsed.action === 'start_simulation' || parsed.action === 'start_live') {
      if (isStandaloneMode) {
        // Start Mock Simulation
        mockStateRef.current = { tick: 0, nodes: [], links: [], currentStage: 0, anomalyScore: 0.1, blastRadius: 0 };
        if (mockIntervalRef.current) clearInterval(mockIntervalRef.current);
        mockIntervalRef.current = setInterval(generateMockTick, MOCK_INTERVAL_MS);
      } else if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(message);
      }
    }
  }, [isStandaloneMode]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (mockIntervalRef.current) clearInterval(mockIntervalRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // If in standalone mode, pretend we are connected so the UI allows "Launch Simulation"
  return { data, isConnected: isConnected || isStandaloneMode, error, sendMessage };
}
