'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useWebSocket } from '@/hooks/useWebSocket';
import { MitreStage, Alert, SOARAction, AuditEntry } from '@/lib/types';

const AttackGraph = dynamic(() => import('@/components/AttackGraph'), { ssr: false });

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws/simulation';

const DEFAULT_STAGES: MitreStage[] = [
  { id: 0, name: 'Normal', probability: 1.0, color: '#22c55e', active: true },
  { id: 1, name: 'Initial Access', probability: 0, color: '#eab308', active: false },
  { id: 2, name: 'Execution', probability: 0, color: '#f97316', active: false },
  { id: 3, name: 'Persistence', probability: 0, color: '#f97316', active: false },
  { id: 4, name: 'Evasion', probability: 0, color: '#ef4444', active: false },
  { id: 5, name: 'Discovery', probability: 0, color: '#ef4444', active: false },
  { id: 6, name: 'Lateral Movement', probability: 0, color: '#dc2626', active: false },
  { id: 7, name: 'Impact', probability: 0, color: '#991b1b', active: false },
];

export default function Dashboard() {
  const { data, isConnected, sendMessage } = useWebSocket(WS_URL);
  const [allAlerts, setAllAlerts] = useState<Alert[]>([]);
  const [allActions, setAllActions] = useState<SOARAction[]>([]);
  const [allAudit, setAllAudit] = useState<AuditEntry[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);

  useEffect(() => {
    if (data) {
      if (data.alerts?.length) {
        setAllAlerts(prev => {
          const newAlerts = data.alerts.filter(a => !prev.some(p => p.id === a.id));
          return [...newAlerts, ...prev].slice(0, 50);
        });
      }
      if (data.soarActions?.length) {
        setAllActions(prev => {
          const newActions = data.soarActions.filter(a => !prev.some(p => p.id === a.id));
          return [...newActions, ...prev].slice(0, 30);
        });
      }
      if (data.auditLog?.length) {
        setAllAudit(prev => {
          const newEntries = data.auditLog.filter(a => !prev.some(p => p.id === a.id));
          return [...newEntries, ...prev].slice(0, 50);
        });
      }
      if (data.complete) {
        setIsSimulating(false);
      }
    }
  }, [data]);

  const stages = data?.mitreStages || DEFAULT_STAGES;
  const blastRadius = data?.blastRadiusScore || 0;
  const anomalyScore = data?.anomalyScore || 0;
  const currentStage = data?.currentStage || 0;
  const progress = data ? Math.round((data.tick / data.totalTicks) * 100) : 0;

  const startSimulation = () => {
    sendMessage(JSON.stringify({ action: 'start_simulation' }));
    setIsSimulating(true);
    setAllAlerts([]);
    setAllActions([]);
    setAllAudit([]);
  };

  const startLiveMode = () => {
    sendMessage(JSON.stringify({ action: 'start_live' }));
    setIsSimulating(true);
    setAllAlerts([]);
    setAllActions([]);
    setAllAudit([]);
  };

  const blastRadiusColor = blastRadius > 80 ? '#ff3b5c' : blastRadius > 50 ? '#ff8c00' : blastRadius > 25 ? '#ffd700' : '#00ff88';

  return (
    <div className="min-h-screen grid-overlay">
      <div className="scan-line" />

      {/* ── Header ── */}
      <header className="border-b border-[rgba(0,240,255,0.08)] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide">
              <span className="glow-text" style={{ color: 'var(--accent-cyan)' }}>AEGIS</span>
              <span className="text-gray-400">-CNI</span>
            </h1>
            <p className="text-[10px] text-gray-500 tracking-widest uppercase">Cyber Resilience Intelligence Platform</p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
            <span className="text-gray-400">{isConnected ? 'SYSTEM ONLINE' : 'DISCONNECTED'}</span>
          </div>

          <button
            id="launch-live-btn"
            onClick={startLiveMode}
            disabled={!isConnected}
            className="px-4 py-1.5 text-xs font-medium rounded-md bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-500 hover:to-orange-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-300 tracking-wider uppercase cursor-pointer"
          >
            {isSimulating ? '🔴 LIVE RESTART' : '🔴 LIVE EDR MODE'}
          </button>

          <button
            id="launch-simulation-btn"
            onClick={startSimulation}
            disabled={!isConnected}
            className="px-4 py-1.5 text-xs font-medium rounded-md bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-300 tracking-wider uppercase cursor-pointer"
          >
            {isSimulating ? '↻ DEMO RESTART' : '▶ LAUNCH DEMO'}
          </button>

          {isSimulating && data && (
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
              <span className="data-mono text-gray-400">{progress}%</span>
            </div>
          )}
        </div>
      </header>

      {/* ── Main Grid ── */}
      <main className="p-4 grid grid-cols-12 gap-3" style={{ height: 'calc(100vh - 60px)' }}>

        {/* Row 1: MITRE ATT&CK Kill Chain */}
        <div className="col-span-12 glass-card p-3">
          <h2 className="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-2">MITRE ATT&CK Kill Chain Progression</h2>
          <div className="flex gap-1">
            {stages.map((stage, i) => (
              <div
                key={stage.id}
                className="flex-1 rounded-md p-2 transition-all duration-500 relative overflow-hidden"
                style={{
                  borderWidth: '1px',
                  borderStyle: 'solid',
                  borderColor: i <= currentStage && data ? `${stage.color}80` : '#1e293b',
                  background: i <= currentStage && data
                    ? `linear-gradient(135deg, ${stage.color}15, ${stage.color}05)`
                    : 'rgba(15, 23, 42, 0.5)',
                  boxShadow: i === currentStage && data ? `0 0 15px ${stage.color}30` : undefined,
                }}
              >
                <div className="text-[9px] data-mono text-gray-500">{stage.id === 0 ? '—' : `TA010${stage.id + 1}`}</div>
                <div className="text-[11px] font-medium mt-0.5" style={{ color: i <= currentStage && data ? stage.color : '#475569' }}>
                  {stage.name}
                </div>
                {data && (
                  <div className="mt-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${stage.probability * 100}%`, background: stage.color }}
                    />
                  </div>
                )}
                <div className="text-[9px] data-mono mt-0.5" style={{ color: stage.color }}>
                  {data ? `${(stage.probability * 100).toFixed(1)}%` : '—'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Row 2 Left: Provenance Graph */}
        <div className="col-span-7 glass-card p-3 flex flex-col" style={{ height: 'calc(100vh - 280px)' }}>
          <h2 className="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-2">
            Live Provenance Graph
            {data && <span className="ml-2 text-cyan-400">{data.graph.nodes.length} nodes · {data.graph.links.length} edges</span>}
          </h2>
          <div className="flex-1 min-h-0 rounded-lg overflow-hidden" style={{ background: '#050810' }}>
            {data?.graph && data.graph.nodes.length > 0 ? (
              <AttackGraph nodes={data.graph.nodes} links={data.graph.links} />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-600 text-sm">
                <div className="text-center">
                  <div className="text-4xl mb-2">⬡</div>
                  <div>Launch simulation to visualize provenance graph</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Row 2 Right: Metrics + Alerts */}
        <div className="col-span-5 flex flex-col gap-3" style={{ height: 'calc(100vh - 280px)' }}>
          {/* Blast Radius + Anomaly Score */}
          <div className="grid grid-cols-2 gap-3">
            <div id="blast-radius-card" className="glass-card p-4 flex flex-col items-center" style={{ boxShadow: blastRadius > 70 ? `0 0 30px ${blastRadiusColor}20` : undefined }}>
              <h3 className="text-[9px] uppercase tracking-[0.2em] text-gray-500 mb-2">Blast Radius</h3>
              <div className="text-4xl font-bold data-mono transition-colors duration-500" style={{ color: blastRadiusColor }}>
                {blastRadius.toFixed(0)}
              </div>
              <div className="text-[10px] text-gray-500 mt-1">/ 100</div>
              <div className="w-full mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${blastRadius}%`, background: blastRadiusColor }} />
              </div>
            </div>

            <div id="anomaly-score-card" className="glass-card p-4 flex flex-col items-center" style={{ boxShadow: anomalyScore > 0.7 ? 'var(--glow-red)' : undefined }}>
              <h3 className="text-[9px] uppercase tracking-[0.2em] text-gray-500 mb-2">Anomaly Score</h3>
              <div className="text-4xl font-bold data-mono transition-colors duration-500" style={{ color: anomalyScore > 0.7 ? '#ff3b5c' : anomalyScore > 0.4 ? '#ff8c00' : '#00ff88' }}>
                {anomalyScore.toFixed(2)}
              </div>
              <div className="text-[10px] text-gray-500 mt-1">L_recon</div>
              <div className="w-full mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${anomalyScore * 100}%`, background: anomalyScore > 0.7 ? '#ff3b5c' : anomalyScore > 0.4 ? '#ff8c00' : '#00ff88' }} />
              </div>
            </div>
          </div>

          {/* Metrics */}
          {data?.metrics && (
            <div className="glass-card p-3 grid grid-cols-4 gap-2">
              {[
                { label: 'Nodes', value: data.metrics.nodesAnalyzed },
                { label: 'Edges', value: data.metrics.edgesAnalyzed },
                { label: 'Recon Err', value: data.metrics.meanReconError.toFixed(3) },
                { label: 'Stage Conf', value: `${(data.metrics.stageConfidence * 100).toFixed(0)}%` },
              ].map((m, i) => (
                <div key={i} className="text-center">
                  <div className="text-lg font-bold data-mono text-cyan-400">{m.value}</div>
                  <div className="text-[9px] text-gray-500 uppercase tracking-wider">{m.label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Alert Feed */}
          <div className="glass-card p-3 flex-1 min-h-0 overflow-hidden flex flex-col">
            <h3 className="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-2">
              Anomaly Alerts
              {allAlerts.length > 0 && <span className="ml-2 text-red-400">({allAlerts.length})</span>}
            </h3>
            <div className="flex-1 overflow-y-auto space-y-1.5">
              {allAlerts.length === 0 ? (
                <div className="text-gray-600 text-xs text-center py-4">No anomalies detected</div>
              ) : (
                allAlerts.map(alert => (
                  <div
                    key={alert.id}
                    className={`p-2 rounded-md text-xs border fade-in ${
                      alert.severity === 'critical' ? 'severity-critical alert-pulse' :
                      alert.severity === 'high' ? 'severity-high' :
                      alert.severity === 'medium' ? 'severity-medium' : 'severity-low'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <span className="font-medium">{alert.message}</span>
                      <span className="data-mono text-[9px] ml-2 shrink-0">{alert.timestamp}</span>
                    </div>
                    <div className="flex gap-2 mt-1">
                      <span className="mitre-chip" style={{ background: 'rgba(0,240,255,0.1)', color: '#00f0ff' }}>
                        {alert.mitreStage}
                      </span>
                      <span className="data-mono text-gray-500">Score: {alert.anomalyScore.toFixed(2)}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Row 3: SOAR Actions */}
        <div className="col-span-6 glass-card p-3 overflow-hidden flex flex-col" style={{ maxHeight: '200px' }}>
          <h3 className="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-2">
            SOAR Autonomous Actions
            {allActions.length > 0 && <span className="ml-2 text-green-400">({allActions.length})</span>}
          </h3>
          <div className="flex-1 overflow-y-auto space-y-1">
            {allActions.length === 0 ? (
              <div className="text-gray-600 text-xs text-center py-4">No containment actions triggered</div>
            ) : (
              allActions.map(action => (
                <div key={action.id} className="flex items-center gap-2 p-2 rounded-md bg-[rgba(0,255,136,0.05)] border border-[rgba(0,255,136,0.15)] text-xs fade-in">
                  <span className="text-green-400 text-sm">
                    {action.type === 'isolate_endpoint' ? '🔒' :
                     action.type === 'block_ip' ? '🚫' :
                     action.type === 'revoke_credential' ? '🔑' :
                     action.type === 'snapshot_vm' ? '📸' : '🛡️'}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-green-300 truncate">{action.description}</div>
                    <div className="data-mono text-gray-500 text-[9px]">{action.target} · Confidence: {(action.confidence * 100).toFixed(0)}%</div>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[9px] shrink-0 ${
                    action.status === 'executed' ? 'bg-green-900/30 text-green-400' :
                    action.status === 'pending_approval' ? 'bg-yellow-900/30 text-yellow-400' :
                    'bg-red-900/30 text-red-400'
                  }`}>
                    {action.status.toUpperCase()}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Row 3: Audit Log */}
        <div className="col-span-6 glass-card p-3 overflow-hidden flex flex-col" style={{ maxHeight: '200px' }}>
          <h3 className="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-2">Immutable Audit Trail</h3>
          <div className="flex-1 overflow-y-auto space-y-1">
            {allAudit.length === 0 ? (
              <div className="text-gray-600 text-xs text-center py-4">No actions recorded</div>
            ) : (
              allAudit.map(entry => (
                <div key={entry.id} className="flex items-center gap-2 p-1.5 text-xs border-b border-gray-800/50 fade-in">
                  <span className={`text-[9px] px-1.5 py-0.5 rounded data-mono shrink-0 ${
                    entry.actor === 'AEGIS_AI' ? 'bg-cyan-900/30 text-cyan-400' : 'bg-purple-900/30 text-purple-400'
                  }`}>
                    {entry.actor}
                  </span>
                  <span className="text-gray-300 flex-1 truncate">{entry.action}</span>
                  <span className="data-mono text-gray-600 text-[8px] shrink-0" title={entry.hash}>
                    {entry.hash.slice(0, 12)}...
                  </span>
                  <span className="data-mono text-gray-500 text-[9px] shrink-0">{entry.timestamp}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
