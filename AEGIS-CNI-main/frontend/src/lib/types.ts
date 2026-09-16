export interface GraphNode {
  id: string;
  label: string;
  type: 'process' | 'file' | 'socket' | 'user' | 'registry' | 'alert';
  anomalyScore: number;
  timestamp: number;
  metadata?: Record<string, string>;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  timestamp: number;
}

export interface MitreStage {
  id: number;
  name: string;
  probability: number;
  color: string;
  active: boolean;
}

export interface Alert {
  id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  nodeId: string;
  mitreStage: string;
  anomalyScore: number;
}

export interface SOARAction {
  id: string;
  timestamp: string;
  type: 'isolate_endpoint' | 'block_ip' | 'revoke_credential' | 'snapshot_vm' | 'acl_update';
  target: string;
  status: 'executed' | 'pending_approval' | 'failed';
  confidence: number;
  description: string;
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor: 'AEGIS_AI' | 'HUMAN_OPERATOR';
  details: string;
  hash: string;
}

export interface SimulationTick {
  timestamp: string;
  tick: number;
  totalTicks: number;
  graph: {
    nodes: GraphNode[];
    links: GraphLink[];
  };
  anomalyScore: number;
  blastRadiusScore: number;
  mitreStages: MitreStage[];
  currentStage: number;
  alerts: Alert[];
  soarActions: SOARAction[];
  auditLog: AuditEntry[];
  metrics: {
    nodesAnalyzed: number;
    edgesAnalyzed: number;
    meanReconError: number;
    stageConfidence: number;
  };
  complete?: boolean;
  summary?: {
    totalAlerts: number;
    totalSOARActions: number;
    playbookCoverage: Record<string, number>;
    mttd_improvement: string;
    mttr_improvement: string;
  };
}
