'use client';

import { useRef, useCallback, useEffect, useState } from 'react';
import ForceGraph2D, { ForceGraphMethods } from 'react-force-graph-2d';
import { GraphNode, GraphLink } from '@/lib/types';

interface AttackGraphProps {
  nodes: GraphNode[];
  links: GraphLink[];
}

const NODE_COLORS: Record<string, string> = {
  process: '#60a5fa',
  file: '#34d399',
  socket: '#f472b6',
  user: '#fbbf24',
  registry: '#a78bfa',
  alert: '#ff3b5c',
};

export default function AttackGraph({ nodes, links }: AttackGraphProps) {
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const graphData = {
    nodes: nodes.map(n => ({
      ...n,
      color: NODE_COLORS[n.type] || '#64748b',
      val: n.anomalyScore > 0.7 ? 6 : n.anomalyScore > 0.3 ? 4 : 2,
    })),
    links: links.map(l => ({
      ...l,
      color: 'rgba(100, 116, 139, 0.2)',
    })),
  };

  const paintNode = useCallback((node: Record<string, unknown>, ctx: CanvasRenderingContext2D) => {
    const size = (node.val as number) || 3;
    const color = (node.color as string) || '#64748b';
    const x = node.x as number;
    const y = node.y as number;
    const anomalyScore = (node.anomalyScore as number) || 0;
    const nodeType = node.type as string;
    const label = node.label as string;

    if (x === undefined || y === undefined) return;

    // Glow effect for anomalous nodes
    if (anomalyScore > 0.5) {
      ctx.beginPath();
      ctx.arc(x, y, size + 4, 0, 2 * Math.PI);
      ctx.fillStyle = `${color}30`;
      ctx.fill();
    }

    // Node shape based on type
    ctx.beginPath();
    if (nodeType === 'alert') {
      // Star shape for alerts
      const spikes = 5;
      const outerRadius = size;
      const innerRadius = size / 2;
      for (let i = 0; i < spikes * 2; i++) {
        const radius = i % 2 === 0 ? outerRadius : innerRadius;
        const angle = (Math.PI * i) / spikes - Math.PI / 2;
        const px = x + Math.cos(angle) * radius;
        const py = y + Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
    } else if (nodeType === 'file') {
      ctx.rect(x - size / 2, y - size / 2, size, size);
    } else {
      ctx.arc(x, y, size, 0, 2 * Math.PI);
    }

    ctx.fillStyle = color;
    ctx.fill();

    // Label for anomalous or large nodes
    if (anomalyScore > 0.5 || size > 3) {
      ctx.font = '3px Inter, sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.textAlign = 'center';
      ctx.fillText(label || '', x, y + size + 5);
    }
  }, []);

  return (
    <div ref={containerRef} className="w-full h-full">
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        backgroundColor="#050810"
        nodeCanvasObject={paintNode}
        linkColor={() => 'rgba(100, 116, 139, 0.15)'}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
        linkWidth={0.5}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        cooldownTicks={100}
        nodeLabel={(node: Record<string, unknown>) =>
          `${node.label} (${node.type}) — Anomaly: ${((node.anomalyScore as number) || 0).toFixed(2)}`
        }
      />
    </div>
  );
}
