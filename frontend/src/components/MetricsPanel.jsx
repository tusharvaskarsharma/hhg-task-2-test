import React from 'react';
import { Database, Clock, Server } from 'lucide-react';

const MetricsPanel = ({ latency, cache }) => {
  if (!latency) return null;

  return (
    <div className="w-full mt-6">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        
        {/* TOTAL CARD */}
        <div className="flex flex-col p-4 glass-panel hover:bg-surfaceHover transition-colors border-border/50 border-l-4 border-l-primary relative">
          <div className="text-[10px] uppercase font-bold tracking-wider text-textMuted mb-1">TOTAL LATENCY</div>
          <div className="text-xs text-textMuted/70 mb-3">Total request duration (STT + RAG + SLM)</div>
          <div className="flex items-baseline space-x-1 mt-auto">
            <span className="text-2xl font-mono font-bold text-primary">
              {latency.total_ms !== undefined ? latency.total_ms.toFixed(2) : '--'}
            </span>
            <span className="text-sm text-textMuted">ms</span>
          </div>
          <div className="text-[10px] text-textMuted/50 mt-1">Target: not configured</div>
        </div>

        {/* PARTIAL CARD */}
        <div className="flex flex-col p-4 glass-panel hover:bg-surfaceHover transition-colors border-border/50 border-l-4 border-l-secondary relative">
          <div className="text-[10px] uppercase font-bold tracking-wider text-textMuted mb-1">PARTIAL LATENCY</div>
          <div className="text-xs text-textMuted/70 mb-3">RAG + SLM (Excludes STT)</div>
          <div className="flex items-baseline space-x-1 mt-auto">
            <span className="text-2xl font-mono font-bold text-text">
              {latency.partial_ms !== undefined ? latency.partial_ms.toFixed(2) : '--'}
            </span>
            <span className="text-sm text-textMuted">ms</span>
          </div>
          <div className="text-[10px] text-textMuted/50 mt-1">Target: not configured</div>
        </div>

        {/* RAG ONLY CARD */}
        <div className="flex flex-col p-4 glass-panel hover:bg-surfaceHover transition-colors border-border/50 border-l-4 border-l-success relative">
          <div className="text-[10px] uppercase font-bold tracking-wider text-textMuted mb-1">RAG ONLY</div>
          <div className="text-xs text-textMuted/70 mb-3">RAG without SLM/STT</div>
          <div className="flex items-baseline space-x-1 mt-auto">
            <span className="text-2xl font-mono font-bold text-success">
              {latency.rag_only_ms !== undefined ? latency.rag_only_ms.toFixed(2) : '--'}
            </span>
            <span className="text-sm text-textMuted">ms</span>
          </div>
          <div className="text-[10px] text-textMuted/50 mt-1">Target &le; 50 ms</div>
          
          {cache?.hit !== undefined && (
             <div className="absolute top-4 right-4 flex items-center space-x-1">
               <Server className="w-3 h-3 text-textMuted" />
               <span className={`text-[9px] font-bold ${cache.hit ? 'text-success' : 'text-textMuted'}`}>
                 {cache.hit ? 'CACHE HIT' : 'MISS'}
               </span>
             </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default MetricsPanel;
