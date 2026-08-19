import React from 'react';
import { Activity, ServerOff } from 'lucide-react';

const StatusIndicator = ({ status }) => {
  if (!status) {
    return (
      <div className="flex items-center space-x-2 glass-panel px-3 py-1.5 text-xs text-textMuted">
        <ServerOff className="w-3.5 h-3.5" />
        <span>Connecting...</span>
      </div>
    );
  }

  const isReady = status.status === "ready";
  const hasSaaras = status.saaras?.enabled;
  const hasSLM = status.slm?.enabled;

  return (
    <div className="flex items-center space-x-3">
      <div className={`glass-panel px-3 py-1.5 text-xs flex items-center space-x-2 ${isReady ? 'text-success' : 'text-error'}`}>
        <Activity className="w-3.5 h-3.5" />
        <span>{isReady ? 'System Ready' : 'Degraded'}</span>
      </div>
      
      <div className="hidden sm:flex space-x-2">
        <div className={`px-2 py-1 rounded text-[10px] uppercase font-bold tracking-wider ${hasSaaras ? 'bg-primary/20 text-primary' : 'bg-surface text-textMuted'}`}>
          STT {hasSaaras ? 'ON' : 'OFF'}
        </div>
        <div className={`px-2 py-1 rounded text-[10px] uppercase font-bold tracking-wider ${hasSLM ? 'bg-primary/20 text-primary' : 'bg-surface text-textMuted'}`}>
          SLM {hasSLM ? 'ON' : 'OFF'}
        </div>
      </div>
    </div>
  );
};

export default StatusIndicator;
