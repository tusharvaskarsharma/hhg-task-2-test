import React from 'react';
import { Layers } from 'lucide-react';
import StatusIndicator from './StatusIndicator';

const Header = ({ status }) => {
  return (
    <header className="w-full flex items-center justify-between p-6 mb-8 gsap-reveal" id="app-header">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 bg-primary/20 rounded-xl flex items-center justify-center border border-primary/30 shadow-glass">
          <Layers className="text-primary w-5 h-5" />
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
            HHG Engine
          </h1>
          <p className="text-xs text-textMuted font-medium">Phase B Search Core</p>
        </div>
      </div>
      
      <StatusIndicator status={status} />
    </header>
  );
};

export default Header;
