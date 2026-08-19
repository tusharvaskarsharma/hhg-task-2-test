import React from 'react';
import { FileText, Star } from 'lucide-react';

const SourceCard = ({ result }) => {
  return (
    <div className="glass-panel p-4 flex flex-col space-y-3 hover:bg-surfaceHover transition-colors">
      <div className="flex justify-between items-start">
        <div className="flex items-center space-x-2">
          <FileText className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-text truncate max-w-[150px]" title={result.doc_id}>
            {result.doc_id}
          </span>
        </div>
        <div className="flex items-center space-x-1 bg-surface px-2 py-0.5 rounded text-xs font-mono border border-border">
          <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
          <span className="text-textMuted">#{result.rank}</span>
        </div>
      </div>
      
      <p className="text-xs text-text/80 line-clamp-3 leading-relaxed">
        {result.text}
      </p>

      <div className="flex justify-between items-center pt-2 mt-auto border-t border-border/50">
        <span className="text-[10px] text-textMuted uppercase font-bold tracking-wider">
          {result.source || 'corpus'}
        </span>
        <span className="text-[10px] font-mono text-textMuted bg-background px-1.5 rounded">
          Score: {result.score.toFixed(3)}
        </span>
      </div>
    </div>
  );
};

export default SourceCard;
