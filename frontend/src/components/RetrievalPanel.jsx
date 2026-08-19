import React, { useState, useRef, useEffect } from 'react';
import { Database, ChevronDown, ChevronUp } from 'lucide-react';
import SourceCard from './SourceCard';
import gsap from 'gsap';

const RetrievalPanel = ({ results }) => {
  const [isOpen, setIsOpen] = useState(false);
  const contentRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      gsap.fromTo(contentRef.current,
        { height: 0, opacity: 0 },
        { height: 'auto', opacity: 1, duration: 0.4, ease: "power2.out" }
      );
    } else if (contentRef.current) {
      gsap.to(contentRef.current, { height: 0, opacity: 0, duration: 0.3 });
    }
  }, [isOpen]);

  if (!results || results.length === 0) return null;

  return (
    <div className="w-full mt-6">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 text-sm font-medium text-textMuted hover:text-text transition-colors py-2 focus:outline-none"
      >
        <Database className="w-4 h-4" />
        <span>Retrieval Context ({results.length} sources)</span>
        {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      <div ref={contentRef} className="overflow-hidden opacity-0 h-0">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 pt-4 pb-2">
          {results.map((res, i) => (
            <SourceCard key={i} result={res} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default RetrievalPanel;
