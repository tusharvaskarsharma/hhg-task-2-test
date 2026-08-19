import React, { useEffect, useRef } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import gsap from 'gsap';

const ErrorState = ({ error, onRetry }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    gsap.fromTo(containerRef.current,
      { opacity: 0, y: 10, rotationX: 10 },
      { opacity: 1, y: 0, rotationX: 0, duration: 0.5, ease: "back.out(1.7)" }
    );
  }, []);

  const getMessage = () => {
    if (!error) return "An unknown error occurred.";
    if (error.data?.error?.message) return error.data.error.message;
    return error.message || String(error);
  };

  return (
    <div ref={containerRef} className="w-full flex flex-col items-center justify-center py-8">
      <div className="glass-panel border-error/30 p-6 flex flex-col items-center max-w-md w-full text-center">
        <div className="w-12 h-12 bg-error/10 rounded-full flex items-center justify-center mb-4">
          <AlertCircle className="w-6 h-6 text-error" />
        </div>
        <h3 className="text-lg font-bold text-white mb-2">Something went wrong</h3>
        <p className="text-sm text-textMuted mb-6 break-words w-full">
          {getMessage()}
        </p>
        
        {onRetry && (
          <button 
            onClick={onRetry}
            className="glass-button px-6 py-2.5 text-sm font-medium hover:text-white group space-x-2"
          >
            <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
            <span>Try Again</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorState;
