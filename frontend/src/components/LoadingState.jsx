import React, { useEffect, useRef } from 'react';
import { Loader2 } from 'lucide-react';
import gsap from 'gsap';

const LoadingState = ({ message = "Processing..." }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    gsap.fromTo(containerRef.current,
      { opacity: 0, y: 10, scale: 0.95 },
      { opacity: 1, y: 0, scale: 1, duration: 0.4, ease: "power2.out" }
    );
  }, []);

  return (
    <div ref={containerRef} className="w-full flex flex-col items-center justify-center py-16">
      <div className="glass-panel p-6 flex items-center space-x-4">
        <Loader2 className="w-6 h-6 text-primary animate-spin" />
        <span className="text-sm font-medium text-text">{message}</span>
      </div>
    </div>
  );
};

export default LoadingState;
