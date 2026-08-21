import React, { useEffect, useRef } from 'react';
import { Mic, Square } from 'lucide-react';
import gsap from 'gsap';

const VoiceInput = ({ state, onStart, onStop, disabled, title }) => {
  const buttonRef = useRef(null);
  const rippleRef = useRef(null);

  useEffect(() => {
    if (state === 'RECORDING') {
      gsap.to(rippleRef.current, {
        scale: 2.5,
        opacity: 0,
        duration: 1.5,
        repeat: -1,
        ease: "sine.out"
      });
      gsap.to(buttonRef.current, {
        scale: 1.1,
        duration: 0.3,
        yoyo: true,
        repeat: -1,
        ease: "sine.inOut"
      });
    } else {
      gsap.killTweensOf(rippleRef.current);
      gsap.killTweensOf(buttonRef.current);
      gsap.set(rippleRef.current, { scale: 1, opacity: 0.5 });
      gsap.to(buttonRef.current, { scale: 1, duration: 0.3 });
    }
  }, [state]);

  const isRecording = state === 'RECORDING';
  
  return (
    <div className="relative flex items-center justify-center">
      <div 
        ref={rippleRef} 
        className={`absolute w-full h-full rounded-full ${isRecording ? 'bg-error/30' : 'bg-primary/20'} pointer-events-none z-0`}
      />
      
      <button
        type="button"
        ref={buttonRef}
        onClick={isRecording ? onStop : onStart}
        disabled={disabled}
        aria-label={isRecording ? "Stop recording" : "Start voice query"}
        title={title || (isRecording ? "Stop recording" : "Ask by voice")}
        className={`relative z-10 w-14 h-14 rounded-full flex items-center justify-center transition-colors shadow-glass focus:outline-none focus:ring-4 ${
          isRecording 
            ? 'bg-error hover:bg-error/90 focus:ring-error/30' 
            : 'bg-primary hover:bg-primaryHover focus:ring-primary/30 disabled:opacity-50 disabled:bg-surface'
        }`}
      >
        {isRecording ? (
          <Square className="w-5 h-5 text-white fill-white" />
        ) : (
          <Mic className={`w-6 h-6 ${disabled ? 'text-textMuted' : 'text-white'}`} />
        )}
      </button>
    </div>
  );
};

export default VoiceInput;
