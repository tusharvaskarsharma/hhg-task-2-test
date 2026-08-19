import React, { useEffect, useRef } from 'react';
import { Bot, AlertTriangle, ShieldCheck } from 'lucide-react';
import gsap from 'gsap';

const AnswerPanel = ({ answer, grounding, transcription, source }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (answer) {
      gsap.fromTo(containerRef.current,
        { opacity: 0, y: 20, rotateX: -10 },
        { opacity: 1, y: 0, rotateX: 0, duration: 0.6, ease: "power3.out" }
      );
    }
  }, [answer]);

  if (!answer) return null;

  const isGrounded = grounding?.grounded;
  const isEnabled = grounding?.enabled;

  return (
    <div ref={containerRef} className="w-full glass-panel p-6 sm:p-8 relative overflow-hidden group">
      {/* Decorative gradient blob */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -z-10 group-hover:bg-primary/10 transition-colors duration-1000" />

      {transcription && (
        <div className="mb-6 p-4 rounded-xl bg-surface/50 border border-border/50 text-sm">
          <span className="text-textMuted uppercase font-bold text-[10px] tracking-wider block mb-1">
            You said (STT):
          </span>
          <span className="text-text/90 italic">"{transcription.text}"</span>
        </div>
      )}

      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0 w-10 h-10 bg-primary/20 rounded-full flex items-center justify-center border border-primary/30 shadow-glass mt-1">
          <Bot className="w-5 h-5 text-primary" />
        </div>
        
        <div className="flex-1 space-y-4">
          <div className="prose prose-invert max-w-none">
            <p className="text-base sm:text-lg text-text leading-relaxed font-medium">
              {answer}
            </p>
          </div>

          <div className="pt-4 mt-4 border-t border-border flex flex-col space-y-2">
            {source && (
              <div className="flex items-center text-xs font-medium text-textMuted/80">
                <span className="mr-2 uppercase tracking-wide opacity-70">Source:</span>
                <span className="bg-surface border border-border px-2 py-0.5 rounded-md">
                  {source}
                </span>
              </div>
            )}
            {isEnabled && grounding?.status && (
              <div className="flex items-center">
                {grounding.status === "SUPPORTED" ? (
                  <div className="flex items-center space-x-2 text-xs font-medium text-success bg-success/10 px-3 py-1.5 rounded-full border border-success/20">
                    <ShieldCheck className="w-4 h-4" />
                    <span>Answer grounded in corpus context</span>
                  </div>
                ) : grounding.status === "INSUFFICIENT_CONTEXT" ? (
                  <div className="flex items-center space-x-2 text-xs font-medium text-warning bg-warning/10 px-3 py-1.5 rounded-full border border-warning/20">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Insufficient context — answer restricted</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2 text-xs font-medium text-error bg-error/10 px-3 py-1.5 rounded-full border border-error/20">
                    <AlertTriangle className="w-4 h-4" />
                    <span>Answer is unsupported by context — may contain hallucinations</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnswerPanel;
