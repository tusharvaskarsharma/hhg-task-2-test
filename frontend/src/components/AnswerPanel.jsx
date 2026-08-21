import React, { useEffect, useRef } from 'react';
import { Bot, AlertTriangle, ShieldCheck } from 'lucide-react';
import gsap from 'gsap';

const AnswerPanel = ({ data }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (data) {
      gsap.fromTo(containerRef.current,
        { opacity: 0, y: 20, rotateX: -10 },
        { opacity: 1, y: 0, rotateX: 0, duration: 0.6, ease: "power3.out" }
      );
    }
  }, [data]);

  if (!data) return null;

  const { transcription, answer, answer_source, extractive_answer, generated_answer, grounding, latency } = data;
  const isAbstain = answer_source === 'abstain' || answer_source === 'generated-unavailable';
  
  return (
    <div ref={containerRef} className="w-full flex flex-col space-y-6 relative overflow-hidden group">
      
      {transcription && (
        <div className="p-4 rounded-xl bg-surface/50 border border-border/50 text-sm glass-panel w-full">
          <span className="text-textMuted uppercase font-bold text-[10px] tracking-wider block mb-1">
            You said (STT):
          </span>
          <span className="text-text/90 italic">"{transcription.text}"</span>
        </div>
      )}

      {isAbstain ? (
        <div className="w-full glass-panel p-6 sm:p-8 relative">
           <div className="flex items-start space-x-4">
              <div className="flex-shrink-0 w-10 h-10 bg-warning/20 rounded-full flex items-center justify-center border border-warning/30 shadow-glass mt-1">
                <AlertTriangle className="w-5 h-5 text-warning" />
              </div>
              <div className="flex-1 space-y-2">
                <h3 className="text-lg font-bold text-text">No supported answer</h3>
                <p className="text-textMuted">{answer || "Insufficient information in the indexed corpus."}</p>
                <div className="flex items-center text-xs font-medium text-textMuted/80 mt-4 pt-4 border-t border-border">
                  <span className="mr-2 uppercase tracking-wide opacity-70">Answer source:</span>
                  <span className="bg-surface border border-border px-2 py-0.5 rounded-md">
                    {answer_source}
                  </span>
                </div>
              </div>
           </div>
        </div>
      ) : (
        <div className="w-full">
          {/* Extractive Answer Card */}
          <div className="w-full glass-panel p-6 relative flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-text flex items-center">
                <Bot className="w-5 h-5 mr-2 text-primary" />
                Extractive answer
              </h3>
              <div className="flex items-center space-x-1 text-xs font-medium text-success bg-success/10 px-2 py-1 rounded-full border border-success/20">
                <ShieldCheck className="w-3 h-3" />
                <span>Dataset grounded</span>
              </div>
            </div>
            <div className="flex-1 text-base text-text leading-relaxed font-medium mb-4">
              {extractive_answer || (answer_source === 'extractive' ? answer : "Not available.")}
            </div>
            <div className="pt-4 border-t border-border flex flex-col space-y-2 text-xs font-medium text-textMuted/80 mt-auto">
              <div className="flex items-center">
                <span className="mr-2 uppercase tracking-wide opacity-70">Source IDs:</span>
                <span className="bg-surface border border-border px-2 py-0.5 rounded-md flex-1 truncate">
                  {grounding?.sources?.map(s => s.id).join(', ') || "N/A"}
                </span>
              </div>
              <div className="flex items-center">
                <span className="mr-2 uppercase tracking-wide opacity-70">Answer source:</span>
                <span className="bg-surface border border-border px-2 py-0.5 rounded-md">
                  extractive
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnswerPanel;
