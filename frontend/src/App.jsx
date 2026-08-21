import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import QueryInput from './components/QueryInput';
import VoiceInput from './components/VoiceInput';
import AnswerPanel from './components/AnswerPanel';
import RetrievalPanel from './components/RetrievalPanel';
import MetricsPanel from './components/MetricsPanel';
import LoadingState from './components/LoadingState';
import ErrorState from './components/ErrorState';
import { useQuery } from './hooks/useQuery';
import { useVoice } from './hooks/useVoice';
import { checkHealth, checkReady } from './api/query';
import { Sparkles } from 'lucide-react';
import gsap from 'gsap';

const App = () => {
  const [sysStatus, setSysStatus] = useState(null);
  const [selectedLang, setSelectedLang] = useState('auto');
  const [sysError, setSysError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  
  const UI_GENERATE = true;
  
  const textAPI = useQuery();
  const voiceAPI = useVoice();

  // Initialization (fetches health status on mount)
  useEffect(() => {
    const init = async () => {
      try {
        const health = await checkHealth();
        try {
          await checkReady();
          setSysStatus(health);
          setSysError(null);
        } catch (readyErr) {
          if (readyErr.status === 503) {
            setSysError("Backend is running but RAG artifacts are not ready");
          } else {
            setSysError("Backend error");
          }
        }
      } catch (healthErr) {
        if (healthErr.message.includes('Failed to fetch') || healthErr.message.includes('NetworkError')) {
          setSysError("Backend is not running on port 8000");
        } else {
          setSysError("Backend error");
        }
      }
    };
    init();
      
    // GSAP Initial entrance
    gsap.fromTo('.gsap-stagger-item', 
      { opacity: 0, y: 30, rotateX: 10 },
      { opacity: 1, y: 0, rotateX: 0, duration: 0.8, stagger: 0.1, ease: 'power3.out' }
    );
  }, [retryCount]);

  const handleTextSubmit = (query, lang) => {
    voiceAPI.reset();
    textAPI.submitQuery(query, lang, 5, UI_GENERATE);
  };

  const handleVoiceStop = () => {
    voiceAPI.stopRecordingAndSubmit(selectedLang, 5, UI_GENERATE);
  };

  const isBusy = textAPI.loading || ['RECORDING', 'PROCESSING'].includes(voiceAPI.state);
  const hasError = textAPI.error || voiceAPI.error;
  const currentData = textAPI.data || voiceAPI.data;
  const currentError = textAPI.error || voiceAPI.error;

  const handleRetry = () => {
    textAPI.reset();
    voiceAPI.reset();
  };

  return (
    <div className="min-h-screen w-full flex flex-col relative z-0">
      <Header status={sysStatus} />
      
      <main className="flex-1 w-full max-w-4xl mx-auto px-4 sm:px-6 pb-20 flex flex-col items-center">
        
        {/* Input Region */}
        <div className="w-full mb-10 flex flex-col sm:flex-row items-center gap-4 gsap-stagger-item">
          <div className="flex-1 w-full">
            <QueryInput 
              onSubmit={handleTextSubmit} 
              disabled={isBusy} 
              selectedLang={selectedLang}
              setSelectedLang={setSelectedLang}
            />
          </div>
          <div className="shrink-0 relative z-20">
            <VoiceInput 
              state={voiceAPI.state}
              onStart={voiceAPI.startRecording}
              onStop={handleVoiceStop}
              disabled={textAPI.loading || (sysStatus && sysStatus.saaras?.enabled === false)}
              title={sysStatus && sysStatus.saaras?.enabled === false ? "Speech to text service is disabled" : undefined}
            />
          </div>
        </div>



        {sysError && (
          <div className="w-full mb-8 p-4 bg-error/10 border border-error/20 rounded-xl flex items-center justify-between gsap-stagger-item text-error">
            <span className="font-medium text-sm">{sysError}</span>
            <button 
              onClick={() => setRetryCount(c => c + 1)}
              className="px-3 py-1.5 bg-error/20 hover:bg-error/30 rounded-lg text-xs font-bold transition-colors"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Dynamic States */}
        <div className="w-full gsap-stagger-item perspective-1000">
          
          {/* Empty State */}
          {!currentData && !isBusy && !hasError && (
            <div className="w-full flex flex-col items-center justify-center py-20 text-center space-y-4 opacity-70">
              <Sparkles className="w-12 h-12 text-primary/40 mb-2" />
              <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/50">
                What would you like to know?
              </h2>
              <p className="text-textMuted max-w-md">
                Search the corpus naturally. The HHG Engine uses dense retrieval (E5) and semantic routing to synthesize answers instantly.
              </p>
            </div>
          )}

          {/* Loading States */}
          {textAPI.loading && <LoadingState message="Retrieving context and generating answer..." />}
          {voiceAPI.state === 'RECORDING' && <LoadingState message="Listening... Click stop when finished." />}
          {voiceAPI.state === 'PROCESSING' && <LoadingState message="Transcribing and analyzing..." />}

          {/* Error State */}
          {hasError && <ErrorState error={currentError} onRetry={handleRetry} />}

          {/* Success Result */}
          {currentData && !isBusy && (
            <div className="flex flex-col w-full space-y-6">
              <AnswerPanel 
                data={currentData}
              />
              <RetrievalPanel results={currentData.results} />
              <MetricsPanel latency={currentData.latency} cache={currentData.cache} />
            </div>
          )}
          
        </div>
      </main>
    </div>
  );
};

export default App;
