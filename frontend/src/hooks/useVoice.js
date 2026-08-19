import { useState, useRef } from 'react';
import { executeVoiceQuery } from '../api/voice';

export const useVoice = () => {
  const [state, setState] = useState('IDLE'); // IDLE, RECORDING, PROCESSING, SUCCESS, ERROR
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstart = () => {
        console.log("VOICE START");
        console.log("VOICE RECORDING");
        setState('RECORDING');
        setError(null);
      };

      mediaRecorder.start();
    } catch (err) {
      setError(new Error('Microphone permission denied or unavailable.'));
      setState('ERROR');
    }
  };

  const stopRecordingAndSubmit = (language) => {
    return new Promise((resolve, reject) => {
      const mediaRecorder = mediaRecorderRef.current;
      if (!mediaRecorder || mediaRecorder.state === 'inactive') return resolve();

      mediaRecorder.onstop = async () => {
        console.log("VOICE STOP");
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        
        console.log("AUDIO SIZE:", audioBlob.size);
        console.log("AUDIO MIME:", audioBlob.type);

        // Stop all tracks to release microphone
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        if (audioBlob.size === 0) {
            console.log("Audio blob is empty. Aborting voice API request.");
            setState('IDLE');
            return resolve();
        }

        console.log("VOICE API REQUEST");
        setState('PROCESSING');
        try {
          const response = await executeVoiceQuery(audioBlob, language);
          setData(response);
          setState('SUCCESS');
          resolve(response);
        } catch (err) {
          setError(err);
          setState('ERROR');
          reject(err);
        }
      };

      mediaRecorder.stop();
    });
  };

  const cancelRecording = () => {
    const mediaRecorder = mediaRecorderRef.current;
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.onstop = null; // discard
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      mediaRecorder.stop();
    }
    setState('IDLE');
    setError(null);
  };

  const reset = () => {
    cancelRecording();
    setState('IDLE');
    setData(null);
    setError(null);
  };

  return {
    state,
    error,
    data,
    startRecording,
    stopRecordingAndSubmit,
    cancelRecording,
    reset
  };
};
