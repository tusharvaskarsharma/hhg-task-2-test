import { fetchClient } from "./client";

export const executeVoiceQuery = async (audioBlob, language, top_k = 10, generate = false) => {
  const formData = new FormData();
  formData.append('audio', audioBlob, `voice_query.wav`);
  if (language && language !== 'auto') {
    formData.append('language', language);
  }
  formData.append('top_k', top_k);
  formData.append('generate', generate);

  return await fetchClient("/api/voice", {
    method: "POST",
    body: formData, // browser automatically sets multipart/form-data boundary
  });
};
