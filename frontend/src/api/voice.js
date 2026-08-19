import { fetchClient } from "./client";

export const executeVoiceQuery = async (audioBlob, language, topK = 5, generate = false) => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "voice_record.webm");
  
  if (language) {
    formData.append("language", language);
  }
  formData.append("top_k", topK);
  formData.append("generate", generate);

  return await fetchClient("/api/voice", {
    method: "POST",
    body: formData, // browser automatically sets multipart/form-data boundary
  });
};
