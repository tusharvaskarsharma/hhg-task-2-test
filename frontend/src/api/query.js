import { fetchClient } from "./client";

export const executeQuery = async (queryText, language, topK = 5, generate = false) => {
  const payload = {
    query: queryText,
    top_k: topK,
    generate: generate
  };

  if (language && language !== 'auto') {
    payload.language = language;
  }

  return await fetchClient("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
};

export const checkHealth = async () => {
  return await fetchClient("/api/health");
};

export const checkReady = async () => {
  return await fetchClient("/api/ready");
};
