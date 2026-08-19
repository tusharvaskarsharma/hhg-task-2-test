import { fetchClient } from "./client";

export const executeQuery = async (queryText, language, topK = 5) => {
  return await fetchClient("/api/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: queryText,
      language: language,
      top_k: topK
    }),
  });
};

export const checkHealth = async () => {
  return await fetchClient("/api/health");
};

export const checkReady = async () => {
  return await fetchClient("/api/ready");
};
