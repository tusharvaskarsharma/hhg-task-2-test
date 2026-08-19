export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class APIError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

export const fetchClient = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, options);

  let data;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }

  if (!response.ok) {
    const message = data?.error?.message || `HTTP Error ${response.status}`;
    throw new APIError(message, response.status, data);
  }

  return data;
};
