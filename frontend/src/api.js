const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "plateReco.token";

export function getToken() {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(token) {
  try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY); } catch { /* storage unavailable */ }
}

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function request(path, { method = "GET", body, form, auth = false } = {}) {
  const headers = {};
  if (body) headers["Content-Type"] = "application/json";
  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, { method, headers, body: form ?? (body ? JSON.stringify(body) : undefined) });
  } catch {
    throw new ApiError("Cannot reach the API. Is the server running?", 0);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join(" · ");   // 422 validation errors
    throw new ApiError(msg ?? `Server error (${res.status})`, res.status);
  }
  return data;
}

export function predict(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/predict", { method: "POST", form });
}
export const getHealth = () => request("/health");
export const register = (email, username, password) =>
  request("/auth/register", { method: "POST", body: { email, username, password } });
export const login = (email, password) => request("/auth/login", { method: "POST", body: { email, password } });
export const getMe = () => request("/auth/me", { auth: true });
export const listFeedback = () => request("/feedback");
export const sendFeedback = (rating, text) => request("/feedback", { method: "POST", body: { rating, text }, auth: true });
export const askValley = (question) => request("/valley/ask", { method: "POST", body: { question }, auth: true });
