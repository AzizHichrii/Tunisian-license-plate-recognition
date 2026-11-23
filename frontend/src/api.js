const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function predict(file) {
  const form = new FormData();
  form.append("file", file);

  let res;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
  } catch {
    throw new Error("Cannot reach the API. Is the server running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Server error (${res.status})`);
  return data;
}
export async function getHealth() {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("API not ready");
  return res.json();
}