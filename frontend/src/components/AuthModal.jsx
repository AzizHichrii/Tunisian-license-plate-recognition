import { useState } from "react";
import { login, register } from "../api";

export default function AuthModal({ mode, onMode, onClose, onSuccess }) {
  const [form, setForm] = useState({ email: "", username: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const isRegister = mode === "register";
  const set = (key) => (e) => setForm({ ...form, [key]: e.target.value });

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      onSuccess(isRegister
        ? await register(form.email, form.username, form.password)
        : await login(form.email, form.password));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <form className="modal fade-up" onClick={(e) => e.stopPropagation()} onSubmit={submit}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">×</button>
        <h2>{isRegister ? "Create your account" : "Welcome back"}</h2>
        <p className="muted">{isRegister ? "Sign up to chat with Valley and leave feedback." : "Log in to chat with Valley."}</p>

        <label>Email
          <input type="email" required autoComplete="email" value={form.email} onChange={set("email")} />
        </label>
        {isRegister && (
          <label>Username
            <input required minLength={3} maxLength={50} value={form.username} onChange={set("username")} />
          </label>
        )}
        <label>Password
          <input type="password" required minLength={isRegister ? 8 : 1} maxLength={72}
                 autoComplete={isRegister ? "new-password" : "current-password"}
                 value={form.password} onChange={set("password")} />
        </label>
        {isRegister && <small className="muted">At least 8 characters.</small>}

        {error && <div className="alert error">{error}</div>}
        <button className="cta full" disabled={loading}>
          {loading ? "Please wait…" : isRegister ? "Sign up" : "Log in"}
        </button>
        <p className="muted switch">
          {isRegister ? "Already have an account?" : "No account yet?"}{" "}
          <button type="button" className="link-btn" onClick={() => { setError(""); onMode(isRegister ? "login" : "register"); }}>
            {isRegister ? "Log in" : "Sign up"}
          </button>
        </p>
      </form>
    </div>
  );
}
