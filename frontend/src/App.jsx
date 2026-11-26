import { useEffect, useState } from "react";
import { getHealth, getMe, getToken, setToken } from "./api";
import AuthModal from "./components/AuthModal";
import AboutPage from "./pages/AboutPage";
import FeedbackPage from "./pages/FeedbackPage";
import ScanPage from "./pages/ScanPage";
import ValleyPage from "./pages/ValleyPage";
import "./App.css";
import "./layout.css";

const NAV = [
  { id: "scan", label: "Scan images", icon: "◎" },
  { id: "valley", label: "Valley", icon: "✦", locked: true },
  { id: "about", label: "About", icon: "ⓘ" },
  { id: "feedback", label: "Feedbacks", icon: "★" },
];

export default function App() {
  const [page, setPage] = useState("scan");
  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState(null);   // null | "login" | "register"
  const [online, setOnline] = useState(null);

  useEffect(() => {
    getHealth().then(() => setOnline(true)).catch(() => setOnline(false));
    if (getToken()) getMe().then(setUser).catch(() => setToken(null));   // restore session
  }, []);

  const onAuthSuccess = ({ access_token, user }) => {
    setToken(access_token);
    setUser(user);
    setAuthMode(null);
  };
  const logout = () => {
    setToken(null);
    setUser(null);
    if (page === "valley") setPage("scan");
  };
  const onUnauthorized = () => { logout(); setAuthMode("login"); };   // token expired
  const openLogin = () => setAuthMode("login");

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="logo"><span className="logo-mark">P</span>PLATE<b>RECO</b></div>

        <nav className="side-nav">
          {NAV.map((n) => (
            <button key={n.id} className={`side-link ${page === n.id ? "active" : ""}`} onClick={() => setPage(n.id)}>
              <span className="side-icon">{n.icon}</span>{n.label}
              {n.locked && !user && <span className="lock">🔒</span>}
            </button>
          ))}
        </nav>

        <div className="side-bottom">
          <div className={`api-pill ${online ? "on" : online === false ? "off" : ""}`}>
            <i />{online ? "API online" : online === false ? "API offline" : "Checking…"}
          </div>
          {user ? (
            <div className="user-box">
              <div className="avatar">{user.username[0].toUpperCase()}</div>
              <div><b>{user.username}</b><small>{user.email}</small></div>
              <button className="link-btn" onClick={logout}>Log out</button>
            </div>
          ) : (
            <div className="auth-btns">
              <button className="cta small" onClick={openLogin}>Log in</button>
              <button className="ghost" onClick={() => setAuthMode("register")}>Sign up</button>
            </div>
          )}
        </div>
      </aside>

      <main className="page">
        <div className="bg-word" aria-hidden="true">تونس</div>
        {page === "scan" && <ScanPage />}
        {page === "valley" && <ValleyPage user={user} onLogin={openLogin} onUnauthorized={onUnauthorized} />}
        {page === "about" && <AboutPage />}
        {page === "feedback" && <FeedbackPage user={user} onLogin={openLogin} onUnauthorized={onUnauthorized} />}
      </main>

      {authMode && (
        <AuthModal mode={authMode} onMode={setAuthMode} onClose={() => setAuthMode(null)} onSuccess={onAuthSuccess} />
      )}
    </div>
  );
}
