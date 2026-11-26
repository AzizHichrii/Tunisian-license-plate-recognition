import { useEffect, useState } from "react";
import { listFeedback, sendFeedback } from "../api";

function Stars({ value, onChange, size = "md" }) {
  return (
    <div className={`stars ${size} ${onChange ? "editable" : ""}`}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button type="button" key={n} disabled={!onChange} className={n <= value ? "on" : ""}
                onClick={() => onChange?.(n)} aria-label={`${n} star${n > 1 ? "s" : ""}`}>★</button>
      ))}
    </div>
  );
}

export default function FeedbackPage({ user, onLogin, onUnauthorized }) {
  const [data, setData] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [rating, setRating] = useState(0);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);

  useEffect(() => {
    listFeedback().then(setData).catch((e) => setLoadError(e.message));
  }, []);

  const submit = async (e) => {
    e.preventDefault();
    if (!rating) { setError("Please choose a rating."); return; }
    setSending(true); setError(""); setSent(false);
    try {
      await sendFeedback(rating, text);
      setRating(0); setText(""); setSent(true);
      setData(await listFeedback());
    } catch (err) {
      if (err.status === 401) { onUnauthorized(); return; }
      setError(err.message);
    } finally {
      setSending(false);
    }
  };

  return (
    <section className="fb fade-up">
      <span className="eyebrow">Community</span>
      <h1>What users <em>think</em></h1>

      <div className="fb-summary">
        <div className="fb-avg">
          <strong>{data?.average != null ? data.average.toFixed(1) : "—"}</strong>
          <Stars value={Math.round(data?.average ?? 0)} />
          <span className="muted">{data?.count ?? 0} review{data?.count === 1 ? "" : "s"}</span>
        </div>

        {user ? (
          <form className="fb-form" onSubmit={submit}>
            <h3>Leave your feedback</h3>
            <Stars value={rating} onChange={setRating} size="lg" />
            <textarea dir="auto" required minLength={3} maxLength={1000} value={text}
                      onChange={(e) => setText(e.target.value)} placeholder="Tell us about your experience…" />
            {error && <div className="alert error">{error}</div>}
            {sent && <div className="alert ok">Thanks for your feedback!</div>}
            <button className="cta" disabled={sending}>{sending ? "Sending…" : "Publish"}</button>
          </form>
        ) : (
          <div className="fb-form">
            <h3>Want to share your opinion?</h3>
            <p className="muted">Log in to leave a review with a rating.</p>
            <button className="cta" onClick={onLogin}>Log in</button>
          </div>
        )}
      </div>

      {loadError && <div className="alert error">{loadError}</div>}
      <div className="fb-list">
        {data?.items.map((f) => (
          <article key={f.id} className="fb-item">
            <div className="fb-top">
              <div className="avatar">{f.username[0].toUpperCase()}</div>
              <div><b>{f.username}</b><small className="muted">{new Date(f.created_at).toLocaleDateString()}</small></div>
              <Stars value={f.rating} size="sm" />
            </div>
            <p dir="auto">{f.text}</p>
          </article>
        ))}
        {data && data.items.length === 0 && <p className="muted">No feedback yet. Be the first!</p>}
      </div>
    </section>
  );
}
