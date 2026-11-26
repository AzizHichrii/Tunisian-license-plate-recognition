import { useEffect, useRef, useState } from "react";
import { askValley } from "../api";

const SUGGESTIONS = [
  "How does a turbocharger work?",
  "Quelles sont les conditions pour obtenir un permis de conduire ?",
  "كم مرة يجب تغيير زيت المحرك؟",
  "What does a flashing red light at a level crossing mean?",
];

/** Tiny Markdown renderer for Gemini answers: paragraphs, bullet lists, **bold**, *italic line*. */
function Markdown({ text }) {
  const inline = (s, key) => s.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={`${key}-${i}`}>{part.slice(2, -2)}</strong> : part);
  const blocks = [];
  let list = null;
  text.split("\n").forEach((raw, i) => {
    const line = raw.trim();
    const bullet = line.match(/^(?:[*-]|\d+\.)\s+(.*)/);
    if (bullet) { (list ??= []).push(<li key={i}>{inline(bullet[1], i)}</li>); return; }
    if (list) { blocks.push(<ul key={`ul-${i}`}>{list}</ul>); list = null; }
    if (!line) return;
    const italic = line.match(/^\*([^*].*[^*])\*$/);
    blocks.push(italic ? <p key={i}><em>{italic[1]}</em></p> : <p key={i}>{inline(line, i)}</p>);
  });
  if (list) blocks.push(<ul key="ul-end">{list}</ul>);
  return <div className="md" dir="auto">{blocks}</div>;
}

function Sources({ sources, ms }) {
  if (!sources?.length) return null;
  const total = ((ms.rewrite + ms.search + ms.generate) / 1000).toFixed(1);
  return (
    <details className="sources">
      <summary>{sources.length} sources · {total} s</summary>
      <ol>
        {sources.map((s) => (
          <li key={s.n}>
            <b>[{s.n}]</b> {s.title}{s.section ? ` — ${s.section}` : ""}{s.page ? ` · p.${s.page}` : ""}{" "}
            {s.source.startsWith("http")
              ? <a href={s.source} target="_blank" rel="noreferrer">open ↗</a>
              : <span className="muted">({s.source})</span>}
          </li>
        ))}
      </ol>
    </details>
  );
}

export default function ValleyPage({ user, onLogin, onUnauthorized }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  if (!user) {
    return (
      <section className="locked fade-up">
        <div className="lock-icon">🔒</div>
        <span className="eyebrow">Members only</span>
        <h1>Meet <em>Valley</em></h1>
        <p>Your AI assistant for cars and the Tunisian code de la route. Ask in Arabic, French or English,
          and get answers backed by official texts and cited sources.</p>
        <button className="cta" onClick={onLogin}>Log in to chat</button>
      </section>
    );
  }

  const send = async (question = input) => {
    const q = question.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const r = await askValley(q);
      setMessages((m) => [...m, { role: "valley", text: r.answer, sources: r.sources, ms: r.timings_ms }]);
    } catch (e) {
      if (e.status === 401) { onUnauthorized(); return; }
      setMessages((m) => [...m, { role: "error", text: e.message }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="chat fade-up">
      <header className="chat-head">
        <span className="eyebrow">RAG · Gemini · pgvector</span>
        <h1>Ask <em>Valley</em></h1>
        <p className="muted">Cars, maintenance and Tunisian road law, in Arabic, French or English. Every answer cites its sources.</p>
      </header>

      <div className="chat-log">
        {messages.length === 0 && (
          <div className="suggestions">
            {SUGGESTIONS.map((s) => <button key={s} className="chip" dir="auto" onClick={() => send(s)}>{s}</button>)}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.role === "valley"
              ? <><Markdown text={m.text} /><Sources sources={m.sources} ms={m.ms} /></>
              : <p dir="auto">{m.text}</p>}
          </div>
        ))}
        {loading && <div className="msg valley typing"><span /><span /><span /></div>}
        <div ref={endRef} />
      </div>

      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <input dir="auto" value={input} maxLength={1000} onChange={(e) => setInput(e.target.value)}
               placeholder="Ask about cars or the Tunisian code de la route…" />
        <button className="cta" disabled={!input.trim() || loading}>Send</button>
      </form>
      <small className="muted disclaimer">Valley can make mistakes. For legal matters, always check the official texts.</small>
    </section>
  );
}
