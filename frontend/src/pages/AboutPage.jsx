const PIPELINE = ["Car photo", "YOLO11n detects the plate", "Crop", "CRNN reads the text", "Format check", "Result"];

const CARDS = [
  {
    title: "Plate recognition",
    text: "A YOLO11n detector finds Tunisian plates (97.6% recall on the test set), then a custom CRNN "
      + "(CNN + BiLSTM + CTC) reads them: 86.5% of plates read exactly, 4.1% character error rate.",
  },
  {
    title: "Valley assistant",
    text: "A RAG chatbot over the Tunisian code de la route and car knowledge: multilingual bge-m3 "
      + "embeddings in pgvector, rank fusion across Arabic, French and English queries, and Gemini answers "
      + "that cite their sources.",
  },
  {
    title: "MLOps",
    text: "Data versioned with DVC, every experiment tracked in MLflow, models compared on validation and "
      + "evaluated once on a held-out test set, API covered by automated tests.",
  },
];

const STACK = ["PyTorch", "Ultralytics YOLO", "MLflow", "DVC", "FastAPI", "PostgreSQL + pgvector",
  "sentence-transformers", "Gemini", "React", "Docker"];

const LIMITS = [
  "Plates are mostly in the 'TUN' format; very few 'ن.ت' plates were available for training.",
  "OCR confidence is a useful signal but not a guarantee: always check uncertain readings.",
  "Valley only knows its documents (code de la route compiled in 2012 + later decrees, Wikipedia). "
    + "It is information, not legal advice.",
];

export default function AboutPage() {
  return (
    <section className="about fade-up">
      <span className="eyebrow">About the project</span>
      <h1>Plate<em>Reco</em></h1>
      <p className="muted lead">An end-to-end AI project: Tunisian license plate recognition plus an assistant
        for cars and road law, built from data collection to deployment.</p>

      <div className="pipeline">
        {PIPELINE.map((step, i) => (
          <span key={step} className="pipe-step">{step}{i < PIPELINE.length - 1 && <i>→</i>}</span>
        ))}
      </div>

      <div className="about-grid">
        {CARDS.map((c) => (
          <div key={c.title} className="about-card"><h3>{c.title}</h3><p>{c.text}</p></div>
        ))}
      </div>

      <div className="about-grid">
        <div className="about-card">
          <h3>Tech stack</h3>
          <div className="tags">{STACK.map((t) => <span key={t} className="badge">{t}</span>)}</div>
        </div>
        <div className="about-card">
          <h3>Known limitations</h3>
          <ul>{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
        </div>
      </div>

      <p className="muted credit">Built by Mohamed Aziz Hichri · Data Science engineering student, TEKUP.</p>
    </section>
  );
}
