import { useEffect, useState } from "react";
import { getHealth, predict } from "./api";
import "./App.css";

const STATS = [
  { value: "97.6%", label: "Detection recall (test)" },
  { value: "86.5%", label: "Plates read exactly" },
  { value: "~60ms", label: "Per image on GPU" },
];
const STATUS_LABEL = { valid: "Valid", uncertain: "Uncertain", invalid: "Invalid format" };

function statusOf(d) {
  if (d.status !== "valid") return "invalid";
  return d.uncertain ? "uncertain" : "valid";
}

function PlateCrop({ src, bbox, size }) {
  const [x1, y1, x2, y2] = bbox;
  const scale = 220 / (x2 - x1);
  return (
    <div className="crop" style={{
      width: 220, height: (y2 - y1) * scale,
      backgroundImage: `url(${src})`,
      backgroundSize: `${size.w * scale}px ${size.h * scale}px`,
      backgroundPosition: `-${x1 * scale}px -${y1 * scale}px`,
    }} />
  );
}

function Meter({ label, value }) {
  return (
    <div className="meter">
      <span>{label}</span>
      <div className="bar"><i style={{ width: `${value * 100}%` }} /></div>
      <b>{(value * 100).toFixed(1)}%</b>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [size, setSize] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [online, setOnline] = useState(null);

  useEffect(() => {
    getHealth().then(() => setOnline(true)).catch(() => setOnline(false));
  }, []);

  const onSelect = (f) => {
    if (!f) return;
    setFile(f); setResult(null); setError(""); setSize(null);
    setPreview(URL.createObjectURL(f));
  };

  const onAnalyze = async () => {
    setLoading(true); setError(""); setResult(null);
    try { setResult(await predict(file)); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="page">
      <div className="bg-word" aria-hidden="true">تونس</div>

      <nav className="nav">
        <div className="nav-links"><a href="#scan">Scan</a><a href="#results">Results</a></div>
        <div className="logo"><span className="logo-mark">P</span>PLATE<b>RECO</b></div>
        <div className={`api-pill ${online ? "on" : online === false ? "off" : ""}`}>
          <i />{online ? "API online" : online === false ? "API offline" : "Checking…"}
        </div>
      </nav>

      <section className="hero">
        <div className="fade-up">
          <span className="eyebrow">AI · Computer Vision · MLOps</span>
          <h1>Read Every<br /><em>Tunisian</em> Plate</h1>
          <p>Upload a car photo. YOLO11n finds the plate, a CRNN reads it, in milliseconds.</p>

          <label className={`dropzone ${dragging ? "drag" : ""}`}
                 onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                 onDragLeave={() => setDragging(false)}
                 onDrop={(e) => { e.preventDefault(); setDragging(false); onSelect(e.dataTransfer.files[0]); }}>
            <input type="file" accept="image/jpeg,image/png" hidden
                   onChange={(e) => onSelect(e.target.files[0])} />
            <span className="dz-icon">⬆</span>
            <span>{file ? file.name : "Drop a car image or click to browse"}</span>
            <small>JPG or PNG · max 5 MB</small>
          </label>

          <button className="cta" disabled={!file || loading} onClick={onAnalyze}>
            {loading ? "Scanning…" : "Analyze plate →"}
          </button>
          {error && <div className="alert error">{error}</div>}
        </div>

        <ul className="stats fade-up delay">
          {STATS.map((s) => <li key={s.label}><strong>{s.value}</strong><span>{s.label}</span></li>)}
        </ul>
      </section>

      {preview && (
        <section id="scan" className="stage fade-up">
          <div className={`viewer ${loading ? "scanning" : ""}`}>
            <img src={preview} alt="uploaded car"
                 onLoad={(e) => setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
            {loading && <div className="scanline" />}
            {size && result?.detections.map((d, i) => {
              const [x1, y1, x2, y2] = d.bbox;
              return (
                <div key={i} className={`box ${statusOf(d)}`} style={{
                  left: `${(x1 / size.w) * 100}%`, top: `${(y1 / size.h) * 100}%`,
                  width: `${((x2 - x1) / size.w) * 100}%`, height: `${((y2 - y1) / size.h) * 100}%`,
                }}>
                  <span className="tag">{d.plate_text_raw || "?"} · {(d.ocr_confidence * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {result && (
        <section id="results" className="results">
          {result.num_plates === 0 && (
            <div className="alert warn">No license plate detected in this image.</div>
          )}
          {size && result.detections.map((d, i) => (
            <article key={i} className={`card ${statusOf(d)} fade-up`} style={{ animationDelay: `${i * 0.1}s` }}>
              <PlateCrop src={preview} bbox={d.bbox} size={size} />
              <div className="card-body">
                <div className="plate" dir="auto">{d.display || "—"}</div>
                <div className="badges">
                  <span className={`badge ${statusOf(d)}`}>{STATUS_LABEL[statusOf(d)]}</span>
                  {d.format && <span className="badge">{d.format}</span>}
                </div>
                <Meter label="Detection" value={d.detection_confidence} />
                <Meter label="OCR" value={d.ocr_confidence} />
                <small>raw: {d.plate_text_raw || "—"}</small>
              </div>
            </article>
          ))}
          <div className="timings">
            {Object.entries(result.timings_ms).map(([k, v]) => (
              <div key={k}><strong>{v}</strong><span>{k} ms</span></div>
            ))}
          </div>
        </section>
      )}

      <footer className="foot">PlateReco · YOLO11n + CRNN · FastAPI + React</footer>
    </div>
  );
}