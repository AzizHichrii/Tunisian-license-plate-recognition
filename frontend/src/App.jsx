import { useState } from "react";
import { predict } from "./api";
import "./App.css";

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

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [size, setSize] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onSelect = (f) => {
    if (!f) return;
    setFile(f); setResult(null); setError(""); setSize(null);
    setPreview(URL.createObjectURL(f));
  };

  const onAnalyze = async () => {
    setLoading(true); setError("");
    try { setResult(await predict(file)); }
    catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <main className="app">
      <header>
        <h1>Tunisian License Plate Recognition</h1>
        <p>YOLO11n detection + CRNN reading</p>
      </header>

      <label className="dropzone"
             onDragOver={(e) => e.preventDefault()}
             onDrop={(e) => { e.preventDefault(); onSelect(e.dataTransfer.files[0]); }}>
        <input type="file" accept="image/jpeg,image/png" hidden
               onChange={(e) => onSelect(e.target.files[0])} />
        {file ? file.name : "Drop a car image here or click to choose (JPG / PNG)"}
      </label>

      <button disabled={!file || loading} onClick={onAnalyze}>
        {loading ? "Analyzing…" : "Analyze"}
      </button>

      {error && <div className="alert error">{error}</div>}

      {preview && (
        <div className="viewer">
          <img src={preview} alt="uploaded"
               onLoad={(e) => setSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })} />
          {size && result?.detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.bbox;
            return <div key={i} className={`box ${statusOf(d)}`} style={{
              left: `${(x1 / size.w) * 100}%`, top: `${(y1 / size.h) * 100}%`,
              width: `${((x2 - x1) / size.w) * 100}%`, height: `${((y2 - y1) / size.h) * 100}%`,
            }} />;
          })}
        </div>
      )}

      {result && result.num_plates === 0 && (
        <div className="alert warn">No license plate detected in this image.</div>
      )}

      {result && size && result.detections.map((d, i) => (
        <section key={i} className={`card ${statusOf(d)}`}>
          <PlateCrop src={preview} bbox={d.bbox} size={size} />
          <div>
            <div className="plate" dir="auto">{d.display || "—"}</div>
            <div className="badges">
              <span className={`badge ${statusOf(d)}`}>
                {statusOf(d) === "valid" ? "Valid" : statusOf(d) === "uncertain" ? "Uncertain" : "Invalid format"}
              </span>
              {d.format && <span className="badge">{d.format}</span>}
            </div>
            <small>
              Detection {(d.detection_confidence * 100).toFixed(1)}% · OCR {(d.ocr_confidence * 100).toFixed(1)}% · raw: {d.plate_text_raw || "—"}
            </small>
          </div>
        </section>
      ))}

      {result && (
        <footer>
          Detection {result.timings_ms.detection} ms · OCR {result.timings_ms.ocr} ms · Total {result.timings_ms.total} ms
        </footer>
      )}
    </main>
  );
}