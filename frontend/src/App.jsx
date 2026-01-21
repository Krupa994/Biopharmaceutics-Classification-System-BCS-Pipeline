import { useState } from "react";
import axios from "axios";
import "./App.css";

const classDescriptions = {
  I: "High Solubility, High Permeability",
  II: "Low Solubility, High Permeability",
  III: "High Solubility, Low Permeability",
  IV: "Low Solubility, Low Permeability"
};

function safeFixed(v, digits = 3) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return String(v);
  return n.toFixed(digits);
}

export default function App() {
  const [smiles, setSmiles] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handlePredict = async () => {
    if (!smiles.trim()) {
      setErrorMsg("Please enter a SMILES string.");
      return;
    }

    setLoading(true);
    setErrorMsg("");
    setResult(null);

    try {
      const res = await axios.post(
        "http://localhost:8000/predict_logS",
        { smiles },
        { timeout: 20000 }
      );
      console.log("API response:", res.data);
      if (!res.data || typeof res.data !== "object") {
        throw new Error("Invalid response from server");
      }
      setResult(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
      const serverMsg = err?.response?.data?.error || err.message;
      setErrorMsg("Prediction failed: " + serverMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      {/* --- Intro Section (your original content) --- */}
      <section className="intro">
        <h1>Biopharmaceutics Classification System (BCS)</h1>
        <p>
          The <strong>BCS Classification</strong> is a scientific framework that categorizes drugs
          based on their <strong>solubility</strong> and <strong>intestinal permeability</strong>.
          It helps in predicting drug absorption and guiding formulation strategies.
        </p>

        <h2>Why is BCS Important?</h2>
        <p>
          BCS plays a crucial role in pharmaceutical development and regulatory approvals.
          It assists in <em>drug discovery, bioequivalence studies, and formulation design</em>.
          Understanding the class of a drug helps in determining whether additional
          bioavailability studies are required.
        </p>

        <h2>The Four BCS Classes</h2>
        <div className="classes">
          <div className="class-card">
            <h3>Class I</h3>
            <p>{classDescriptions.I}</p>
            <small>Drugs are well absorbed with minimal formulation challenges.</small>
          </div>

          <div className="class-card">
            <h3>Class II</h3>
            <p>{classDescriptions.II}</p>
            <small>Absorption depends on improving solubility.</small>
          </div>

          <div className="class-card">
            <h3>Class III</h3>
            <p>{classDescriptions.III}</p>
            <small>Absorption depends on permeability enhancement strategies.</small>
          </div>

          <div className="class-card">
            <h3>Class IV</h3>
            <p>{classDescriptions.IV}</p>
            <small>Poor absorption; formulation is highly challenging.</small>
          </div>
        </div>
      </section>

      {/* --- Prediction Tool Section (robust) --- */}
      <section className="tool" style={{ marginTop: 40 }}>
        <h2>Try the BCS Prediction Tool</h2>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handlePredict();
          }}
          style={{ display: "flex", gap: 12, marginTop: 12 }}
        >
          <input
            type="text"
            placeholder="Enter SMILES string"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            style={{ flex: 1, padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "10px 16px",
              borderRadius: 6,
              border: "none",
              background: loading ? "#95a5a6" : "#3498db",
              color: "white",
              cursor: loading ? "not-allowed" : "pointer"
            }}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {errorMsg && (
          <div style={{ color: "crimson", marginTop: 12 }}>{errorMsg}</div>
        )}

        {result ? (
          <div className="result" style={{ marginTop: 18 }}>
            <h2>Predictions</h2>

            <p>
              <strong>Predicted logS:</strong> {safeFixed(result.logS)}
            </p>
            <p>
              <strong>logS explanation:</strong>{" "}
              {result.logS_explanation ?? result.explanation ?? "—"}
            </p>

            <p>
              <strong>Predicted logP:</strong> {safeFixed(result.logP)}
            </p>

            <h3>Predicted BCS Class</h3>
            <p>
              {Array.isArray(result.class) && result.class.length > 0
                ? result.class
                    .map((c) => `${c} (${classDescriptions[c] ?? "No desc"})`)
                    .join(", ")
                : (result.cb_predicted_class ?? result.predicted_class ?? "Uncertain")}
            </p>
            <p>
              <strong>Class explanation:</strong>{" "}
              {result.class_explanation ?? result.cb_predicted_class ?? "—"}
            </p>

            <h3>CatBoost Explanation (top descriptors)</h3>
            {Array.isArray(result.cb_top_descriptors) && result.cb_top_descriptors.length > 0 ? (
              <ul>
                {result.cb_top_descriptors.map((d, i) => (
                  <li key={i}>
                    <strong>{d.name}</strong> ({d.meaning ?? ""}) — value: {safeFixed(d.value)} contribution: {safeFixed(d.contribution)}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No CatBoost top-descriptors returned.</p>
            )}

            <h3>Molecular Descriptors</h3>
            {result.Values && typeof result.Values === "object" ? (
              <table className="descriptor-table" style={{ width: "100%", marginTop: 8 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: 8 }}>Descriptor</th>
                    <th style={{ textAlign: "left", padding: 8 }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.Values).map(([k, v]) => (
                    <tr key={k}>
                      <td style={{ padding: 8 }}>{k}</td>
                      <td style={{ padding: 8 }}>{safeFixed(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No descriptor values returned.</p>
            )}
          </div>
        ) : (
          <div style={{ marginTop: 18, color: "#666" }}>
            {loading ? "Waiting for results..." : "Enter a SMILES and click Predict."}
          </div>
        )}
      </section>
    </div>
  );
}
