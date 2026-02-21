import React, { useState } from "react";
import "./App.css";

function App() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            const reader = new FileReader();
            reader.onload = (ev) => setPreview(ev.target.result);
            reader.readAsDataURL(file);
        }
    };

    const handleAnalyze = async () => {
        if (!image) {
            setError("Please select an image");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("image", image);

            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Prediction failed");
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
            <header>
                <h1>🩺 Dermatology AI Assistant</h1>
                <p>Analyze skin lesions with AI-powered insights</p>
            </header>

            <div className="upload-section">
                <input type="file" accept="image/*" onChange={handleImageUpload} />
                {preview && (
                    <div className="preview">
                        <img src={preview} alt="Preview" />
                    </div>
                )}
                <button onClick={handleAnalyze} disabled={!image || loading}>
                    {loading ? "Analyzing..." : "Analyze"}
                </button>
                {error && <p className="error">{error}</p>}
            </div>

            {result && (
                <div className="results">
                    <h2>Prediction</h2>
                    <p>
                        <strong>Disease:</strong>{" "}
                        {result.prediction.primary.disease || "Unknown"}
                    </p>
                    <p>
                        <strong>Confidence:</strong>{" "}
                        {(result.prediction.primary.confidence * 100).toFixed(1)}%
                    </p>
                    <p>
                        <strong>Severity:</strong> {result.prediction.primary.severity}
                    </p>

                    <h3>Features Analyzed</h3>
                    {result.analysis && (
                        <div className="features">
                            <ul>
                                {Object.entries(result.analysis.features_analyzed).map(
                                    ([k, v]) => (
                                        <li key={k}>
                                            <strong>{k}:</strong> {v.toFixed ? v.toFixed(3) : v}
                                        </li>
                                    )
                                )}
                            </ul>
                        </div>
                    )}

                    <h3>Recommendations</h3>
                    {result.recommendations && (
                        <div className="recommendations">
                            <p>{result.recommendations.description}</p>
                            <ul>
                                {(result.recommendations.precautions || []).map(
                                    (p, idx) => (
                                        <li key={idx}>{p}</li>
                                    )
                                )}
                            </ul>
                            <p>
                                <strong>Urgency:</strong> {result.recommendations.urgency_level}
                            </p>
                        </div>
                    )}

                    {result.explanations && result.explanations.heatmap && (
                        <div className="heatmap">
                            <h3>AI Heatmap / Saliency</h3>
                            {result.explanations.heatmap.image_base64 ? (
                                <img
                                    src={result.explanations.heatmap.image_base64}
                                    alt="Heatmap"
                                />
                            ) : (
                                <p>{result.explanations.heatmap.description}</p>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
