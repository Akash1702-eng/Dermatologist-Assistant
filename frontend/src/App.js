import React, { useState } from 'react';
import './App.css';

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
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', image);

      const response = await fetch('http://localhost:5000/predict?with_xai=1', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
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
      <div className="container">
        <header className="header">
          <h1>🏥 Dermatology AI Assistant</h1>
          <p>AI-powered skin analysis with explainable insights</p>
        </header>

        <div className="main-content">
          {/* Upload Section */}
          <div className="upload-section">
            <div className="upload-box">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                id="image-input"
              />
              <label htmlFor="image-input">
                {preview ? '✓ Image Selected' : '📷 Click to upload image'}
              </label>
            </div>

            {preview && (
              <div className="preview">
                <img src={preview} alt="Preview" />
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className="analyze-btn"
            >
              {loading ? '🔄 Analyzing...' : '🔍 Analyze'}
            </button>

            {error && <div className="error">{error}</div>}
          </div>

          {/* Results Section */}
          {result && (
            <div className="results-section">
              <div className="result-card">
                <h2>Analysis Results</h2>

                {/* Prediction */}
                <div className="prediction-box">
                  <div className="disease-name">{result.disease}</div>
                  <div className="confidence">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </div>
                  <div className={`severity severity-${result.severity.toLowerCase()}`}>
                    Severity: {result.severity}
                  </div>
                </div>

                {/* Explanation */}
                <div className="explanation-box">
                  <h3>💡 Why This Prediction?</h3>
                  <p>{result.explanation}</p>
                </div>

                {/* Feature Importance */}
                <div className="features-box">
                  <h3>📊 Key Features Analyzed</h3>
                  <div className="features-list">
                    {result.feature_importance.map((feature, idx) => (
                      <div key={idx} className="feature-item">
                        <span className="feature-name">{feature.name}</span>
                        <span className="feature-value">{feature.value}</span>
                        <span className={`feature-influence influence-${feature.influence.toLowerCase()}`}>
                          {feature.influence}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Heatmap */}
                <div className="heatmap-box">
                  <h3>🔥 Visual Explanation (Saliency Map)</h3>
                  <p className="heatmap-explanation">
                    Red areas show where the AI focused to make this diagnosis
                  </p>
                  <img src={result.heatmap} alt="Saliency Map" className="heatmap" />
                </div>

                {/* Probabilities */}
                <div className="probabilities-box">
                  <h3>📈 Prediction Probabilities</h3>
                  {Object.entries(result.probabilities).map(([disease, prob]) => (
                    <div key={disease} className="probability-bar">
                      <span>{disease}</span>
                      <div className="bar">
                        <div className="fill" style={{width: `${prob * 100}%`}}></div>
                      </div>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>

                {/* Precautions */}
                <div className="precautions-box">
                  <h3>✅ Recommended Precautions</h3>
                  <ul>
                    {result.precautions.map((precaution, idx) => (
                      <li key={idx}>{precaution}</li>
                    ))}
                  </ul>
                </div>

                {/* Disclaimer */}
                <div className="disclaimer">
                  ⚠️ {result.disclaimer}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;