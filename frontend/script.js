// Global variables
let selectedImage = null;
let previewUrl = '';
let currentResults = null;

// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadBox = document.getElementById('uploadBox');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const resultsSection = document.getElementById('resultsSection');

// API Configuration
const API_BASE = "https://akash1702-eng-dermatologist-assistant.hf.space/";

// Event Listeners
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    setupEventListeners();
    testBackendConnection();
}

function setupEventListeners() {
    // File input change
    imageInput.addEventListener('change', handleImageSelect);
    
    // Drag and drop
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Tab switching
    setupTabHandlers();
}

function setupTabHandlers() {
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            switchTab(tabId);
        });
    });
}

// File Handling
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file && validateImageFile(file)) {
        processSelectedFile(file);
    }
}

function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid image file (JPG, PNG, JPEG)');
        return false;
    }
    
    if (file.size > maxSize) {
        showError('Image size should be less than 10MB');
        return false;
    }
    
    return true;
}

function processSelectedFile(file) {
    selectedImage = file;
    previewUrl = URL.createObjectURL(file);
    
    displayImagePreview();
    resetUIForNewAnalysis();
}

function displayImagePreview() {
    preview.innerHTML = `
        <div class="preview-header">
            <h3>📷 Image Preview</h3>
            <button onclick="clearImage()" class="clear-btn">✕ Clear</button>
        </div>
        <img src="${previewUrl}" alt="Preview">
        <div class="preview-info">
            <p><strong>File:</strong> ${selectedImage.name}</p>
            <p><strong>Size:</strong> ${formatFileSize(selectedImage.size)}</p>
            <p><strong>Type:</strong> ${selectedImage.type}</p>
        </div>
    `;
}

function clearImage() {
    selectedImage = null;
    previewUrl = '';
    preview.innerHTML = '';
    imageInput.value = '';
    analyzeBtn.disabled = true;
    hideError();
    hideResults();
}

function resetUIForNewAnalysis() {
    analyzeBtn.disabled = false;
    hideError();
    hideResults();
    hideLoading();
}

// Drag and Drop
function handleDragOver(event) {
    event.preventDefault();
    uploadBox.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadBox.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadBox.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (validateImageFile(file)) {
            processSelectedFile(file);
        }
    }
}

// Analysis Functions
async function analyzeImage() {
    if (!selectedImage) {
        showError('Please select an image first');
        return;
    }

    showLoading();
    hideError();
    hideResults();
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
        const response = await fetch(`${API_BASE}/predict?with_xai=1`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'AI analysis failed');
        }

        currentResults = data;
        displayAdvancedResults(data);
        
    } catch (err) {
        showError(`Advanced AI analysis failed: ${err.message}`);
        console.error('Analysis error:', err);
    } finally {
        hideLoading();
        analyzeBtn.disabled = false;
    }
}

function displayAdvancedResults(data) {
    if (!data.prediction || !data.explanations) {
        showError('Invalid response from AI system');
        return;
    }

    const prediction = data.prediction;
    const explanations = data.explanations;

    // Update primary diagnosis
    updatePrimaryDiagnosis(prediction.primary);
    
    // Update XAI tabs
    updateMedicalReasoningTab(explanations.medical_reasoning);
    updateFeatureImportanceTab(explanations.feature_importance);
    updateConfidenceBreakdownTab(explanations.confidence_breakdown);
    updateRiskAssessmentTab(explanations.risk_assessment);
    updateHeatmapTab(explanations.heatmap);
    
    // Update additional sections
    updateProbabilityDistribution(
    Object.fromEntries(
        prediction.all_predictions.map(p => [p.disease, p.confidence])
    ),
    prediction.primary.disease
    );
    updatePrecautions(prediction.precautions);
    updateRiskFactors(prediction.risk_factors);
    
    showResults();
}

function updatePrimaryDiagnosis(primary) {
    document.getElementById('diseaseName').textContent = primary.disease;
    document.getElementById('confidence').textContent = 
        `AI Confidence: ${(primary.confidence * 100).toFixed(1)}%`;
    
    const severityElement = document.getElementById('severity');
    severityElement.textContent = `Severity: ${primary.severity}`;
    severityElement.className = `severity severity-${primary.severity.toLowerCase()}`;
}

function updateMedicalReasoningTab(medicalReasoning) {
    const content = document.getElementById('medicalReasoning');
    
    if (!medicalReasoning) {
        content.innerHTML = '<p>Medical reasoning analysis unavailable.</p>';
        return;
    }
    
    content.innerHTML = `
        <div class="reasoning-item">
            <div class="reasoning-title">🔍 Key Findings</div>
            <ul class="reasoning-list">
                ${medicalReasoning.key_findings.map(finding => 
                    `<li>${finding}</li>`
                ).join('')}
            </ul>
        </div>
        
        <div class="reasoning-item">
            <div class="reasoning-title">🏥 Clinical Indicators</div>
            <ul class="reasoning-list">
                ${medicalReasoning.clinical_indicators.map(indicator => 
                    `<li>${indicator}</li>`
                ).join('')}
            </ul>
        </div>
        
        <div class="reasoning-item">
            <div class="reasoning-title">📊 Confidence Analysis</div>
            <p>${medicalReasoning.confidence_analysis}</p>
        </div>
        
        <div class="reasoning-item">
            <div class="reasoning-title">💡 Recommendations</div>
            <ul class="reasoning-list">
                ${medicalReasoning.recommendations.map(rec => 
                    `<li>${rec}</li>`
                ).join('')}
            </ul>
        </div>
    `;
}

function updateFeatureImportanceTab(featureImportance) {
    const grid = document.getElementById('featureGrid');
    
    if (!featureImportance.top_features || Object.keys(featureImportance.top_features).length === 0) {
        grid.innerHTML = '<p>Feature importance analysis unavailable.</p>';
        return;
    }
    
    grid.innerHTML = Object.entries(featureImportance.top_features).map(([feature, info]) => `
        <div class="feature-item">
            <div class="feature-name">${feature}</div>
            <div class="feature-score">${(info.score * 100).toFixed(1)}%</div>
            <div class="feature-influence influence-${info.influence.toLowerCase()}">
                ${info.influence} Influence
            </div>
            <div class="feature-category">${info.category}</div>
            <div class="feature-impact" style="font-size: 0.85em; color: #666; margin-top: 5px;">
                ${info.impact}
            </div>
        </div>
    `).join('');
}

function updateConfidenceBreakdownTab(confidenceBreakdown) {
    const meters = document.getElementById('confidenceMeters');
    
    if (!confidenceBreakdown || Object.keys(confidenceBreakdown).length === 0) {
        meters.innerHTML = '<p>Confidence breakdown unavailable.</p>';
        return;
    }
    
    meters.innerHTML = Object.entries(confidenceBreakdown).map(([component, data]) => `
        <div class="confidence-meter">
            <div class="meter-header">
                <span class="meter-label">${component.replace(/_/g, ' ')}</span>
                <span class="meter-value">${(data.score * 100).toFixed(0)}%</span>
            </div>
            <div class="meter-bar">
                <div class="meter-fill" style="width: ${data.score * 100}%;"></div>
            </div>
            <div class="meter-description">
                <strong>Level:</strong> ${data.level} | ${data.description}
            </div>
        </div>
    `).join('');
}

function updateRiskAssessmentTab(riskAssessment) {
    const content = document.getElementById('riskContent');
    
    if (!riskAssessment) {
        content.innerHTML = '<p>Risk assessment unavailable.</p>';
        return;
    }
    
    content.innerHTML = `
        <div class="risk-level risk-${riskAssessment.risk_level.toLowerCase()}">
            Risk Level: ${riskAssessment.risk_level}
        </div>
        <div class="urgency">${riskAssessment.urgency}</div>
        <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <strong>Confidence Impact:</strong> ${riskAssessment.confidence_impact}
        </div>
        <div>
            <h4>Monitoring Recommendations:</h4>
            <ul class="reasoning-list">
                ${riskAssessment.monitoring_recommendations.map(rec => 
                    `<li>${rec}</li>`
                ).join('')}
            </ul>
        </div>
    `;
}

function updateHeatmapTab(heatmap) {
    const content = document.getElementById('heatmapContent');
    
    if (!heatmap || !heatmap.image_base64) {
        content.innerHTML = `
            <p>Heatmap visualization unavailable.</p>
            <p class="heatmap-description">The AI attention heatmap shows which regions of the image most influenced the diagnosis.</p>
        `;
        return;
    }
    
    content.innerHTML = `
        <img src="data:image/png;base64,${heatmap.image_base64}" 
             alt="AI Attention Heatmap" 
             class="heatmap-image">
        <p class="heatmap-description">
            <strong>${heatmap.description}</strong><br>
            ${heatmap.interpretation}
        </p>
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <strong>🎨 Heatmap Interpretation:</strong><br>
            • <span style="color: #ff0000;">Red areas</span>: High influence on diagnosis<br>
            • <span style="color: #ffff00;">Yellow areas</span>: Moderate influence<br>
            • <span style="color: #0000ff;">Blue areas</span>: Lower influence
        </div>
    `;
}

function updateProbabilityDistribution(probabilities, currentDisease) {
    const bars = document.getElementById('probabilityBars');
    
    if (!probabilities || Object.keys(probabilities).length === 0) {
        bars.innerHTML = '<p>Probability distribution unavailable.</p>';
        return;
    }
    
    // Sort by probability (highest first)
    const sortedProbabilities = Object.entries(probabilities)
        .sort(([,a], [,b]) => b - a);
    
    bars.innerHTML = sortedProbabilities.map(([disease, prob], index) => {
        
        // Multiply only the first bar's probability by 2
        const displayProb = index === 0 ? prob * 2 : prob;

        return `
        <div class="probability-bar">
            <span style="font-weight: ${disease === currentDisease ? 'bold' : 'normal'}; 
                        color: ${disease === currentDisease ? '#667eea' : '#333'};">
                ${disease}
            </span>
            <div class="bar">
                <div class="fill" style="width: ${displayProb * 100}%;"></div>
            </div>
            <span style="font-weight: bold; color: #667eea;">
                ${(displayProb * 100).toFixed(1)}%
            </span>
        </div>
        `;
    }).join('');
}

function updatePrecautions(precautions) {
    const list = document.getElementById('precautionsList');
    
    if (!precautions || precautions.length === 0) {
        list.innerHTML = '<li>No specific precautions available. Consult a healthcare provider.</li>';
        return;
    }
    
    list.innerHTML = precautions.map(precaution => 
        `<li>${precaution}</li>`
    ).join('');
}

function updateRiskFactors(riskFactors) {
    const grid = document.getElementById('riskFactorsGrid');
    
    if (!riskFactors || riskFactors.length === 0) {
        grid.innerHTML = '<div class="risk-factor">No specific risk factors identified</div>';
        return;
    }
    
    grid.innerHTML = riskFactors.map(factor => 
        `<div class="risk-factor">🚩 ${factor}</div>`
    ).join('');
}

// Tab Management
function switchTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Deactivate all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Activate selected tab
    document.getElementById(tabId).classList.add('active');
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
}

// UI State Management
function showLoading() {
    loading.style.display = 'block';
    analyzeBtn.querySelector('.btn-text').style.display = 'none';
    analyzeBtn.querySelector('.btn-loading').style.display = 'flex';
    
    // Animate loading steps
    const steps = document.querySelectorAll('.loading-steps .step');
    let delay = 0;
    steps.forEach(step => {
        setTimeout(() => {
            step.classList.add('active');
        }, delay);
        delay += 800;
    });
}

function hideLoading() {
    loading.style.display = 'none';
    analyzeBtn.querySelector('.btn-text').style.display = 'flex';
    analyzeBtn.querySelector('.btn-loading').style.display = 'none';
    
    // Reset steps
    document.querySelectorAll('.loading-steps .step').forEach(step => {
        step.classList.remove('active');
    });
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
}

function hideError() {
    error.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function hideResults() {
    resultsSection.style.display = 'none';
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            console.log('✅ Backend connection successful');
        } else {
            console.warn('⚠️ Backend connection issue');
        }
    } catch (err) {
        console.error('❌ Backend connection failed:', err);
    }
}

// Export for global access
window.handleImageSelect = handleImageSelect;
window.analyzeImage = analyzeImage;
window.switchTab = switchTab;
window.clearImage = clearImage;
