/**
 * =============================================================================
 * CardioPredict - Frontend Client Controller
 * =============================================================================
 * Handles:
 * 1. Form input gathering & client-side validation
 * 2. Communication with the Flask ML REST API via fetch()
 * 3. Rendering dynamic prediction results, badges, and animated probability bars
 * 4. Error handling with friendly troubleshooting instructions
 * 5. Tab navigation and quick test sample presets
 * =============================================================================
 */

// Target API endpoint on Flask server
const API_BASE_URL = "http://127.0.0.1:5000";
const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/`;

// Clinical feature keys expected by the Flask ML pipeline
const FEATURE_KEYS = [
  "age", "sex", "cp", "trestbps", "chol", "fbs", 
  "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
];

// Sample test profiles based on validated Cleveland records
const SAMPLE_PROFILES = {
  healthy: {
    age: 45,
    sex: "0",         // Female
    cp: "1",          // Typical Angina
    trestbps: 120,    // Normal BP
    chol: 180,        // Healthy cholesterol
    fbs: "0",         // Normal glucose (<= 120 mg/dl)
    restecg: "0",     // Normal ECG
    thalach: 175,     // High max heart rate
    exang: "0",       // No exercise angina
    oldpeak: 0.0,     // No ST depression
    slope: "1",       // Upsloping
    ca: "0",          // 0 major vessels
    thal: "3"         // Normal perfusion
  },
  atRisk: {
    age: 67,
    sex: "1",         // Male
    cp: "4",          // Asymptomatic (severe risk category)
    trestbps: 160,    // Stage 2 hypertension
    chol: 286,        // High cholesterol
    fbs: "0",         // <= 120 mg/dl
    restecg: "2",     // Left ventricular hypertrophy
    thalach: 108,     // Blunted max heart rate
    exang: "1",       // Exercise induced angina
    oldpeak: 1.5,     // Significant ST depression
    slope: "2",       // Flat ST slope
    ca: "3",          // 3 vessels colored by fluoroscopy
    thal: "3"         // Normal / fixed defect
  }
};

// DOM Elements
const heartForm = document.getElementById("heartForm");
const resetFormBtn = document.getElementById("resetFormBtn");
const loadHealthySampleBtn = document.getElementById("loadHealthySampleBtn");
const loadAtRiskSampleBtn = document.getElementById("loadAtRiskSampleBtn");

// Result card state containers
const initialView = document.getElementById("initialView");
const loadingView = document.getElementById("loadingView");
const errorView = document.getElementById("errorView");
const resultView = document.getElementById("resultView");

const errorTitle = document.getElementById("errorTitle");
const errorMessageText = document.getElementById("errorMessageText");
const errorRetryBtn = document.getElementById("errorRetryBtn");

// Outcome display elements
const outcomeBox = document.getElementById("outcomeBox");
const outcomeIconCircle = document.getElementById("outcomeIconCircle");
const outcomeRiskBadge = document.getElementById("outcomeRiskBadge");
const outcomeHeading = document.getElementById("outcomeHeading");

// Probability bars & metrics
const noDiseasePercent = document.getElementById("noDiseasePercent");
const noDiseaseProgress = document.getElementById("noDiseaseProgress");
const diseasePercent = document.getElementById("diseasePercent");
const diseaseProgress = document.getElementById("diseaseProgress");

const rawClassVal = document.getElementById("rawClassVal");
const modelConfidenceVal = document.getElementById("modelConfidenceVal");
const riskCategoryVal = document.getElementById("riskCategoryVal");
const clinicalAdviceText = document.getElementById("clinicalAdviceText");

// API Status Chip
const apiStatusChip = document.getElementById("apiStatusChip");
const apiStatusLabel = document.getElementById("apiStatusLabel");

/**
 * Switch Result Card View States
 * @param {'initial'|'loading'|'error'|'result'} state 
 */
function setResultState(state) {
  initialView.style.display = (state === 'initial') ? 'block' : 'none';
  loadingView.style.display = (state === 'loading') ? 'block' : 'none';
  errorView.style.display = (state === 'error') ? 'block' : 'none';
  resultView.style.display = (state === 'result') ? 'block' : 'none';
}

/**
 * Periodically check if the Flask ML API is online
 */
async function checkServerHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, { method: "GET" });
    if (response.ok) {
      apiStatusChip.className = "api-status-chip online";
      apiStatusLabel.textContent = "Flask API: Online";
      return true;
    } else {
      throw new Error(`Status ${response.status}`);
    }
  } catch (err) {
    apiStatusChip.className = "api-status-chip offline";
    apiStatusLabel.textContent = "Flask API: Offline";
    return false;
  }
}

/**
 * Populate form inputs from a preset object
 * @param {Object} profile 
 */
function applySampleProfile(profile) {
  FEATURE_KEYS.forEach(key => {
    const input = document.getElementById(key);
    if (input && profile[key] !== undefined) {
      input.value = profile[key];
      // Visual feedback highlight
      input.style.borderColor = "var(--primary-blue)";
      setTimeout(() => {
        input.style.borderColor = "";
      }, 600);
    }
  });
}

/**
 * Collect and validate all 13 clinical biomarkers from the form
 * Returns an object with numerical floats or throws an Error
 */
function extractFormData() {
  const payload = {};
  const emptyFields = [];

  FEATURE_KEYS.forEach(key => {
    const input = document.getElementById(key);
    if (!input || input.value.trim() === "") {
      emptyFields.push(key);
    } else {
      const val = parseFloat(input.value.trim());
      if (isNaN(val)) {
        emptyFields.push(key);
      } else {
        payload[key] = val;
      }
    }
  });

  if (emptyFields.length > 0) {
    throw new Error(`Please fill in all clinical fields with valid numbers. Missing or invalid: ${emptyFields.join(", ")}`);
  }

  return payload;
}

/**
 * Submit data to Flask ML backend and render prediction
 */
async function handleFormSubmit(event) {
  if (event) event.preventDefault();

  let payload;
  try {
    payload = extractFormData();
  } catch (validationErr) {
    alert(validationErr.message);
    return;
  }

  // Switch to loading state
  setResultState('loading');

  try {
    // Send POST request with JSON payload to Flask backend
    const response = await fetch(PREDICT_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => null);
      const msg = (errorJson && errorJson.error) ? errorJson.error : `Server returned status ${response.status}`;
      throw new Error(msg);
    }

    const data = await response.json();
    console.log("[CardioPredict] Inference response:", data);

    // Render results received strictly from Flask Random Forest model
    renderPredictionResult(data);

  } catch (err) {
    console.error("[CardioPredict] Error submitting prediction:", err);
    errorTitle.textContent = "Prediction Request Failed";
    errorMessageText.textContent = err.message || "Failed to communicate with Flask backend API.";
    setResultState('error');
  }
}

/**
 * Display the returned predictions on the Result Card
 * @param {Object} data - Prediction payload from Flask
 */
function renderPredictionResult(data) {
  const { prediction, result, risk, no_disease_probability, disease_probability } = data;

  const isLowRisk = (prediction === 0);

  // 1. Outcome badge styling & icons
  if (isLowRisk) {
    outcomeBox.className = "outcome-box low-risk";
    outcomeIconCircle.textContent = "✓";
    outcomeRiskBadge.textContent = risk || "Lower Risk";
    outcomeHeading.textContent = result || "No Heart Disease";
    
    clinicalAdviceText.textContent = 
      "The Support Vector Machine (SVM) model computed a high probability of healthy coronary indicators. Maintain a balanced diet, exercise routine, and regular wellness visits.";
  } else {
    outcomeBox.className = "outcome-box high-risk";
    outcomeIconCircle.textContent = "!";
    outcomeRiskBadge.textContent = risk || "Higher Risk";
    outcomeHeading.textContent = result || "Heart Disease Detected";
    
    clinicalAdviceText.textContent = 
      "Biomarkers reflect elevated likelihood of coronary artery narrowing. Follow-up consultation, ECG monitoring, and coronary angiography are advised by cardiology guidelines.";
  }

  // 2. Progress bars animation
  const noDiseaseVal = Math.max(0, Math.min(100, no_disease_probability));
  const diseaseVal = Math.max(0, Math.min(100, disease_probability));

  noDiseasePercent.textContent = `${noDiseaseVal.toFixed(1)}%`;
  diseasePercent.textContent = `${diseaseVal.toFixed(1)}%`;

  // Animate width transition
  noDiseaseProgress.style.width = "0%";
  diseaseProgress.style.width = "0%";

  setTimeout(() => {
    noDiseaseProgress.style.width = `${noDiseaseVal}%`;
    diseaseProgress.style.width = `${diseaseVal}%`;
  }, 50);

  // 3. Mini summary tiles
  rawClassVal.textContent = isLowRisk ? "Class 0 (Negative)" : "Class 1 (Positive)";
  const confidence = isLowRisk ? noDiseaseVal : diseaseVal;
  modelConfidenceVal.textContent = `${confidence.toFixed(1)}%`;
  riskCategoryVal.textContent = risk;

  // Show result state
  setResultState('result');
}

/**
 * Tab Navigation Controller
 * @param {string} targetTabId 
 */
function switchNavTab(targetTabId) {
  if (targetTabId === "biomarkers-link") {
    // Show predict tab and smooth-scroll down to the embedded Biomarkers Guide
    switchNavTab("predict-tab");
    setTimeout(() => {
      const guideEl = document.getElementById("biomarkers-guide-section");
      if (guideEl) {
        guideEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }, 50);

    // Keep Biomarkers Guide button highlighted in sidebar
    document.querySelectorAll(".nav-btn").forEach(btn => {
      if (btn.getAttribute("data-tab") === "biomarkers-link") {
        btn.classList.add("active");
      }
    });
    return;
  }

  // Update nav buttons
  document.querySelectorAll(".nav-btn").forEach(btn => {
    if (btn.getAttribute("data-tab") === targetTabId) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Update tab panels
  document.querySelectorAll(".tab-panel").forEach(panel => {
    if (panel.id === targetTabId) {
      panel.style.display = "block";
    } else {
      panel.style.display = "none";
    }
  });
}

// Attach event listeners
document.addEventListener("DOMContentLoaded", () => {
  // Navigation tabs
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      switchNavTab(tabId);
    });
  });

  // Form submission
  heartForm.addEventListener("submit", handleFormSubmit);

  // Reset button
  resetFormBtn.addEventListener("click", () => {
    heartForm.reset();
    setResultState('initial');
  });

  // Preset sample buttons
  loadHealthySampleBtn.addEventListener("click", () => {
    applySampleProfile(SAMPLE_PROFILES.healthy);
  });

  loadAtRiskSampleBtn.addEventListener("click", () => {
    applySampleProfile(SAMPLE_PROFILES.atRisk);
  });

  // Scroll to 13 Parameters Guide button
  const scrollToGuideBtn = document.getElementById("scrollToGuideBtn");
  if (scrollToGuideBtn) {
    scrollToGuideBtn.addEventListener("click", () => {
      const guideEl = document.getElementById("biomarkers-guide-section");
      if (guideEl) {
        guideEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  }

  // Retry button
  errorRetryBtn.addEventListener("click", () => {
    handleFormSubmit();
  });

  // Initial health check and periodic heartbeat every 15s
  checkServerHealth();
  setInterval(checkServerHealth, 15000);
});
