/**
 * app.js
 * ------
 * Phase 4 | Executed: Local (browser, served by nginx)
 *
 * Handles:
 * - Image upload via file picker and drag-and-drop
 * - Calls POST /predict on FastAPI backend
 * - Displays prediction result with color coding
 * - Feedback submission via POST /feedback
 * - System status check via GET /health and GET /ready
 * - Pipeline dashboard with live metrics
 */

// const API_BASE = window.API_BASE || "http://localhost:8000"; 
const API_BASE = "";  // This is needed for ngrok

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile       = null;
let currentPrediction = null;

// ── DOM references ─────────────────────────────────────────────────────────
const fileInput      = document.getElementById("file-input");
const dropzone       = document.getElementById("dropzone");
const dropzoneInner  = document.getElementById("dropzone-inner");
const previewImg     = document.getElementById("preview-img");
const btnAnalyse     = document.getElementById("btn-analyse");
const resultCard     = document.getElementById("result-card");
const resultHeader   = document.getElementById("result-header");
const resultIcon     = document.getElementById("result-icon");
const resultLabel    = document.getElementById("result-label");
const spinner        = document.getElementById("spinner");
const errorBanner    = document.getElementById("error-banner");
const errorMessage   = document.getElementById("error-message");
const feedbackStatus = document.getElementById("feedback-status");


// ── File selection ───────────────────────────────────────────────────────────

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
});

dropzone.addEventListener("click", (e) => {
    if (e.target.tagName !== "LABEL" && e.target.tagName !== "INPUT") {
        fileInput.click();
    }
});

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");

    const file = e.dataTransfer.files[0];

    if (file) handleFileSelected(file);
});


function handleFileSelected(file) {
    if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
        showError("Only JPEG and PNG images are supported.");
        return;
    }

    currentFile = file;
    hideError();

    const reader = new FileReader();

    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.remove("hidden");
        dropzoneInner.classList.add("hidden");
    };

    reader.readAsDataURL(file);

    btnAnalyse.disabled = false;
}


// ── Analyse ─────────────────────────────────────────────────────────────────

btnAnalyse.addEventListener("click", async () => {
    if (!currentFile) return;
    await runPrediction(currentFile);
});


async function runPrediction(file) {
    showSpinner();
    hideError();
    resultCard.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        currentPrediction = { ...data };

        displayResult(data);

    } catch (err) {
        showError(`Prediction failed: ${err.message}`);

    } finally {
        hideSpinner();
    }
}


function displayResult(data) {
    const isMalignant = data.label === "malignant";

    resultHeader.className = `result-header ${data.label}`;

    resultIcon.textContent =
        isMalignant ? "⚠️" : "✅";

    resultLabel.textContent =
        isMalignant
            ? "Malignant Detected"
            : "Benign — Low Risk";

    document.getElementById("detail-confidence").textContent =
        `${(data.confidence * 100).toFixed(1)}%`;

    document.getElementById("detail-prob").textContent =
        `${(data.malignant_prob * 100).toFixed(1)}%`;

    document.getElementById("detail-threshold").textContent =
        data.threshold_used.toFixed(2);

    document.getElementById("recommendation").textContent =
        data.recommendation;

    feedbackStatus.classList.add("hidden");
    feedbackStatus.textContent = "";

    resultCard.classList.remove("hidden");

    resultCard.scrollIntoView({
        behavior: "smooth"
    });
}


// ── Feedback ─────────────────────────────────────────────────────────────────

async function submitFeedback(trueLabel) {
    if (!currentPrediction) return;

    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image_id:        currentPrediction.image_id,
                predicted_label: currentPrediction.label,
                true_label:      trueLabel
            })
        });

        if (!response.ok) {
            throw new Error("Feedback submission failed.");
        }

        feedbackStatus.textContent =
            "✅ Feedback recorded. Thank you!";

        feedbackStatus.classList.remove("hidden");

        document.querySelectorAll(".btn-feedback").forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = "0.5";
        });

    } catch (err) {
        feedbackStatus.textContent =
            `❌ ${err.message}`;

        feedbackStatus.classList.remove("hidden");
    }
}


// ── Reset ────────────────────────────────────────────────────────────────────

function resetUI() {
    currentFile       = null;
    currentPrediction = null;

    fileInput.value = "";

    previewImg.src = "";
    previewImg.classList.add("hidden");

    dropzoneInner.classList.remove("hidden");

    btnAnalyse.disabled = true;

    resultCard.classList.add("hidden");

    hideError();

    document.querySelectorAll(".btn-feedback").forEach(btn => {
        btn.disabled = false;
        btn.style.opacity = "1";
    });

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


// ── System status check ──────────────────────────────────────────────────────

async function checkSystemStatus() {

    const dotApi =
        document.getElementById("dot-api");

    const dotModel =
        document.getElementById("dot-model");

    const modelVersionText =
        document.getElementById("model-version-text");

    try {

        const res =
            await fetch(`${API_BASE}/health`);

        dotApi.className =
            res.ok
                ? "status-dot green"
                : "status-dot red";

    } catch {

        dotApi.className =
            "status-dot red";
    }


    try {

        const res =
            await fetch(`${API_BASE}/ready`);

        if (res.ok) {

            const data =
                await res.json();

            if (data.model_loaded) {

                dotModel.className =
                    "status-dot green";

                modelVersionText.textContent =
                    `Model: ${data.model_name || "unknown"} | Version: ${data.model_version || "?"}`;

            } else {

                dotModel.className =
                    "status-dot red";

                modelVersionText.textContent =
                    "Model not loaded";
            }

        } else {

            dotModel.className =
                "status-dot red";
        }

    } catch {

        dotModel.className =
            "status-dot red";
    }
}


// ── Tab navigation ───────────────────────────────────────────────────────────

function showTab(tab) {

    document
        .getElementById("tab-screening")
        .classList.add("hidden");

    document
        .getElementById("tab-pipeline")
        .classList.add("hidden");

    document
        .getElementById("tab-" + tab)
        .classList.remove("hidden");

    document
        .querySelectorAll(".tab-btn")
        .forEach(btn => btn.classList.remove("active"));

    event.target.classList.add("active");

    if (tab === "pipeline") {
        fetchPipelineMetrics();
    }
}


// ── Pipeline dashboard metrics ───────────────────────────────────────────────

async function fetchPipelineMetrics() {

    try {

        const res =
            await fetch(`${API_BASE}/metrics`);

        if (!res.ok) return;

        const text =
            await res.text();

        const get = (name) => {

            const lines =
                text
                    .split("\n")
                    .filter(
                        l =>
                            l.startsWith(name) &&
                            !l.startsWith("#")
                    );

            if (!lines.length) return null;

            return lines.reduce((sum, l) => {

                const val =
                    parseFloat(l.split(" ").pop());

                return sum +
                    (isNaN(val) ? 0 : val);

            }, 0);
        };


        const requests =
            get("melanoma_request_total");

        const predictions =
            get("melanoma_prediction_total");

        const feedback =
            get("melanoma_feedback_total");

        const drift =
            get("melanoma_drift_score");

        const recall =
            get("melanoma_real_world_recall");

        const misclass =
            get("melanoma_misclassification_rate");


        document.getElementById("m-requests").textContent =
            requests != null
                ? Math.round(requests)
                : "—";

        document.getElementById("m-predictions").textContent =
            predictions != null
                ? Math.round(predictions)
                : "—";

        document.getElementById("m-feedback").textContent =
            feedback != null
                ? Math.round(feedback)
                : "—";

        document.getElementById("m-drift").textContent =
            drift != null
                ? drift.toFixed(3)
                : "—";

        document.getElementById("m-recall").textContent =
            recall != null
                ? (recall * 100).toFixed(1) + "%"
                : "—";

        document.getElementById("m-misclass").textContent =
            misclass != null
                ? (misclass * 100).toFixed(1) + "%"
                : "—";


        if (drift != null) {

            const pct =
                Math.min(drift * 100, 100);

            document
                .getElementById("drift-bar-fill")
                .style.width = pct + "%";

            document
                .getElementById("drift-bar-pct")
                .textContent = pct.toFixed(1) + "%";

            document
                .getElementById("drift-bar-fill")
                .classList.toggle(
                    "warn",
                    drift > 0.20
                );
        }

    } catch (e) {

        console.error(
            "Failed to fetch pipeline metrics:",
            e
        );
    }


    try {

        const res =
            await fetch(`${API_BASE}/ready`);

        if (res.ok) {

            const data =
                await res.json();

            document
                .getElementById("pi-model-name")
                .textContent =
                    data.model_name || "—";

            document
                .getElementById("pi-model-version")
                .textContent =
                    data.model_version || "—";

            document
                .getElementById("pi-threshold")
                .textContent =
                    data.classification_threshold
                        ? data.classification_threshold.toFixed(2)
                        : "—";

            document
                .getElementById("pi-status")
                .textContent =
                    data.status || "—";
        }

    } catch (e) {}
}


// Auto-refresh pipeline metrics every 15s when on dashboard tab

setInterval(() => {

    const pipelineTab =
        document.getElementById("tab-pipeline");

    if (
        pipelineTab &&
        !pipelineTab.classList.contains("hidden")
    ) {
        fetchPipelineMetrics();
    }

}, 15000);


// ── Helpers ──────────────────────────────────────────────────────────────────

function showSpinner() {
    spinner.classList.remove("hidden");
}

function hideSpinner() {
    spinner.classList.add("hidden");
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
}

function hideError() {
    errorBanner.classList.add("hidden");
}


// ── Init ──────────────────────────────────────────────────────────────────────

checkSystemStatus();

setInterval(
    checkSystemStatus,
    30000
);