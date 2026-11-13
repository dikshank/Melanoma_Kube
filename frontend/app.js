/**
 * app.js
 * ------
 * Phase 4 | Executed: Local (browser, served by nginx)
 *
 * Handles:
 * - Image upload via file picker and drag-and-drop
 * - Calls POST /predict on FastAPI backend (with the photo type from the dropdown)
 * - Displays prediction result with color coding
 * - Feedback submission via POST /feedback
 * - Fallback-model banner via GET /ready
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
const imageType      = document.getElementById("image-type");


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
    formData.append("image_type", imageType.value);   // backend uses this type's threshold

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

    document.getElementById("detail-prob").textContent =
        `${(data.malignant_prob * 100).toFixed(1)}%`;

    document.getElementById("recommendation").textContent =
        data.recommendation;

    // is_fallback is always in the /predict response (true / false)
    document.getElementById("fallback-note")
        .classList.toggle("hidden", !data.is_fallback);

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


// ── Fallback banner ──────────────────────────────────────────────────────────

async function checkSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/ready`);
        if (!res.ok) return;
        const data = await res.json();
        // is_fallback is always in the /ready response (true / false)
        document.getElementById("fallback-banner")
            .classList.toggle("hidden", !(data.model_loaded && data.is_fallback));
    } catch {
        // backend not reachable: the analyse button will show the error
    }
}


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