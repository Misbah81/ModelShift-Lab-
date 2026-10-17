// ModelShift Lab — frontend
"use strict";

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "modelshift_history";
const COLORS = { c0: "#6ee7e0", c1: "#b19aff", red: "#ff7777" };

let accuracyChart = null;
let latestExperiment = null;
let latestBoundaryModel = null;

/* ---------- helpers ---------- */

// Escape text before putting it in innerHTML (defensive; values are server-validated).
function esc(value) {
    return String(value).replace(/[&<>"']/g, (ch) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
    ));
}
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const DATASET_LABELS = { moons: "Two Moons", circles: "Concentric Circles", classification: "Class Imbalance" };

/* ---------- range labels & control state ---------- */

function updateRangeLabels() {
    $("samplesValue").textContent = $("samples").value;
    $("noiseValue").textContent = Number($("noise").value).toFixed(2);
    $("imbalanceValue").textContent = Number($("imbalance").value).toFixed(2);
    $("shiftValue").textContent = Number($("shift").value).toFixed(2);
}

function updateImbalanceState() {
    const active = $("dataset").value === "classification";
    $("imbalance").disabled = !active;
    $("imbalanceNote").textContent = active
        ? "Fraction of samples in class 0."
        : "Only used by the Class Imbalance dataset.";
}

function selectedModels() {
    return [...document.querySelectorAll("#modelOptions input:checked")].map((c) => c.value);
}

// Keep the boundary-model dropdown limited to models that will actually be trained.
function syncBoundaryOptions() {
    const chosen = selectedModels();
    const select = $("boundaryModel");
    [...select.options].forEach((opt) => {
        opt.disabled = chosen.length > 0 && !chosen.includes(opt.value);
    });
    if (chosen.length && !chosen.includes(select.value)) select.value = chosen[0];
}

["samples", "noise", "imbalance", "shift"].forEach((id) => $(id).addEventListener("input", updateRangeLabels));
$("dataset").addEventListener("change", updateImbalanceState);
document.querySelectorAll("#modelOptions input").forEach((c) => c.addEventListener("change", syncBoundaryOptions));
updateRangeLabels();
updateImbalanceState();
syncBoundaryOptions();

/* ---------- sidebar navigation ---------- */

function setupNavigation() {
    const items = [...document.querySelectorAll(".nav-item[data-section]")];
    const setActive = (id) => items.forEach((n) => n.classList.toggle("active", n.dataset.section === id));

    items.forEach((item) => item.addEventListener("click", (event) => {
        event.preventDefault();
        const section = $(item.dataset.section);
        if (!section) return;
        setActive(item.dataset.section);
        section.scrollIntoView({ behavior: "smooth", block: "start" });
    }));

    // Highlight the section currently in view.
    if ("IntersectionObserver" in window) {
        const visible = new Map();
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((e) => visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0));
            const order = items.map((n) => n.dataset.section);
            const current = order.find((id) => (visible.get(id) || 0) > 0 && id !== "experiment-studio")
                || "experiment-studio";
            setActive(current);
        }, { rootMargin: "-10% 0px -60% 0px", threshold: [0, 0.01, 0.1] });
        items.forEach((n) => { const s = $(n.dataset.section); if (s && n.dataset.section !== "experiment-studio") observer.observe(s); });
    }
}
setupNavigation();

/* ---------- Plotly ---------- */

const baseLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Manrope, Arial, sans-serif", color: "#aebbd0", size: 11 },
    margin: { l: 48, r: 15, t: 35, b: 42 },
    xaxis: { title: "Feature 1", gridcolor: "rgba(115,137,165,.13)", zerolinecolor: "rgba(115,137,165,.2)" },
    yaxis: { title: "Feature 2", gridcolor: "rgba(115,137,165,.13)", zerolinecolor: "rgba(115,137,165,.2)" },
    legend: { orientation: "h", y: 1.12, x: 0 },
    autosize: true,
};
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function plotlyReady() {
    if (typeof Plotly === "undefined") {
        console.error("Plotly failed to load (check your internet connection / CDN access).");
        return false;
    }
    return true;
}

function drawBoundary(data) {
    if (!data.boundary || !plotlyReady()) return;
    const b = data.boundary;
    const train = (label) => data.train_points.filter((p) => p.label === label);

    const traces = [
        {
            type: "heatmap", x: b.x, y: b.y, z: b.z, showscale: false, hoverinfo: "skip",
            zmin: 0, zmax: 1,
            colorscale: [[0, "rgba(110,231,224,.22)"], [0.5, "rgba(110,231,224,.22)"],
                         [0.5, "rgba(177,154,255,.22)"], [1, "rgba(177,154,255,.22)"]],
        },
        ...[0, 1].map((label) => ({
            type: "scatter", mode: "markers", name: `Training · class ${label}`,
            x: train(label).map((p) => p.x), y: train(label).map((p) => p.y),
            marker: { color: label === 0 ? COLORS.c0 : COLORS.c1, size: 6, opacity: 0.85 },
        })),
    ];

    Plotly.newPlot("boundaryPlot", traces, {
        ...baseLayout,
        title: { text: `Decision regions · ${esc(b.model)}`, font: { size: 12, color: "#edf2fa" }, x: 0.02 },
        xaxis: { ...baseLayout.xaxis, range: [b.x[0], b.x[b.x.length - 1]] },
        yaxis: { ...baseLayout.yaxis, range: [b.y[0], b.y[b.y.length - 1]], scaleanchor: "x", scaleratio: 1 },
    }, plotConfig);
}

function drawShift(data) {
    const points = data.test_points || [];
    if (!points.length || !plotlyReady()) return;
    const traces = [];

    [0, 1].forEach((label) => {
        const g = points.filter((p) => p.label === label);
        const color = label === 0 ? COLORS.c0 : COLORS.c1;

        traces.push({
            type: "scatter", mode: "markers", name: `Original · class ${label}`,
            x: g.map((p) => p.x), y: g.map((p) => p.y),
            marker: { color, size: 7, symbol: "circle-open", line: { width: 1.5, color } },
        });
        traces.push({
            type: "scatter", mode: "markers", name: `Shifted · class ${label}`,
            x: g.map((p) => p.shift_x), y: g.map((p) => p.shift_y),
            marker: {
                color: g.map((p) => (p.shift_correct ? color : COLORS.red)),
                size: 8,
                symbol: g.map((p) => (p.shift_correct ? "circle" : "x")),
                line: { width: 1, color: g.map((p) => (p.shift_correct ? color : COLORS.red)) },
            },
        });
    });

    const xs = points.flatMap((p) => [p.x, p.shift_x]);
    const ys = points.flatMap((p) => [p.y, p.shift_y]);
    Plotly.newPlot("shiftPlot", traces, {
        ...baseLayout,
        xaxis: { ...baseLayout.xaxis, range: [Math.min(...xs) - 0.3, Math.max(...xs) + 0.3] },
        yaxis: { ...baseLayout.yaxis, range: [Math.min(...ys) - 0.3, Math.max(...ys) + 0.3], scaleanchor: "x", scaleratio: 1 },
    }, plotConfig);
}

/* ---------- failure analysis ---------- */

function confusionTable(m) {
    return `<table class="confusion-table">
        <thead><tr><th>True ↓ / Pred →</th><th>0</th><th>1</th></tr></thead>
        <tbody>
            <tr><th>0</th><td class="correct-cell">${m[0][0]}</td><td class="error-cell">${m[0][1]}</td></tr>
            <tr><th>1</th><td class="error-cell">${m[1][0]}</td><td class="correct-cell">${m[1][1]}</td></tr>
        </tbody></table>`;
}

function renderConfusionMatrices(data) {
    const container = $("confusionMatrices");
    const results = data.results || [];
    if (!results.length) {
        container.innerHTML = '<div class="empty-state">No model results available.</div>';
        return;
    }
    container.innerHTML = results.map((r) => `
        <div class="confusion-card">
            <div class="confusion-title"><strong>${esc(r.name)}</strong></div>
            <div class="confusion-pair">
                <div><h4>CLEAN TEST</h4>${confusionTable(r.clean_confusion)}</div>
                <div><h4>SHIFTED TEST</h4>${confusionTable(r.confusion)}</div>
            </div>
        </div>`).join("");
}

function renderFailureSummary(data) {
    const results = data.results || [];
    $("failureSummary").innerHTML = `<div class="failure-summary-grid">${results.map((r) => `
        <div class="failure-stat">
            <span>${esc(r.name)}</span>
            <strong>${r.shift_errors}</strong>
            <small>shifted errors (clean: ${r.clean_errors})</small>
            <div class="failure-mini">
                New failures after shift: ${r.new_failures_after_shift}<br>
                Recovered after shift: ${r.recovered_after_shift}<br>
                FP: ${r.confusion[0][1]} · FN: ${r.confusion[1][0]}
            </div>
        </div>`).join("")}</div>`;
}

/* ---------- model comparison ---------- */

function renderModelComparison(data) {
    const results = data.results || [];
    if (!results.length) return;

    const best = [...results].sort((a, b) => b.accuracy - a.accuracy)[0];
    $("bestAccuracy").textContent = pct(best.accuracy);
    $("bestModel").textContent = best.name;
    $("testSamples").textContent = data.summary.test_samples;

    const avgChange = results.reduce((s, r) => s + r.accuracy_change_pp, 0) / results.length;
    $("avgDrop").textContent = `${avgChange > 0 ? "+" : avgChange < 0 ? "−" : ""}${Math.abs(avgChange).toFixed(1)} pp`;

    if (accuracyChart) accuracyChart.destroy();
    if (typeof Chart === "undefined") {
        console.error("Chart.js failed to load (check your internet connection / CDN access).");
    } else {
        accuracyChart = new Chart($("accuracyChart"), {
            type: "bar",
            data: {
                labels: results.map((r) => r.name),
                datasets: [
                    { label: "Clean test accuracy", data: results.map((r) => r.clean_accuracy * 100), backgroundColor: COLORS.c0, borderRadius: 5 },
                    { label: "Shifted test accuracy", data: results.map((r) => r.accuracy * 100), backgroundColor: "#7c9cff", borderRadius: 5 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: "#aebbd0", usePointStyle: true, boxWidth: 8 } } },
                scales: {
                    x: { ticks: { color: "#aebbd0" }, grid: { display: false } },
                    y: { min: 0, max: 100, ticks: { color: "#8493aa", callback: (v) => `${v}%` }, grid: { color: "rgba(115,137,165,.12)" } },
                },
            },
        });
    }

    $("resultsTable").innerHTML = `<table class="results-table">
        <thead><tr><th>MODEL</th><th>CLEAN ACC.</th><th>SHIFTED ACC.</th><th>CHANGE</th><th>PRECISION</th><th>RECALL</th><th>F1</th></tr></thead>
        <tbody>${results.map((r) => `<tr>
            <td>${esc(r.name)}</td>
            <td>${pct(r.clean_accuracy)}</td>
            <td><strong>${pct(r.accuracy)}</strong></td>
            <td class="${r.accuracy_change_pp < 0 ? "neg" : r.accuracy_change_pp > 0 ? "pos" : ""}">${r.accuracy_change_pp > 0 ? "+" : ""}${r.accuracy_change_pp.toFixed(1)} pp</td>
            <td>${pct(r.precision)}</td><td>${pct(r.recall)}</td><td>${pct(r.f1)}</td>
        </tr>`).join("")}</tbody></table>`;
}

/* ---------- history ---------- */

function getHistory() {
    try {
        const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY));
        return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
}

function setHistory(list) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); } catch (e) { console.warn("History not saved:", e); }
}

function saveHistory(data) {
    const s = data.summary;
    const best = [...data.results].sort((a, b) => b.accuracy - a.accuracy)[0];
    const history = getHistory();
    history.unshift({
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        dataset: s.dataset, samples: s.samples, noise: s.noise, imbalance: s.imbalance,
        shift: s.shift, seed: s.seed, models: s.models, boundary_model: s.boundary_model,
        best: { name: best.name, accuracy: best.accuracy },
    });
    setHistory(history.slice(0, 10));
    renderHistory();
}

function renderHistory() {
    const container = $("historyList");
    const history = getHistory();
    if (!history.length) {
        container.innerHTML = '<div class="empty-state">Your completed runs will appear here.</div>';
        return;
    }
    container.innerHTML = history.map((h, i) => `
        <button type="button" class="history-item" data-index="${i}">
            <div><strong>Run ${history[0].id === h.id ? "(latest)" : `#${history.length - i}`}</strong><small>${esc(h.timestamp)}</small></div>
            <div class="history-meta">
                ${esc(DATASET_LABELS[h.dataset] || h.dataset)} · shift ${Number(h.shift).toFixed(2)} · seed ${esc(h.seed)}<br>
                ${h.best ? `Best: ${esc(h.best.name)} ${pct(h.best.accuracy)}` : ""}
            </div>
        </button>`).join("");
}

function loadSettings(h) {
    $("dataset").value = h.dataset;
    $("samples").value = h.samples;
    $("noise").value = h.noise;
    if (h.imbalance !== undefined) $("imbalance").value = h.imbalance;
    $("shift").value = h.shift;
    $("seed").value = h.seed;
    if (Array.isArray(h.models)) {
        document.querySelectorAll("#modelOptions input").forEach((c) => { c.checked = h.models.includes(c.value); });
    }
    syncBoundaryOptions();
    if (h.boundary_model) $("boundaryModel").value = h.boundary_model;
    updateRangeLabels();
    updateImbalanceState();
    $("insightText").textContent = "Settings loaded from history. Click “Run experiment” to reproduce the run.";
    $("experiment-studio").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("historyList").addEventListener("click", (event) => {
    const item = event.target.closest(".history-item");
    if (!item) return;
    const h = getHistory()[Number(item.dataset.index)];
    if (h) loadSettings(h);
});

$("clearHistoryBtn").addEventListener("click", () => {
    try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
    renderHistory();
});
renderHistory();

/* ---------- CSV export ---------- */

function csvCell(value) {
    let text = String(value);
    // Neutralise spreadsheet formula injection for text cells that start with a formula character.
    if (/^[=+\-@]/.test(text) && Number.isNaN(Number(text))) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
}

function exportCSV() {
    const d = latestExperiment;
    if (!d) return;
    const s = d.summary;
    const rows = [
        ["ModelShift Lab experiment report"],
        ["Dataset", s.dataset], ["Samples", s.samples], ["Train samples", s.train_samples],
        ["Test samples", s.test_samples], ["Noise", s.noise], ["Class 0 proportion", s.imbalance],
        ["Shift", s.shift], ["Random seed", s.seed], ["Boundary model", s.boundary_model],
        [],
        ["Model", "Clean accuracy", "Shifted accuracy", "Accuracy change (percentage points)",
         "Clean precision", "Clean recall", "Clean F1", "Shifted precision", "Shifted recall", "Shifted F1",
         "Clean errors", "Shifted errors", "New failures after shift", "Recovered after shift",
         "Clean confusion matrix", "Shifted confusion matrix"],
        ...d.results.map((r) => [
            r.name, r.clean_accuracy, r.accuracy, r.accuracy_change_pp,
            r.clean_precision, r.clean_recall, r.clean_f1, r.precision, r.recall, r.f1,
            r.clean_errors, r.shift_errors, r.new_failures_after_shift, r.recovered_after_shift,
            JSON.stringify(r.clean_confusion), JSON.stringify(r.confusion),
        ]),
        [],
        [`Test-sample failures after shift (boundary model: ${s.boundary_model})`],
        ["True label", "Clean prediction", "Shifted prediction", "Clean correct", "Shifted correct",
         "Original x", "Original y", "Shifted x", "Shifted y"],
        ...d.test_points.filter((p) => !p.shift_correct).map((p) => [
            p.label, p.clean_pred, p.shift_pred, p.clean_correct, p.shift_correct, p.x, p.y, p.shift_x, p.shift_y,
        ]),
    ];
    const csv = "﻿" + rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `modelshift-report-${s.dataset}-seed-${s.seed}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}
$("exportBtn").addEventListener("click", exportCSV);

/* ---------- run experiment ---------- */

function buildInsight(data) {
    const s = data.summary;
    const worst = [...data.results].sort((a, b) => a.accuracy_change_pp - b.accuracy_change_pp)[0];
    const best = [...data.results].sort((a, b) => b.accuracy - a.accuracy)[0];
    if (s.shift === 0) {
        return `Baseline run (shift 0, seed ${s.seed}). ${best.name} reached ${pct(best.accuracy)} accuracy. Raise the shift and rerun with the same seed to measure robustness.`;
    }
    const change = worst.accuracy_change_pp;
    const detail = change < 0
        ? `${worst.name} changed the most (${change.toFixed(1)} pp).`
        : "No model lost accuracy at this shift level.";
    return `Shift ${s.shift.toFixed(1)} was applied to the held-out test features only. ${best.name} had the highest shifted accuracy (${pct(best.accuracy)}). ${detail} Results come from a single seed; repeat across seeds before drawing conclusions.`;
}

$("runBtn").addEventListener("click", async () => {
    const models = selectedModels();
    if (!models.length) { alert("Select at least one model."); return; }

    const seedValue = Number.parseInt($("seed").value, 10);
    if (!Number.isInteger(seedValue) || seedValue < 0 || seedValue > 100000) {
        alert("Seed must be a whole number between 0 and 100000.");
        return;
    }

    const button = $("runBtn");
    button.disabled = true;
    button.innerHTML = "Training models… <span>◌</span>";

    try {
        const response = await fetch("/api/experiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset: $("dataset").value,
                samples: Number($("samples").value),
                noise: Number($("noise").value),
                imbalance: Number($("imbalance").value),
                shift: Number($("shift").value),
                seed: seedValue,
                models,
                boundary_model: $("boundaryModel").value,
            }),
        });

        let data;
        try { data = await response.json(); } catch { throw new Error("Server returned an unreadable response."); }
        if (!response.ok) throw new Error(data.error || "Experiment failed.");

        latestExperiment = data;
        latestBoundaryModel = data.boundary.model;
        drawBoundary(data);
        drawShift(data);
        renderModelComparison(data);
        renderConfusionMatrices(data);
        renderFailureSummary(data);
        saveHistory(data);
        $("exportBtn").disabled = false;
        $("insightText").textContent = buildInsight(data);
    } catch (error) {
        console.error("ModelShift experiment error:", error);
        alert("Could not run experiment.\n\n" + error.message);
    } finally {
        button.disabled = false;
        button.innerHTML = "Run experiment <span>↗</span>";
    }
});
