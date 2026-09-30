/* OcuGrade front-end: plain JavaScript, no libraries, works offline.
   All text from the server is inserted with textContent (never innerHTML), so nothing can inject markup. */
"use strict";

const STAGE_COLORS = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"];
const STAGE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
const PIPE_TABS = [["original", "Original"], ["cropped", "Cropped"], ["denoised", "Denoised"],
                   ["clahe", "CLAHE"], ["sharpened", "Sharpened (model input)"]];
const PIPE_WHY = {
  original: "The uploaded photograph, padded to a square and resized to 224x224.",
  cropped: "Black borders removed so the retina fills the frame.",
  denoised: "Light Gaussian smoothing suppresses sensor noise before contrast is boosted.",
  clahe: "CLAHE on the lightness channel lifts local contrast so small lesions stand out.",
  sharpened: "Unsharp masking sharpens vessel and lesion edges. This is exactly what the network sees."
};
const state = { lastResult: null, busy: false };

/* ---------- tiny DOM helpers ---------- */
function h(tag, props, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node[k] = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  add(node, kids);
  return node;
}
function add(node, kids) {
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.appendChild(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
  }
}
function svgEl(tag, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, String(v));
  return n;
}
const pct = (x, d = 1) => (100 * Number(x)).toFixed(d) + "%";
const num = (x, d = 3) => (x === null || x === undefined || Number.isNaN(Number(x))) ? "n/a" : Number(x).toFixed(d);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " (" + r.status + ")");
  return r.json();
}
async function getText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " (" + r.status + ")");
  return r.text();
}
function parseCSV(text) {
  const rows = text.trim().split(/\r?\n/).map(line => line.split(","));
  const head = rows.shift() || [];
  return rows.map(r => Object.fromEntries(head.map((c, i) => [c, r[i]])));
}
function missing(what) {
  return h("div", { class: "banner info", text: what + " is not available yet. Run the Colab notebook, download export.zip and unzip it into the project folder." });
}
function figure(name, caption) {
  const fig = h("figure", { class: "fig" });
  const img = h("img", { src: "/api/assets/" + encodeURIComponent(name), alt: caption || name, loading: "lazy" });
  img.onerror = () => { fig.replaceChildren(missing(caption || name)); };
  add(fig, [img, caption ? h("figcaption", { text: caption }) : null]);
  return fig;
}
function table(rows, columns) {
  if (!rows || !rows.length) return h("p", { class: "hint", text: "No data." });
  const cols = columns || Object.keys(rows[0]);
  const fmt = v => (typeof v === "number") ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v ?? "");
  return h("div", { class: "scroll" }, h("table", {},
    h("thead", {}, h("tr", {}, cols.map(c => h("th", { text: c })))),
    h("tbody", {}, rows.map(r => h("tr", {}, cols.map(c => h("td", { text: fmt(r[c]) })))))));
}
async function tableFromAsset(name, columns) {
  try {
    if (name.endsWith(".json")) return table(await getJSON("/api/assets/" + name), columns);
    return table(parseCSV(await getText("/api/assets/" + name)), columns);
  } catch (e) { return missing(name); }
}

/* ---------- router ---------- */
const pages = { analyze: pageAnalyze, dataset: pageDataset, training: pageTraining, evaluation: pageEvaluation, impact: pageImpact };
function route() {
  const name = (location.hash || "#analyze").slice(1);
  const page = pages[name] ? name : "analyze";
  document.querySelectorAll("#nav a").forEach(a => {
    const on = a.dataset.page === page;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const view = document.getElementById("view");
  view.replaceChildren();
  Promise.resolve(pages[page](view)).catch(e => view.appendChild(h("div", { class: "banner err", text: "Could not load this page: " + e.message })));
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);

/* =====================================================================
   Analyze
   ===================================================================== */
async function pageAnalyze(view) {
  view.append(h("h1", { text: "Analyze a retinal image" }),
    h("p", { class: "lead", text: "Upload a fundus photograph to see the predicted diabetic retinopathy stage, the model's confidence, the preprocessing steps and a Grad-CAM explanation." }));

  const status = h("div", { id: "status", "aria-live": "polite" });
  const results = h("div", { id: "results" });
  const preview = h("div", { id: "preview" });
  const input = h("input", { type: "file", id: "file", accept: "image/png,image/jpeg" });
  const zone = h("label", { class: "dropzone", tabindex: "0", for: "file" },
    h("strong", { text: "Drag and drop a fundus image here" }),
    h("span", { class: "hint", text: "or click to browse (PNG or JPEG, up to 10 MB)" }), input);

  zone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach(t => zone.addEventListener(t, e => { e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => zone.addEventListener(t, e => { e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", e => { if (e.dataTransfer && e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });
  input.addEventListener("change", () => { if (input.files[0]) analyze(input.files[0]); });

  const samples = h("div", { class: "row", id: "samples" });
  view.append(h("div", { class: "card" }, zone, preview,
    h("h3", { text: "Try a sample" }), samples,
    h("div", { class: "hint", text: "Sample images come from your own test split (see the README)." })), status, results);
  loadHealth(status);
  loadSamples(samples);

  async function analyze(file) {
    if (state.busy) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) return showError(status, "Please choose a PNG or JPEG image.");
    if (file.size > 10 * 1024 * 1024) return showError(status, "The file is larger than 10 MB.");
    state.busy = true;
    results.replaceChildren();
    preview.replaceChildren(h("img", { class: "preview", src: URL.createObjectURL(file), alt: "Uploaded image preview" }),
      h("div", { class: "hint", text: file.name + " - " + (file.size / 1024).toFixed(0) + " KB" }));
    status.replaceChildren(h("div", { class: "banner info" }, h("span", { class: "spinner" }), "Analyzing..."));
    try {
      const resp = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": file.type }, body: file });
      const data = await resp.json().catch(() => ({ error: "Unexpected server response." }));
      if (!resp.ok) throw new Error(data.error || ("Server error " + resp.status));
      state.lastResult = data;
      status.replaceChildren();
      renderResult(results, data);
    } catch (e) {
      showError(status, e.message);
    } finally { state.busy = false; }
  }
  window.__analyze = analyze;
}
function showError(box, msg) { box.replaceChildren(h("div", { class: "banner err", role: "alert", text: msg })); }

async function loadHealth(box) {
  try {
    const hl = await getJSON("/api/health");
    if (!hl.model_loaded) box.replaceChildren(h("div", { class: "banner warn", text: hl.message || "The model is not loaded." }));
  } catch (e) { showError(box, "Cannot reach the server."); }
}
async function loadSamples(box) {
  try {
    const names = await getJSON("/api/samples");
    if (!names.length) { box.append(h("span", { class: "hint", text: "No samples found in webapp/samples/ (optional)." })); return; }
    names.forEach(n => box.append(h("button", { class: "btn ghost", type: "button", text: n.replace(/\.[^.]+$/, ""),
      onclick: async () => {
        const r = await fetch("/api/samples/" + encodeURIComponent(n));
        const blob = await r.blob();
        window.__analyze(new File([blob], n, { type: blob.type || "image/png" }));
      } })));
  } catch (e) { /* samples are optional */ }
}

function gauge(stage, confidence) {
  const L = Math.PI * 90;
  const frac = (stage + 1) / 5;
  const svg = svgEl("svg", { viewBox: "0 0 200 125", class: "gauge", role: "img", "aria-label": "Stage " + stage + " of 4" });
  svg.append(svgEl("path", { d: "M 10 105 A 90 90 0 0 1 190 105", fill: "none", stroke: "#eee7da", "stroke-width": 16, "stroke-linecap": "round" }),
    svgEl("path", { d: "M 10 105 A 90 90 0 0 1 190 105", fill: "none", stroke: STAGE_COLORS[stage], "stroke-width": 16, "stroke-linecap": "round",
      "stroke-dasharray": L.toFixed(2), "stroke-dashoffset": (L * (1 - frac)).toFixed(2) }));
  const t1 = svgEl("text", { x: 100, y: 92, "text-anchor": "middle", class: "stage", fill: "#1f2a2e" }); t1.textContent = String(stage);
  const t2 = svgEl("text", { x: 100, y: 112, "text-anchor": "middle", class: "sub" }); t2.textContent = "of 4 (" + pct(confidence, 0) + " confident)";
  svg.append(t1, t2);
  return svg;
}

function renderResult(box, r) {
  const color = STAGE_COLORS[r.stage];
  const banners = [];
  if (r.low_confidence) banners.push(h("div", { class: "banner warn", role: "alert", text: "Low confidence: needs human review. The top probability is below " + pct(r.threshold, 0) + "." }));
  (r.quality_warnings || []).forEach(w => banners.push(h("div", { class: "banner warn", text: w })));

  const strip = h("div", { class: "strip", "aria-hidden": "true" }, STAGE_NAMES.map((n, i) =>
    h("div", { class: i === r.stage ? "on" : "", style: "background:" + STAGE_COLORS[i], text: i + ". " + n })));

  const resultCard = h("div", { class: "card" }, h("h2", { text: "Result" }),
    h("div", { class: "gauge-wrap" }, gauge(r.stage, r.confidence)),
    h("p", { style: "text-align:center;margin:.2rem 0" }, h("strong", { text: r.stage_name }), " ", h("span", { class: "triage", text: r.triage })),
    strip, banners,
    h("div", { class: "card", style: "background:#f6f0e4;margin:12px 0 0" },
      h("strong", { text: "Suggested action: " }), r.action, h("div", { class: "hint", text: r.action_note })),
    h("div", { class: "row", style: "margin-top:12px" }, h("button", { class: "btn", type: "button", text: "Download PDF report", onclick: downloadReport })));

  const bars = h("div", { class: "bars" }, r.probabilities.map((p, i) =>
    h("div", { class: "bar" + (i === r.stage ? " top" : "") },
      h("span", { text: p.name }),
      h("div", { class: "track", role: "presentation" }, h("div", { class: "fill", style: "width:" + (100 * p.probability).toFixed(1) + "%;background:" + STAGE_COLORS[i] })),
      h("span", { class: "pct", text: pct(p.probability) }))));
  const probCard = h("div", { class: "card" }, h("h2", { text: "Class probabilities" }), bars);

  box.replaceChildren(h("div", { class: "grid2" }, resultCard, h("div", {}, probCard, pipelineCard(r))), camCard(r));
}

function pipelineCard(r) {
  const img = h("img", { class: "pipe-img", alt: "Pipeline step" });
  const why = h("p", { class: "hint" });
  const tabs = h("div", { class: "tabs", role: "tablist" });
  function show(key) {
    img.src = "data:image/png;base64," + r.images[key];
    why.textContent = PIPE_WHY[key];
    tabs.querySelectorAll("button").forEach(b => b.setAttribute("aria-selected", b.dataset.key === key ? "true" : "false"));
  }
  PIPE_TABS.forEach(([key, label]) => tabs.append(h("button", { class: "tab", type: "button", role: "tab", "data-key": key, text: label, onclick: () => show(key) })));
  show("sharpened");
  return h("div", { class: "card" }, h("h2", { text: "Preprocessing pipeline" }), tabs, img, why);
}

function camCard(r) {
  const heat = h("img", { class: "heat", alt: "Grad-CAM heat-map", src: "data:image/png;base64," + r.images.heatmap, style: "opacity:0.55" });
  const slider = h("input", { type: "range", min: "0", max: "100", value: "55", "aria-label": "Heat-map opacity" });
  slider.addEventListener("input", () => { heat.style.opacity = String(slider.value / 100); });
  return h("div", { class: "card" }, h("h2", { text: "Grad-CAM explanation" }),
    h("div", { class: "cam-pair" },
      h("figure", {}, h("div", { class: "cam" }, h("img", { alt: "Model input", src: "data:image/png;base64," + r.images.sharpened })), h("figcaption", { text: "Model input" })),
      h("figure", {}, h("div", { class: "cam" }, h("img", { alt: "Model input under heat-map", src: "data:image/png;base64," + r.images.sharpened }), heat),
        h("figcaption", { text: "Heat-map over the image" }), h("label", { class: "hint" }, "Heat-map opacity", slider))),
    h("p", { class: "hint", text: "Warm colours mark the regions that pushed the model towards its prediction. Grad-CAM shows where the model looked, not a clinical finding; a clinician must interpret the image." }));
}

async function downloadReport() {
  if (!state.lastResult) return;
  try {
    const r = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.lastResult) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "Report failed");
    const url = URL.createObjectURL(await r.blob());
    const a = h("a", { href: url, download: "ocugrade_report.pdf" });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (e) { alert("Could not create the report: " + e.message); }
}

/* =====================================================================
   Dataset & Pipeline
   ===================================================================== */
async function pageDataset(view) {
  view.append(h("h1", { text: "Dataset & pipeline" }),
    h("p", { class: "lead", text: "Diabetic retinopathy (DR) damages the retinal blood vessels in people with diabetes and is a leading cause of preventable blindness. Screening programmes photograph the retina and grade severity on a five-step scale, so early detection can trigger timely treatment." }));
  view.append(h("div", { class: "card" }, h("h2", { text: "Dataset: APTOS 2019 Blindness Detection (Kaggle)" }),
    h("p", { text: "Retinal fundus photographs graded 0 (No DR) to 4 (Proliferative DR) by clinicians. The data is relevant because it is clinically graded, publicly available and covers every severity stage; images are not redistributed in this repository." }),
    await summaryBlock(), figure("class_distribution.png", "Class distribution and imbalance"), figure("sample_images.png", "Sample images for each grade")));
  view.append(h("div", { class: "card" }, h("h2", { text: "Stratified train / validation / test split (70 / 15 / 15)" }),
    h("p", { text: "Every class keeps the same proportion in all three sets, and the code asserts that no image appears in more than one set (no data leakage). Preprocessing statistics, augmentation and class weights use the training set only." }),
    await tableFromAsset("split_counts.csv"), figure("split_distribution.png", "Images per class in each split")));
  view.append(h("div", { class: "card" }, h("h2", { text: "Limitations and ethical concerns" }), h("ul", { class: "clean" },
    h("li", { text: "Small, imbalanced dataset: Severe and Proliferative grades have few examples." }),
    h("li", { text: "Single source: performance may drop on other cameras, clinics and populations." }),
    h("li", { text: "Labels are subjective near grade boundaries, so some label noise is unavoidable." }),
    h("li", { text: "Demographic information is unavailable, so fairness across age, sex and ethnicity cannot be checked." }),
    h("li", { text: "Medical images are sensitive: uploads here are processed in memory and never stored." }))));
  view.append(h("div", { class: "card" }, h("h2", { text: "Preprocessing" }),
    h("p", { text: "Crop borders, pad to square and resize to 224x224, light denoising, CLAHE contrast enhancement and unsharp-mask edge enhancement. The same function runs in training, evaluation and this app. EfficientNet normalises pixel values internally, so images stay in the 0-255 range." }),
    figure("preprocessing_steps.png", "Each preprocessing step, one example per grade"),
    figure("preprocessing_metrics.png", "Contrast and sharpness before and after preprocessing"), await preprocMetrics()));
  view.append(h("div", { class: "card" }, h("h2", { text: "Augmentation and class imbalance" }),
    h("p", { text: "Training images are randomly flipped, rotated (up to 20 degrees), zoomed (10%) and brightness/contrast-jittered; these keep the diagnosis unchanged while adding variety. Imbalance is handled with class weights; minority oversampling was compared on the validation set." }),
    figure("augmentation_examples.png", "Augmentation examples"), figure("imbalance_handling.png", "Class counts, oversampling and class weights"),
    h("h3", { text: "Imbalance strategies compared (validation set)" }), await tableFromAsset("imbalance_comparison.json", ["strategy", "accuracy", "macro_f1", "qwk", "seconds"])));
}
async function summaryBlock() {
  try {
    const s = await getJSON("/api/assets/eda_summary.json");
    return h("div", { class: "cards4" },
      h("div", { class: "metric" }, h("b", { text: String(s.n_images) }), h("span", { text: "labelled images" })),
      h("div", { class: "metric" }, h("b", { text: s.imbalance_ratio + " : 1" }), h("span", { text: "largest / smallest class" })),
      h("div", { class: "metric" }, h("b", { text: String(s.n_distinct_image_sizes) }), h("span", { text: "distinct image sizes" })));
  } catch (e) { return missing("Dataset summary"); }
}
async function preprocMetrics() {
  try {
    const m = await getJSON("/api/assets/preprocessing_metrics.json");
    return h("p", { class: "hint", text: "Measured on " + m.n_images + " images: contrast (grey-level std) " + num(m.contrast_before, 1) + " -> " + num(m.contrast_after, 1) + ", sharpness (Laplacian variance) " + num(m.sharpness_before, 1) + " -> " + num(m.sharpness_after, 1) + "." });
  } catch (e) { return null; }
}

/* =====================================================================
   Model & Training
   ===================================================================== */
async function pageTraining(view) {
  view.append(h("h1", { text: "Model & training" }));
  view.append(h("div", { class: "card" }, h("h2", { text: "Architecture: EfficientNetB0 with transfer learning" }),
    h("p", { text: "EfficientNetB0 balances accuracy and size (about 4 million parameters, pretrained on ImageNet), so it trains on a free Colab GPU and is small enough to ship with this app. The head is GlobalAveragePooling, BatchNorm, Dropout and a 5-way softmax." }),
    h("ul", { class: "clean" },
      h("li", {}, h("strong", { text: "Phase 1: " }), "backbone frozen, only the new head is trained (learning rate 1e-3)."),
      h("li", {}, h("strong", { text: "Phase 2: " }), "the top backbone layers are unfrozen and fine-tuned with a very small learning rate (1e-5); BatchNorm layers stay frozen."),
      h("li", {}, h("strong", { text: "Regularisation and control: " }), "dropout, early stopping on validation loss, learning-rate reduction on plateau, best-model checkpointing, class weights, fixed random seeds."))));
  view.append(h("div", { class: "card" }, h("h2", { text: "Backbone comparison (validation set)" }),
    await tableFromAsset("backbone_comparison.json", ["backbone", "accuracy", "macro_f1", "qwk", "parameters", "seconds"])));
  view.append(h("div", { class: "card" }, h("h2", { text: "Hyper-parameter search (validation set)" }),
    h("p", { class: "hint", text: "One factor changed at a time around the baseline; the best validation macro-F1 was used for the final model. The test set was never used for tuning." }),
    await tableFromAsset("hyperparameter_search.json", ["dropout", "unfreeze", "head_lr", "accuracy", "macro_f1", "qwk", "seconds"])));
  view.append(h("div", { class: "card" }, h("h2", { text: "Training curves" }), await historySummary(), figure("training_curves.png", "Accuracy and loss for phase 1, phase 2 and combined")));
}
async function historySummary() {
  try {
    const hist = await getJSON("/api/assets/history.json");
    const items = [];
    for (const p of ["phase1", "phase2"]) {
      if (!hist[p]) continue;
      const va = hist[p].val_accuracy || [];
      items.push(h("li", { text: p.replace("phase", "Phase ") + ": " + (hist[p].loss || []).length + " epochs, best validation accuracy " + pct(Math.max(...va)) }));
    }
    return h("ul", { class: "clean" }, items);
  } catch (e) { return null; }
}

/* =====================================================================
   Evaluation
   ===================================================================== */
async function pageEvaluation(view) {
  view.append(h("h1", { text: "Evaluation" }), h("p", { class: "lead", text: "All numbers are measured on the held-out test set, which was never used for training or tuning." }));
  let m;
  try { m = await getJSON("/api/metrics"); } catch (e) { view.append(missing("Evaluation metrics")); return; }
  view.append(h("div", { class: "cards4" },
    [["Accuracy", pct(m.accuracy)], ["Macro precision", num(m.macro_precision)], ["Macro recall", num(m.macro_recall)],
     ["Macro F1", num(m.macro_f1)], ["Quadratic weighted kappa", num(m.quadratic_weighted_kappa)], ["Macro ROC-AUC", num(m.macro_roc_auc)]]
      .map(([label, value]) => h("div", { class: "metric" }, h("b", { text: value }), h("span", { text: label })))));
  const rows = Object.entries(m.per_class || {}).map(([name, v]) => ({ class: name, precision: v.precision, recall: v.recall, f1: v.f1, support: v.support }));
  view.append(h("div", { class: "card" }, h("h2", { text: "Per-class precision, recall and F1" }), table(rows)));
  view.append(h("div", { class: "card" }, h("h2", { text: "Confusion matrix and ROC curves" }), figure("confusion_matrix.png", "Confusion matrix (counts and row-normalised)"),
    h("div", { class: "grid2" }, figure("roc_curves.png", "One-vs-rest ROC curves"), figure("confidence_histogram.png", "Confidence of correct vs wrong predictions"))));
  view.append(h("div", { class: "card" }, h("h2", { text: "Error analysis" }), await errorText(),
    figure("misclassified_gallery.png", "Misclassified test images, most confident errors first"), figure("gradcam_examples.png", "Grad-CAM on correct and wrong predictions"),
    h("h3", { text: "Most confident mistakes" }), await tableFromAsset("misclassified.json")));
}
async function errorText() {
  try {
    const text = await getText("/api/assets/error_analysis.md");
    return h("pre", { class: "text", text: text.replace(/[*#]/g, "").trim() });
  } catch (e) { return missing("Error analysis text"); }
}

/* =====================================================================
   Impact & Ethics
   ===================================================================== */
async function pageImpact(view) {
  view.append(h("h1", { text: "Impact & ethics" }));
  const card = (title, items) => h("div", { class: "card" }, h("h2", { text: title }), h("ul", { class: "clean" }, items.map(t => h("li", { text: t }))));
  view.append(
    card("Real-world healthcare impact", [
      "Screening for DR is often limited by the number of trained graders. A tool like this could prioritise images for review so that severe cases are seen sooner, especially where specialists are scarce.",
      "Grad-CAM heat-maps and a low-confidence warning help a clinician see why a prediction was made and when not to trust it."]),
    card("Deployment feasibility", [
      "The model is small (about 17 MB) and runs on an ordinary CPU, so it could run on a clinic computer or a local server without a GPU.",
      "A real deployment would need clinical validation on local data, regulatory approval, integration with imaging systems and audit logging."]),
    card("Risks and limitations", [
      "False negatives on severe disease could delay treatment; false positives add unnecessary referrals.",
      "Trained on one dataset: performance on other cameras, image quality or populations is unknown.",
      "Users may over-trust automated output; every result needs clinician review."]),
    card("Privacy and fairness", [
      "Uploaded images are processed in memory and are not saved by this app.",
      "Real patient data would need consent, de-identification and secure handling under the applicable data-protection law.",
      "Demographic performance was not measured and must be audited before any real use."]),
    card("Future work", [
      "Higher resolution input (for example EfficientNetB3/B4) to keep microaneurysms visible.",
      "Ordinal regression losses that respect the ordering of severity grades.",
      "Ensembles, external validation on other datasets, uncertainty estimation and lesion segmentation."]));
}
