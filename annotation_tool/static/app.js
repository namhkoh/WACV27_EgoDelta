"use strict";

const $ = (id) => document.getElementById(id);
const elements = {
  annotatorName: $("annotatorName"), saveState: $("saveState"), progressText: $("progressText"),
  flagText: $("flagText"), progressBar: $("progressBar"), sceneSelect: $("sceneSelect"),
  sceneStrip: $("sceneStrip"),
  previousButton: $("previousButton"), nextButton: $("nextButton"), nextOpenButton: $("nextOpenButton"),
  componentList: $("componentList"), brushSize: $("brushSize"), brushOutput: $("brushOutput"),
  undoButton: $("undoButton"), redoButton: $("redoButton"), finishPolygonButton: $("finishPolygonButton"),
  clearComponentButton: $("clearComponentButton"), resetComponentButton: $("resetComponentButton"),
  cancelPolygonButton: $("cancelPolygonButton"), opacity: $("opacity"), opacityOutput: $("opacityOutput"),
  zoom: $("zoom"), zoomOutput: $("zoomOutput"), overlayToggle: $("overlayToggle"),
  edgeOnlyToggle: $("edgeOnlyToggle"),
  taskPosition: $("taskPosition"), taskTitle: $("taskTitle"), taskMeta: $("taskMeta"),
  taskStatus: $("taskStatus"), canvasViewport: $("canvasViewport"), canvasStage: $("canvasStage"),
  imageCanvas: $("imageCanvas"), overlayCanvas: $("overlayCanvas"), guideCanvas: $("guideCanvas"),
  maskCanvas: $("maskCanvas"), noteInput: $("noteInput"), flagButton: $("flagButton"),
  saveButton: $("saveButton"), completeButton: $("completeButton"), errorPanel: $("errorPanel"),
  errorMessage: $("errorMessage"),
  finalReviewButton: $("finalReviewButton"), finalReviewPanel: $("finalReviewPanel"),
  closeFinalReviewButton: $("closeFinalReviewButton"), reviewDatasetMeta: $("reviewDatasetMeta"),
  reviewSceneSelect: $("reviewSceneSelect"), reviewSceneStrip: $("reviewSceneStrip"),
  reviewPreviousButton: $("reviewPreviousButton"), reviewNextButton: $("reviewNextButton"),
  reviewPosition: $("reviewPosition"), reviewOpacity: $("reviewOpacity"),
  reviewOpacityOutput: $("reviewOpacityOutput"), reviewBoundaryToggle: $("reviewBoundaryToggle"),
  reviewTaskTitle: $("reviewTaskTitle"), reviewTaskMeta: $("reviewTaskMeta"),
  reviewLoadState: $("reviewLoadState"), reviewRgbCanvas: $("reviewRgbCanvas"),
  reviewOverlayCanvas: $("reviewOverlayCanvas"), reviewMaskCanvas: $("reviewMaskCanvas"),
  reviewFlagButton: $("reviewFlagButton"), reviewFlagMeta: $("reviewFlagMeta"),
  correctionReferenceSection: $("correctionReferenceSection"),
  referenceCanvas: $("referenceCanvas"), referencePreviewCanvas: $("referencePreviewCanvas"),
  referenceToggle: $("referenceToggle"),
};

const contexts = {
  image: elements.imageCanvas.getContext("2d"),
  overlay: elements.overlayCanvas.getContext("2d"),
  reference: elements.referenceCanvas.getContext("2d", { willReadFrequently: true }),
  guide: elements.guideCanvas.getContext("2d"),
  mask: elements.maskCanvas.getContext("2d", { willReadFrequently: true }),
};

const reviewContexts = {
  rgb: elements.reviewRgbCanvas.getContext("2d"),
  overlay: elements.reviewOverlayCanvas.getContext("2d"),
  mask: elements.reviewMaskCanvas.getContext("2d"),
};

const urlParameters = new URLSearchParams(window.location.search);
const token = urlParameters.get("token") || "";
const openFinalReviewOnLoad = urlParameters.get("review") === "final";
let tasks = [];
let currentIndex = -1;
let currentTask = null;
let currentStatus = "not_started";
let revision = 0;
let selectedLabel = 1;
let tool = "brush";
let dirty = false;
let editGeneration = 0;
let savePromise = null;
let autosaveTimer = null;
let drawing = false;
let lastPoint = null;
let polygon = [];
let undoStack = [];
let redoStack = [];
let initialMaskImageData = null;
let referenceMaskImageData = null;
let panning = null;
let temporaryTool = null;
let lastWheelZoom = 0;
const taskAssetCache = new Map();
const maximumUndo = 20;
let finalReviewTasks = [];
let finalReviewIndex = -1;
let finalReviewMetadata = null;
let finalReviewRgb = null;
let finalReviewMaskPixels = null;
let finalReviewLoadGeneration = 0;
let finalReviewFlagSaving = false;
const finalReviewAssetCache = new Map();

async function api(path, options = {}) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}token=${encodeURIComponent(token)}`, options);
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function imageUrl(kind, taskId) {
  return `/api/${kind}?id=${encodeURIComponent(taskId)}&token=${encodeURIComponent(token)}`;
}

function finalAssetUrl(kind, taskId) {
  return `/api/final/${kind}?id=${encodeURIComponent(taskId)}&token=${encodeURIComponent(token)}`;
}

function fetchTaskAssets(index) {
  const brief = tasks[index];
  if (!brief) return Promise.reject(new Error("Task index is out of range"));
  if (taskAssetCache.has(brief.task_id)) return taskAssetCache.get(brief.task_id);
  const request = (async () => {
    const payload = await api(`/api/task?id=${encodeURIComponent(brief.task_id)}`);
    const imagePromise = loadImage(imageUrl("image", payload.task_id));
    const maskPromise = loadImage(imageUrl("mask", payload.task_id));
    const initialPromise = payload.has_saved_mask
      ? loadImage(`${imageUrl("mask", payload.task_id)}&initial=1`)
      : maskPromise;
    const referencePromise = payload.has_reference_mask
      ? loadImage(imageUrl("reference-mask", payload.task_id))
      : Promise.resolve(null);
    const [image, mask, initialMask, referenceMask] = await Promise.all([
      imagePromise, maskPromise, initialPromise, referencePromise,
    ]);
    return { payload, image, mask, initialMask, referenceMask };
  })().catch((error) => {
    taskAssetCache.delete(brief.task_id);
    throw error;
  });
  taskAssetCache.set(brief.task_id, request);
  return request;
}

function prefetchUpcoming() {
  for (let offset = 1; offset <= 2; offset += 1) {
    const index = currentIndex + offset;
    if (index < tasks.length) fetchTaskAssets(index).catch(() => {});
  }
}

async function initialize() {
  if (!token) throw new Error("This URL has no annotator token.");
  const bootstrap = await api("/api/bootstrap");
  tasks = bootstrap.tasks;
  elements.annotatorName.textContent = bootstrap.annotator;
  if (bootstrap.final_review?.enabled) {
    finalReviewMetadata = bootstrap.final_review;
    elements.finalReviewButton.hidden = false;
    elements.finalReviewButton.textContent = `View all ${bootstrap.final_review.total} final masks`;
  }
  if (!tasks.length) {
    if (openFinalReviewOnLoad && bootstrap.final_review?.enabled) {
      await openFinalReview();
      return;
    }
    throw new Error("This annotator has no assigned editing tasks. Add &review=final to open the global final-mask review.");
  }
  populateScenes();
  updateProgress();
  if (openFinalReviewOnLoad && bootstrap.final_review?.enabled) {
    await openFinalReview();
    return;
  }
  const firstOpen = tasks.findIndex((row) => row.status !== "complete");
  await loadTask(firstOpen >= 0 ? firstOpen : 0);
}

function populateScenes() {
  const scenes = [...new Set(tasks.map((task) => task.scene_id))];
  elements.sceneSelect.replaceChildren();
  for (const scene of scenes) {
    const option = document.createElement("option");
    option.value = scene;
    const sceneTasks = tasks.filter((task) => task.scene_id === scene);
    const complete = sceneTasks.filter((task) => task.status === "complete").length;
    option.textContent = `${shortScene(scene)} · ${complete}/${sceneTasks.length}`;
    elements.sceneSelect.append(option);
  }
}

function renderSceneStrip() {
  elements.sceneStrip.replaceChildren();
  if (!currentTask) return;
  tasks.forEach((task, index) => {
    if (task.scene_id !== currentTask.scene_id) return;
    const button = document.createElement("button");
    button.className = `scene-chip ${task.status}${index === currentIndex ? " current" : ""}`;
    button.textContent = task.image_name.match(/(\d+)(?!.*\d)/)?.[1]?.replace(/^0+/, "") || String(index + 1);
    button.title = `${task.image_name} · ${task.status.replaceAll("_", " ")}`;
    button.addEventListener("click", () => loadTask(index).catch(showFatalError));
    elements.sceneStrip.append(button);
  });
}

async function loadTask(index) {
  if (index < 0 || index >= tasks.length) return;
  if (dirty) await saveCurrent("in_progress");
  cancelPolygon();
  setControlsDisabled(true);
  currentIndex = index;
  const brief = tasks[index];
  const { payload, image, mask, initialMask, referenceMask } = await fetchTaskAssets(index);
  taskAssetCache.delete(brief.task_id);
  currentTask = payload;
  revision = payload.record.revision;
  currentStatus = payload.record.status;
  dirty = false;
  editGeneration = 0;
  undoStack = [];
  redoStack = [];
  selectedLabel = payload.components[0]?.label || 1;
  configureCanvases(payload.width, payload.height);
  contexts.image.clearRect(0, 0, payload.width, payload.height);
  contexts.image.drawImage(image, 0, 0, payload.width, payload.height);
  contexts.mask.globalCompositeOperation = "source-over";
  contexts.mask.clearRect(0, 0, payload.width, payload.height);
  contexts.mask.drawImage(mask, 0, 0, payload.width, payload.height);
  const currentMask = contexts.mask.getImageData(0, 0, payload.width, payload.height);
  contexts.mask.clearRect(0, 0, payload.width, payload.height);
  contexts.mask.drawImage(initialMask, 0, 0, payload.width, payload.height);
  initialMaskImageData = contexts.mask.getImageData(0, 0, payload.width, payload.height);
  contexts.mask.putImageData(currentMask, 0, 0);
  referenceMaskImageData = null;
  contexts.reference.clearRect(0, 0, payload.width, payload.height);
  if (referenceMask) {
    contexts.reference.drawImage(referenceMask, 0, 0, payload.width, payload.height);
    referenceMaskImageData = contexts.reference.getImageData(0, 0, payload.width, payload.height);
    rebuildReferenceDisplay();
  }
  elements.correctionReferenceSection.hidden = !referenceMask;
  elements.referenceCanvas.hidden = !referenceMask || !elements.referenceToggle.checked;
  rebuildOverlay();
  renderComponents(payload.components);
  elements.noteInput.value = payload.record.note || "";
  elements.taskPosition.textContent = `Image ${index + 1} of ${tasks.length}`;
  elements.taskTitle.textContent = payload.image_name;
  if (payload.correction?.workflow === "flagged_final_mask_redo") {
    elements.taskMeta.textContent = `${shortScene(payload.scene_id)} · flagged final-mask correction · one foreground component · magenta = original 3D GT projection`;
    elements.completeButton.textContent = "Save correction & next";
  } else if (payload.adjudication?.workflow === "human_second_pass") {
    const source = payload.adjudication.original_annotator.replace("annotator_", "A");
    const action = payload.adjudication.policy_review_required
      ? "clear and redraw if the projected base remains"
      : "correct the queued quality issue";
    elements.taskMeta.textContent = `${shortScene(payload.scene_id)} · human review of ${source} · one foreground component · ${action}`;
    elements.completeButton.textContent = "Approve human mask & next";
  } else {
    elements.taskMeta.textContent = `${shortScene(payload.scene_id)} · ${payload.task_kind} · ${payload.components.length} instance${payload.components.length === 1 ? "" : "s"} · initialized from projected 3D oracle components`;
    elements.completeButton.textContent = "Complete & next";
  }
  elements.sceneSelect.value = payload.scene_id;
  renderSceneStrip();
  updateTaskStatus();
  updateUndoControls();
  updateSaveState("saved", payload.has_saved_mask ? "Saved" : "Initial mask");
  elements.previousButton.disabled = index === 0;
  elements.nextButton.disabled = index === tasks.length - 1;
  setControlsDisabled(false);
  elements.canvasViewport.scrollTo({ top: 0, left: 0 });
  elements.guideCanvas.focus();
  prefetchUpcoming();
}

function configureCanvases(width, height) {
  for (const canvas of [elements.imageCanvas, elements.overlayCanvas, elements.referenceCanvas, elements.guideCanvas, elements.maskCanvas]) {
    canvas.width = width;
    canvas.height = height;
  }
  applyZoom();
}

function rebuildReferenceDisplay() {
  if (!currentTask || !referenceMaskImageData) return;
  const { width, height } = currentTask;
  const source = referenceMaskImageData.data;
  const outline = contexts.reference.createImageData(width, height);
  const previewContext = elements.referencePreviewCanvas.getContext("2d");
  elements.referencePreviewCanvas.width = width;
  elements.referencePreviewCanvas.height = height;
  const preview = previewContext.createImageData(width, height);
  const foregroundAt = (pixel) => source[pixel * 4] > 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    const foreground = foregroundAt(pixel);
    const value = foreground ? 255 : 0;
    preview.data[offset] = value;
    preview.data[offset + 1] = value;
    preview.data[offset + 2] = value;
    preview.data[offset + 3] = 255;
    if (!foreground) continue;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const boundary = x === 0 || y === 0 || x === width - 1 || y === height - 1
      || !foregroundAt(pixel - 1) || !foregroundAt(pixel + 1)
      || !foregroundAt(pixel - width) || !foregroundAt(pixel + width);
    if (!boundary) continue;
    outline.data[offset] = 255;
    outline.data[offset + 1] = 93;
    outline.data[offset + 2] = 143;
    outline.data[offset + 3] = 255;
  }
  contexts.reference.clearRect(0, 0, width, height);
  contexts.reference.putImageData(outline, 0, 0);
  previewContext.putImageData(preview, 0, 0);
}

function renderComponents(components) {
  elements.componentList.replaceChildren();
  for (const component of components) {
    const label = document.createElement("label");
    label.className = "component-option";
    label.style.setProperty("--component-color", component.color);
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "component";
    input.value = String(component.label);
    input.checked = component.label === selectedLabel;
    input.addEventListener("change", () => { selectedLabel = component.label; });
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    const name = document.createElement("span");
    name.textContent = `Component ${component.component_id}`;
    const gt = document.createElement("small");
    gt.textContent = `GT ${component.adjacent_gt_indices.join(", ")}`;
    label.append(input, swatch, name, gt);
    elements.componentList.append(label);
  }
}

function selectComponent(label) {
  if (!currentTask?.components.some((component) => component.label === label)) return;
  selectedLabel = label;
  const input = elements.componentList.querySelector(`input[value="${label}"]`);
  if (input) input.checked = true;
}

function replaceSelectedComponent(useInitial) {
  if (!currentTask || !initialMaskImageData) return;
  cancelPolygon();
  pushUndo();
  const current = contexts.mask.getImageData(0, 0, currentTask.width, currentTask.height);
  const original = initialMaskImageData.data;
  for (let offset = 0; offset < current.data.length; offset += 4) {
    if (current.data[offset] === selectedLabel) {
      current.data[offset] = 0;
      current.data[offset + 1] = 0;
      current.data[offset + 2] = 0;
      current.data[offset + 3] = 255;
    }
    if (useInitial && original[offset] === selectedLabel) {
      current.data[offset] = selectedLabel;
      current.data[offset + 1] = selectedLabel;
      current.data[offset + 2] = selectedLabel;
      current.data[offset + 3] = 255;
    }
  }
  contexts.mask.putImageData(current, 0, 0);
  rebuildOverlay();
  markDirty();
}

function rebuildOverlay() {
  if (!currentTask) return;
  const width = currentTask.width;
  const height = currentTask.height;
  const source = contexts.mask.getImageData(0, 0, width, height).data;
  const target = contexts.overlay.createImageData(width, height);
  const colors = new Map(currentTask.components.map((component) => [component.label, hexToRgb(component.color)]));
  const edgeOnly = elements.edgeOnlyToggle.checked;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const label = source[pixel * 4];
    const color = colors.get(label);
    if (!color) continue;
    if (edgeOnly) {
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      const inside = x > 0 && x < width - 1 && y > 0 && y < height - 1;
      if (
        inside
        && source[(pixel - 1) * 4] === label
        && source[(pixel + 1) * 4] === label
        && source[(pixel - width) * 4] === label
        && source[(pixel + width) * 4] === label
      ) continue;
    }
    const offset = pixel * 4;
    target.data[offset] = color[0];
    target.data[offset + 1] = color[1];
    target.data[offset + 2] = color[2];
    target.data[offset + 3] = 255;
  }
  contexts.overlay.clearRect(0, 0, width, height);
  contexts.overlay.putImageData(target, 0, 0);
}

function setTool(nextTool) {
  if (tool === "polygon" && nextTool !== "polygon") cancelPolygon();
  tool = nextTool;
  document.querySelectorAll(".tool").forEach((button) => button.classList.toggle("active", button.dataset.tool === tool));
}

function canvasPoint(event) {
  const rect = elements.guideCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * elements.guideCanvas.width / rect.width,
    y: (event.clientY - rect.top) * elements.guideCanvas.height / rect.height,
  };
}

function beginStroke(event) {
  if (!currentTask) return;
  if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
    event.preventDefault();
    panning = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      scrollLeft: elements.canvasViewport.scrollLeft,
      scrollTop: elements.canvasViewport.scrollTop,
    };
    elements.guideCanvas.setPointerCapture(event.pointerId);
    elements.guideCanvas.style.cursor = "grabbing";
    return;
  }
  if (event.button !== 0) return;
  const point = canvasPoint(event);
  if (tool === "polygon") {
    polygon.push(point);
    drawPolygonGuide(point);
    elements.finishPolygonButton.disabled = polygon.length < 3;
    elements.cancelPolygonButton.disabled = false;
    return;
  }
  pushUndo();
  drawing = true;
  lastPoint = point;
  elements.guideCanvas.setPointerCapture(event.pointerId);
  drawSegment(point, point);
}

function continueStroke(event) {
  if (panning) {
    elements.canvasViewport.scrollLeft = panning.scrollLeft - (event.clientX - panning.clientX);
    elements.canvasViewport.scrollTop = panning.scrollTop - (event.clientY - panning.clientY);
    return;
  }
  if (!drawing) return;
  const samples = typeof event.getCoalescedEvents === "function"
    ? event.getCoalescedEvents()
    : [event];
  for (const sample of samples.length ? samples : [event]) {
    const point = canvasPoint(sample);
    drawSegment(lastPoint, point);
    lastPoint = point;
  }
}

function endStroke(event) {
  if (panning) {
    panning = null;
    if (elements.guideCanvas.hasPointerCapture(event.pointerId)) elements.guideCanvas.releasePointerCapture(event.pointerId);
    elements.guideCanvas.style.cursor = "crosshair";
    return;
  }
  if (!drawing) return;
  const point = canvasPoint(event);
  if (lastPoint) drawSegment(lastPoint, point);
  drawing = false;
  lastPoint = null;
  if (elements.guideCanvas.hasPointerCapture(event.pointerId)) elements.guideCanvas.releasePointerCapture(event.pointerId);
  // Rebuild from the exact binary label canvas so the visible result is
  // identical to what will be serialized and shown after a reload.
  rebuildOverlay();
  markDirty();
}

function drawSegment(from, to) {
  const size = Number(elements.brushSize.value);
  const label = tool === "eraser" ? 0 : selectedLabel;
  rasterizeMaskSegment(from, to, size, label);
  if (tool === "eraser") {
    stroke(contexts.overlay, from, to, size, "rgba(0,0,0,1)", "destination-out");
  } else {
    stroke(contexts.overlay, from, to, size, componentColor(selectedLabel), "source-over");
  }
}

function rasterizeMaskSegment(from, to, width, label) {
  // Do not paint low-valued labels (1, 2, ...) through the browser's
  // anti-aliased canvas compositor. Fractional edge coverage can quantize a
  // continuous stroke into disconnected dabs and can turn a higher label
  // into a lower one. Rasterize an exact, fully opaque binary capsule instead.
  const radius = Math.max(0.5, width / 2);
  const canvasWidth = elements.maskCanvas.width;
  const canvasHeight = elements.maskCanvas.height;
  const left = Math.max(0, Math.floor(Math.min(from.x, to.x) - radius - 1));
  const top = Math.max(0, Math.floor(Math.min(from.y, to.y) - radius - 1));
  const right = Math.min(canvasWidth, Math.ceil(Math.max(from.x, to.x) + radius + 1));
  const bottom = Math.min(canvasHeight, Math.ceil(Math.max(from.y, to.y) + radius + 1));
  if (right <= left || bottom <= top) return;

  const pixels = contexts.mask.getImageData(left, top, right - left, bottom - top);
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const lengthSquared = dx * dx + dy * dy;
  const radiusSquared = radius * radius;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const centerX = x + 0.5;
      const centerY = y + 0.5;
      let position = 0;
      if (lengthSquared > 0) {
        position = ((centerX - from.x) * dx + (centerY - from.y) * dy) / lengthSquared;
        position = Math.max(0, Math.min(1, position));
      }
      const nearestX = from.x + position * dx;
      const nearestY = from.y + position * dy;
      const distanceX = centerX - nearestX;
      const distanceY = centerY - nearestY;
      if (distanceX * distanceX + distanceY * distanceY > radiusSquared) continue;
      const offset = ((y - top) * (right - left) + (x - left)) * 4;
      pixels.data[offset] = label;
      pixels.data[offset + 1] = label;
      pixels.data[offset + 2] = label;
      pixels.data[offset + 3] = 255;
    }
  }
  contexts.mask.putImageData(pixels, left, top);
}

function stroke(context, from, to, width, color, operation) {
  context.save();
  context.globalCompositeOperation = operation;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = width;
  context.strokeStyle = color;
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.restore();
}

function drawPolygonGuide(cursor = null) {
  contexts.guide.clearRect(0, 0, elements.guideCanvas.width, elements.guideCanvas.height);
  if (!polygon.length) return;
  contexts.guide.save();
  contexts.guide.strokeStyle = componentColor(selectedLabel);
  contexts.guide.fillStyle = "#ffffff";
  contexts.guide.lineWidth = Math.max(1, elements.guideCanvas.width / 500);
  contexts.guide.setLineDash([7, 5]);
  contexts.guide.beginPath();
  contexts.guide.moveTo(polygon[0].x, polygon[0].y);
  for (const point of polygon.slice(1)) contexts.guide.lineTo(point.x, point.y);
  if (cursor) contexts.guide.lineTo(cursor.x, cursor.y);
  contexts.guide.stroke();
  contexts.guide.setLineDash([]);
  for (const point of polygon) {
    contexts.guide.beginPath();
    contexts.guide.arc(point.x, point.y, 3, 0, Math.PI * 2);
    contexts.guide.fill();
  }
  contexts.guide.restore();
}

function finishPolygon() {
  if (polygon.length < 3) return;
  pushUndo();
  fillPolygon(contexts.mask, `rgb(${selectedLabel},${selectedLabel},${selectedLabel})`);
  fillPolygon(contexts.overlay, componentColor(selectedLabel));
  cancelPolygon();
  if (elements.edgeOnlyToggle.checked) rebuildOverlay();
  markDirty();
}

function fillPolygon(context, color) {
  context.save();
  context.globalCompositeOperation = "source-over";
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(polygon[0].x, polygon[0].y);
  for (const point of polygon.slice(1)) context.lineTo(point.x, point.y);
  context.closePath();
  context.fill();
  context.restore();
}

function cancelPolygon() {
  polygon = [];
  contexts.guide.clearRect(0, 0, elements.guideCanvas.width, elements.guideCanvas.height);
  elements.finishPolygonButton.disabled = true;
  elements.cancelPolygonButton.disabled = true;
}

function pushUndo() {
  if (!currentTask) return;
  undoStack.push(contexts.mask.getImageData(0, 0, currentTask.width, currentTask.height));
  if (undoStack.length > maximumUndo) undoStack.shift();
  redoStack = [];
  updateUndoControls();
}

function undo() {
  if (!undoStack.length || !currentTask) return;
  redoStack.push(contexts.mask.getImageData(0, 0, currentTask.width, currentTask.height));
  contexts.mask.putImageData(undoStack.pop(), 0, 0);
  rebuildOverlay();
  updateUndoControls();
  markDirty();
}

function redo() {
  if (!redoStack.length || !currentTask) return;
  undoStack.push(contexts.mask.getImageData(0, 0, currentTask.width, currentTask.height));
  contexts.mask.putImageData(redoStack.pop(), 0, 0);
  rebuildOverlay();
  updateUndoControls();
  markDirty();
}

function markDirty() {
  dirty = true;
  editGeneration += 1;
  if (currentStatus === "complete") currentStatus = "in_progress";
  updateTaskStatus();
  updateSaveState("dirty", "Unsaved edits");
  clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => saveCurrent("in_progress").catch(showTransientError), 1800);
}

async function saveCurrent(status) {
  clearTimeout(autosaveTimer);
  if (!currentTask) return;
  if (savePromise) {
    await savePromise;
    if (!dirty && status === "in_progress") return;
  }
  const taskId = currentTask.task_id;
  const generation = editGeneration;
  const requestRevision = revision;
  const data = {
    task_id: taskId,
    status,
    revision: requestRevision,
    note: elements.noteInput.value,
    mask_data_url: elements.maskCanvas.toDataURL("image/png"),
  };
  updateSaveState("dirty", "Saving…");
  setActionButtonsDisabled(true);
  savePromise = api("/api/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  try {
    const response = await savePromise;
    if (!currentTask || currentTask.task_id !== taskId) return;
    revision = response.record.revision;
    currentStatus = response.record.status;
    tasks[currentIndex].status = currentStatus;
    tasks[currentIndex].revision = revision;
    taskAssetCache.delete(taskId);
    if (editGeneration === generation) dirty = false;
    updateSaveState(dirty ? "dirty" : "saved", dirty ? "New edits" : "Saved");
    updateTaskStatus();
    updateProgress();
    populateScenes();
    elements.sceneSelect.value = currentTask.scene_id;
    renderSceneStrip();
  } catch (error) {
    updateSaveState("error", error.status === 409 ? "Revision conflict" : "Save failed");
    if (error.status === 409) {
      window.alert("This task was saved in another tab. Reload the page before editing further; your current pixels have not overwritten the server copy.");
    }
    throw error;
  } finally {
    savePromise = null;
    setActionButtonsDisabled(false);
  }
}

function updateProgress() {
  const complete = tasks.filter((task) => task.status === "complete").length;
  const flagged = tasks.filter((task) => task.status === "needs_review").length;
  elements.progressText.textContent = `${complete} / ${tasks.length} complete`;
  elements.flagText.textContent = `${flagged} flagged`;
  elements.progressBar.style.width = `${tasks.length ? 100 * complete / tasks.length : 0}%`;
}

function updateTaskStatus() {
  elements.taskStatus.textContent = currentStatus.replaceAll("_", " ");
  elements.taskStatus.className = `status-pill ${currentStatus}`;
}

function updateSaveState(className, text) {
  elements.saveState.className = `save-state ${className}`;
  elements.saveState.textContent = text;
}

function updateUndoControls() {
  elements.undoButton.disabled = !undoStack.length;
  elements.redoButton.disabled = !redoStack.length;
}

function applyZoom(focus = null) {
  const zoom = Number(elements.zoom.value);
  elements.zoomOutput.value = `${zoom}%`;
  if (!currentTask) return;
  const oldRect = elements.canvasStage.getBoundingClientRect();
  const viewportRect = elements.canvasViewport.getBoundingClientRect();
  const focusX = focus?.clientX ?? viewportRect.left + viewportRect.width / 2;
  const focusY = focus?.clientY ?? viewportRect.top + viewportRect.height / 2;
  const relativeX = oldRect.width ? (focusX - oldRect.left) / oldRect.width : 0.5;
  const relativeY = oldRect.height ? (focusY - oldRect.top) / oldRect.height : 0.5;
  elements.canvasStage.style.width = `${currentTask.width * zoom / 100}px`;
  elements.canvasStage.style.height = `${currentTask.height * zoom / 100}px`;
  const newRect = elements.canvasStage.getBoundingClientRect();
  elements.canvasViewport.scrollLeft += newRect.left + relativeX * newRect.width - focusX;
  elements.canvasViewport.scrollTop += newRect.top + relativeY * newRect.height - focusY;
}

function changeImageZoom(delta, focus = null) {
  const minimum = Number(elements.zoom.min);
  const maximum = Number(elements.zoom.max);
  const step = Number(elements.zoom.step) || 10;
  const next = Math.max(minimum, Math.min(maximum, Number(elements.zoom.value) + delta * step));
  if (next === Number(elements.zoom.value)) return;
  elements.zoom.value = String(next);
  applyZoom(focus);
}

function applyOpacity() {
  elements.opacityOutput.value = `${elements.opacity.value}%`;
  elements.overlayCanvas.style.opacity = String(Number(elements.opacity.value) / 100);
}

function nextUnfinished() {
  for (let offset = 1; offset <= tasks.length; offset += 1) {
    const index = (currentIndex + offset) % tasks.length;
    if (tasks[index].status !== "complete") return loadTask(index);
  }
  window.alert("All assigned masks are complete. Thank you!");
}

function emptyComponentNames() {
  if (!currentTask) return [];
  const counts = new Map(currentTask.components.map((component) => [component.label, 0]));
  const data = contexts.mask.getImageData(0, 0, currentTask.width, currentTask.height).data;
  for (let offset = 0; offset < data.length; offset += 4) {
    if (counts.has(data[offset])) counts.set(data[offset], counts.get(data[offset]) + 1);
  }
  return currentTask.components
    .filter((component) => counts.get(component.label) === 0)
    .map((component) => `component ${component.component_id}`);
}

async function completeAndNext() {
  const empty = emptyComponentNames();
  if (empty.length && !window.confirm(
    `${empty.join(", ")} ${empty.length === 1 ? "has" : "have"} no visible pixels. This is valid if the target is genuinely not visible. Complete anyway?`,
  )) return;
  await saveCurrent("complete");
  await nextUnfinished();
}

async function flagAndNext() {
  await saveCurrent("needs_review");
  await nextUnfinished();
}

function taskInSelectedScene() {
  const scene = elements.sceneSelect.value;
  let index = tasks.findIndex((task) => task.scene_id === scene && task.status !== "complete");
  if (index < 0) index = tasks.findIndex((task) => task.scene_id === scene);
  if (index >= 0) loadTask(index).catch(showFatalError);
}

function setControlsDisabled(disabled) {
  elements.guideCanvas.style.pointerEvents = disabled ? "none" : "auto";
}

function setActionButtonsDisabled(disabled) {
  elements.flagButton.disabled = disabled;
  elements.saveButton.disabled = disabled;
  elements.completeButton.disabled = disabled;
  elements.clearComponentButton.disabled = disabled;
  elements.resetComponentButton.disabled = disabled;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url.split("?")[0]}`));
    image.src = url;
  });
}

function componentColor(label) {
  return currentTask?.components.find((component) => component.label === label)?.color || "#20d6c7";
}

function hexToRgb(hex) {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function shortScene(scene) {
  return scene.replace("egodelta_", "").replace("_leaf", " · leaf").replace("_tomato", " · tomato");
}

function fetchFinalReviewAssets(index) {
  const task = finalReviewTasks[index];
  if (!task) return Promise.reject(new Error("Final-review index is out of range"));
  if (finalReviewAssetCache.has(task.task_id)) return finalReviewAssetCache.get(task.task_id);
  const request = Promise.all([
    loadImage(finalAssetUrl("image", task.task_id)),
    loadImage(finalAssetUrl("mask", task.task_id)),
  ]).then(([image, mask]) => ({ image, mask })).catch((error) => {
    finalReviewAssetCache.delete(task.task_id);
    throw error;
  });
  finalReviewAssetCache.set(task.task_id, request);
  while (finalReviewAssetCache.size > 30) {
    finalReviewAssetCache.delete(finalReviewAssetCache.keys().next().value);
  }
  return request;
}

async function openFinalReview() {
  elements.finalReviewPanel.hidden = false;
  elements.reviewLoadState.textContent = "Loading index…";
  if (!finalReviewTasks.length) {
    const payload = await api("/api/final/bootstrap");
    finalReviewMetadata = payload;
    finalReviewTasks = payload.tasks;
    populateFinalReviewScenes();
    updateFinalReviewSummary();
  }
  const matchingIndex = currentTask
    ? finalReviewTasks.findIndex((task) => task.task_id === currentTask.task_id)
    : -1;
  await loadFinalReview(matchingIndex >= 0 ? matchingIndex : Math.max(finalReviewIndex, 0));
}

function closeFinalReview() {
  elements.finalReviewPanel.hidden = true;
}

function populateFinalReviewScenes() {
  elements.reviewSceneSelect.replaceChildren();
  const scenes = [...new Set(finalReviewTasks.map((task) => task.scene_id))];
  for (const scene of scenes) {
    const option = document.createElement("option");
    option.value = scene;
    const sceneTasks = finalReviewTasks.filter((task) => task.scene_id === scene);
    const flagged = sceneTasks.filter((task) => task.needs_redo).length;
    option.textContent = `${shortScene(scene)} · ${sceneTasks.length} masks${flagged ? ` · ${flagged} redo` : ""}`;
    elements.reviewSceneSelect.append(option);
  }
}

function renderFinalReviewSceneStrip() {
  elements.reviewSceneStrip.replaceChildren();
  const current = finalReviewTasks[finalReviewIndex];
  if (!current) return;
  finalReviewTasks.forEach((task, index) => {
    if (task.scene_id !== current.scene_id) return;
    const button = document.createElement("button");
    button.className = `scene-chip complete${task.needs_redo ? " needs-redo" : ""}${index === finalReviewIndex ? " current" : ""}`;
    button.textContent = task.image_name.match(/(\d+)(?!.*\d)/)?.[1]?.replace(/^0+/, "") || String(index + 1);
    button.title = task.image_name;
    button.addEventListener("click", () => loadFinalReview(index).catch(showFinalReviewError));
    elements.reviewSceneStrip.append(button);
  });
}

async function loadFinalReview(index) {
  if (index < 0 || index >= finalReviewTasks.length) return;
  const generation = ++finalReviewLoadGeneration;
  finalReviewIndex = index;
  const task = finalReviewTasks[index];
  elements.reviewLoadState.className = "save-state dirty";
  elements.reviewLoadState.textContent = "Loading…";
  elements.reviewPreviousButton.disabled = index === 0;
  elements.reviewNextButton.disabled = index === finalReviewTasks.length - 1;
  elements.reviewSceneSelect.value = task.scene_id;
  renderFinalReviewSceneStrip();

  const { image, mask } = await fetchFinalReviewAssets(index);
  if (generation !== finalReviewLoadGeneration) return;
  finalReviewRgb = image;
  const scratch = document.createElement("canvas");
  scratch.width = task.width;
  scratch.height = task.height;
  const scratchContext = scratch.getContext("2d", { willReadFrequently: true });
  scratchContext.drawImage(mask, 0, 0, task.width, task.height);
  finalReviewMaskPixels = scratchContext.getImageData(0, 0, task.width, task.height).data;
  for (const canvas of [elements.reviewRgbCanvas, elements.reviewOverlayCanvas, elements.reviewMaskCanvas]) {
    canvas.width = task.width;
    canvas.height = task.height;
  }
  renderFinalReviewCanvases();
  elements.reviewPosition.textContent = `${index + 1} / ${finalReviewTasks.length}`;
  elements.reviewTaskTitle.textContent = task.image_name;
  const source = task.mask_source?.replaceAll("_", " ") || "final human annotation";
  const annotator = task.annotator?.replace("annotator_", "annotator ") || "unknown annotator";
  elements.reviewTaskMeta.textContent = `${shortScene(task.scene_id)} · ${task.task_kind} · ${annotator} · ${source}`;
  updateFinalReviewFlagUI();
  elements.reviewLoadState.className = "save-state saved";
  elements.reviewLoadState.textContent = "Final mask";

  for (let offset = 1; offset <= 2; offset += 1) {
    if (index + offset < finalReviewTasks.length) {
      fetchFinalReviewAssets(index + offset).catch(() => {});
    }
  }
}

function updateFinalReviewSummary() {
  if (!finalReviewMetadata) return;
  const policy = finalReviewMetadata.component_policy?.replaceAll("_", " ") || "final export";
  const cleanup = finalReviewMetadata.solid_binary_masks
    ? "solid-mask cleanup verified"
    : "raw export";
  const flagged = Number(finalReviewMetadata.flagged_count || 0);
  const mine = Number(finalReviewMetadata.reviewer_flagged_count || 0);
  elements.reviewDatasetMeta.textContent = `${finalReviewMetadata.total} masks · ${policy} · ${cleanup} · ${flagged} flagged for redo${mine && mine !== flagged ? ` (${mine} by you)` : ""}`;
}

function updateFinalReviewFlagUI() {
  const task = finalReviewTasks[finalReviewIndex];
  if (!task) return;
  const names = (task.flagged_by || []).map((name) => name.replace("annotator_", "A"));
  elements.reviewFlagMeta.textContent = task.needs_redo
    ? `Needs redo · ${names.join(", ")}`
    : "Not flagged";
  elements.reviewFlagMeta.className = `review-flag-meta${task.needs_redo ? " needs-redo" : ""}`;
  elements.reviewFlagButton.className = task.flagged ? "warning" : "secondary";
  elements.reviewFlagButton.textContent = task.flagged
    ? "Flagged for redo — undo"
    : task.needs_redo
      ? "Also flag for redo (F)"
      : "Flag for redo (F)";
  elements.reviewFlagButton.disabled = finalReviewFlagSaving || !finalReviewMetadata?.flagging_enabled;
}

async function toggleFinalReviewFlag() {
  const task = finalReviewTasks[finalReviewIndex];
  if (!task || finalReviewFlagSaving || !finalReviewMetadata?.flagging_enabled) return;
  finalReviewFlagSaving = true;
  updateFinalReviewFlagUI();
  elements.reviewLoadState.className = "save-state dirty";
  elements.reviewLoadState.textContent = "Saving flag…";
  try {
    const response = await api("/api/final/flag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: task.task_id, flagged: !task.flagged }),
    });
    Object.assign(task, response.state);
    finalReviewMetadata.flagged_count = response.state.flagged_count;
    finalReviewMetadata.reviewer_flagged_count = response.state.reviewer_flagged_count;
    const selectedScene = task.scene_id;
    populateFinalReviewScenes();
    elements.reviewSceneSelect.value = selectedScene;
    renderFinalReviewSceneStrip();
    updateFinalReviewSummary();
    elements.reviewLoadState.className = "save-state saved";
    elements.reviewLoadState.textContent = task.flagged ? "Redo flag saved" : "Redo flag removed";
  } finally {
    finalReviewFlagSaving = false;
    updateFinalReviewFlagUI();
  }
}

function renderFinalReviewCanvases() {
  const task = finalReviewTasks[finalReviewIndex];
  if (!task || !finalReviewRgb || !finalReviewMaskPixels) return;
  const { width, height } = task;
  reviewContexts.rgb.clearRect(0, 0, width, height);
  reviewContexts.rgb.drawImage(finalReviewRgb, 0, 0, width, height);
  reviewContexts.overlay.clearRect(0, 0, width, height);
  reviewContexts.overlay.drawImage(finalReviewRgb, 0, 0, width, height);

  const standalone = reviewContexts.mask.createImageData(width, height);
  const overlay = reviewContexts.overlay.createImageData(width, height);
  const opacity = Number(elements.reviewOpacity.value) / 100;
  const boundaryOnly = elements.reviewBoundaryToggle.checked;
  const foregroundAt = (pixel) => finalReviewMaskPixels[pixel * 4] > 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    const foreground = foregroundAt(pixel);
    const value = foreground ? 255 : 0;
    standalone.data[offset] = value;
    standalone.data[offset + 1] = value;
    standalone.data[offset + 2] = value;
    standalone.data[offset + 3] = 255;
    if (!foreground) continue;

    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const boundary = x === 0 || y === 0 || x === width - 1 || y === height - 1
      || !foregroundAt(pixel - 1) || !foregroundAt(pixel + 1)
      || !foregroundAt(pixel - width) || !foregroundAt(pixel + width);
    if (boundaryOnly && !boundary) continue;
    overlay.data[offset] = 32;
    overlay.data[offset + 1] = 214;
    overlay.data[offset + 2] = 199;
    overlay.data[offset + 3] = Math.round(255 * opacity);
  }
  reviewContexts.mask.putImageData(standalone, 0, 0);
  const overlayLayer = document.createElement("canvas");
  overlayLayer.width = width;
  overlayLayer.height = height;
  overlayLayer.getContext("2d").putImageData(overlay, 0, 0);
  reviewContexts.overlay.drawImage(overlayLayer, 0, 0);
  elements.reviewOpacityOutput.value = `${elements.reviewOpacity.value}%`;
}

function loadFinalReviewScene() {
  const index = finalReviewTasks.findIndex(
    (task) => task.scene_id === elements.reviewSceneSelect.value,
  );
  if (index >= 0) loadFinalReview(index).catch(showFinalReviewError);
}

function showFinalReviewError(error) {
  console.error(error);
  elements.reviewLoadState.className = "save-state error";
  elements.reviewLoadState.textContent = error.message || "Review failed";
}

function showTransientError(error) {
  console.error(error);
  updateSaveState("error", error.message || "Error");
}

function showFatalError(error) {
  console.error(error);
  elements.errorMessage.textContent = error.message || String(error);
  elements.errorPanel.hidden = false;
}

document.querySelectorAll(".tool").forEach((button) => button.addEventListener("click", () => setTool(button.dataset.tool)));
elements.guideCanvas.addEventListener("pointerdown", beginStroke);
elements.guideCanvas.addEventListener("pointermove", (event) => {
  if (drawing || panning) continueStroke(event);
  else if (tool === "polygon" && polygon.length) drawPolygonGuide(canvasPoint(event));
});
elements.guideCanvas.addEventListener("pointerup", endStroke);
elements.guideCanvas.addEventListener("pointercancel", endStroke);
elements.guideCanvas.addEventListener("contextmenu", (event) => { event.preventDefault(); cancelPolygon(); });
elements.previousButton.addEventListener("click", () => loadTask(currentIndex - 1).catch(showFatalError));
elements.nextButton.addEventListener("click", () => loadTask(currentIndex + 1).catch(showFatalError));
elements.nextOpenButton.addEventListener("click", () => nextUnfinished().catch(showFatalError));
elements.sceneSelect.addEventListener("change", taskInSelectedScene);
elements.undoButton.addEventListener("click", undo);
elements.redoButton.addEventListener("click", redo);
elements.clearComponentButton.addEventListener("click", () => replaceSelectedComponent(false));
elements.resetComponentButton.addEventListener("click", () => replaceSelectedComponent(true));
elements.finishPolygonButton.addEventListener("click", finishPolygon);
elements.cancelPolygonButton.addEventListener("click", cancelPolygon);
elements.brushSize.addEventListener("input", () => { elements.brushOutput.value = `${elements.brushSize.value} px`; });
elements.opacity.addEventListener("input", applyOpacity);
elements.zoom.addEventListener("input", () => applyZoom());
elements.canvasViewport.addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  const now = performance.now();
  if (now - lastWheelZoom < 45) return;
  lastWheelZoom = now;
  changeImageZoom(event.deltaY < 0 ? 1 : -1, event);
}, { passive: false });
elements.overlayToggle.addEventListener("change", () => { elements.overlayCanvas.hidden = !elements.overlayToggle.checked; });
elements.edgeOnlyToggle.addEventListener("change", rebuildOverlay);
elements.referenceToggle.addEventListener("change", () => {
  elements.referenceCanvas.hidden = !elements.referenceToggle.checked || !referenceMaskImageData;
});
elements.noteInput.addEventListener("input", markDirty);
elements.saveButton.addEventListener("click", () => saveCurrent("in_progress").catch(showTransientError));
elements.flagButton.addEventListener("click", () => flagAndNext().catch(showTransientError));
elements.completeButton.addEventListener("click", () => completeAndNext().catch(showTransientError));
elements.finalReviewButton.addEventListener("click", () => openFinalReview().catch(showFinalReviewError));
elements.closeFinalReviewButton.addEventListener("click", closeFinalReview);
elements.reviewPreviousButton.addEventListener("click", () => loadFinalReview(finalReviewIndex - 1).catch(showFinalReviewError));
elements.reviewNextButton.addEventListener("click", () => loadFinalReview(finalReviewIndex + 1).catch(showFinalReviewError));
elements.reviewSceneSelect.addEventListener("change", loadFinalReviewScene);
elements.reviewOpacity.addEventListener("input", renderFinalReviewCanvases);
elements.reviewBoundaryToggle.addEventListener("change", renderFinalReviewCanvases);
elements.reviewFlagButton.addEventListener("click", () => toggleFinalReviewFlag().catch(showFinalReviewError));

document.addEventListener("keydown", (event) => {
  if (elements.finalReviewPanel.hidden) return;
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopImmediatePropagation();
    closeFinalReview();
  } else if (!typing && ["ArrowLeft", "a", "A"].includes(event.key)) {
    event.preventDefault();
    event.stopImmediatePropagation();
    loadFinalReview(finalReviewIndex - 1).catch(showFinalReviewError);
  } else if (!typing && ["ArrowRight", "d", "D"].includes(event.key)) {
    event.preventDefault();
    event.stopImmediatePropagation();
    loadFinalReview(finalReviewIndex + 1).catch(showFinalReviewError);
  } else if (!typing && ["f", "F"].includes(event.key)) {
    event.preventDefault();
    event.stopImmediatePropagation();
    toggleFinalReviewFlag().catch(showFinalReviewError);
  }
  // The review panel is read-only. Keep all remaining keystrokes away from
  // the hidden annotation editor while still allowing normal browser/default
  // behavior inside the review controls.
  event.stopImmediatePropagation();
}, true);

window.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? redo() : undo(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") { event.preventDefault(); redo(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrent("in_progress").catch(showTransientError); return; }
  if (
    (event.ctrlKey || event.metaKey)
    && (elements.canvasViewport.matches(":hover") || document.activeElement === elements.guideCanvas)
    && ["+", "=", "-", "0"].includes(event.key)
  ) {
    event.preventDefault();
    if (event.key === "0") {
      elements.zoom.value = "100";
      applyZoom();
    } else {
      changeImageZoom(event.key === "-" ? -1 : 1);
    }
    return;
  }
  if (typing) return;
  const key = event.key.toLowerCase();
  if (/^[1-8]$/.test(key)) { event.preventDefault(); selectComponent(Number(key)); }
  else if (key === "b") { event.preventDefault(); setTool("brush"); }
  else if (key === "e") { event.preventDefault(); setTool("eraser"); }
  else if (key === "p") { event.preventDefault(); setTool("polygon"); }
  else if (key === "x" && !event.repeat && tool !== "polygon") {
    event.preventDefault();
    temporaryTool = tool;
    setTool("eraser");
  }
  else if (key === "enter" && tool === "polygon") { event.preventDefault(); finishPolygon(); }
  else if (key === "escape") { event.preventDefault(); cancelPolygon(); }
  else if (key === "a") { event.preventDefault(); loadTask(currentIndex - 1).catch(showFatalError); }
  else if (key === "d") { event.preventDefault(); loadTask(currentIndex + 1).catch(showFatalError); }
  else if (key === "r") { event.preventDefault(); replaceSelectedComponent(true); }
  else if (key === "delete" || key === "backspace") { event.preventDefault(); replaceSelectedComponent(false); }
  else if (key === "f") { event.preventDefault(); flagAndNext().catch(showTransientError); }
  else if (key === "v") {
    event.preventDefault();
    elements.edgeOnlyToggle.checked = !elements.edgeOnlyToggle.checked;
    rebuildOverlay();
  }
  else if ((event.code === "Space" || key === " ") && !event.repeat) {
    event.preventDefault();
    completeAndNext().catch(showTransientError);
  }
  else if (key === "[") { event.preventDefault(); elements.brushSize.value = Math.max(2, Number(elements.brushSize.value) - 2); elements.brushSize.dispatchEvent(new Event("input")); }
  else if (key === "]") { event.preventDefault(); elements.brushSize.value = Math.min(80, Number(elements.brushSize.value) + 2); elements.brushSize.dispatchEvent(new Event("input")); }
});

window.addEventListener("keyup", (event) => {
  if (event.key.toLowerCase() === "x" && temporaryTool) {
    setTool(temporaryTool);
    temporaryTool = null;
  }
});

window.addEventListener("beforeunload", (event) => {
  if (dirty) { event.preventDefault(); event.returnValue = ""; }
});

applyOpacity();
initialize().catch(showFatalError);
