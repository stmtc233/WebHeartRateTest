import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

const MIN_BPM = 45;
const MAX_BPM = 180;
const MAX_SIGNAL_SECONDS = 32;
const MIN_ANALYSIS_SECONDS = 8;
const TRACK_TTL_SECONDS = 1.4;
const ESTIMATE_INTERVAL_MS = 900;
const MIN_INFERENCE_INTERVAL_MS = 100;
const CAMERA_WARMUP_MS = 700;
const MAX_PROCESSING_SIDE = 480;
const MIN_FACE_BOX_AREA_RATIO = 0.015;
const MAX_FACE_BOX_AREA_RATIO = 0.62;
const MIN_FACE_BOX_ASPECT_RATIO = 0.5;
const MAX_FACE_BOX_ASPECT_RATIO = 1.9;

const palette = [
  "#0d8b72",
  "#d85c45",
  "#6a62b7",
  "#b97b16",
  "#1477a3",
  "#7a8f2b",
];

const el = {
  activeTracksText: document.querySelector("#activeTracksText"),
  cameraFeed: document.querySelector("#cameraFeed"),
  cameraSelect: document.querySelector("#cameraSelect"),
  faceCountText: document.querySelector("#faceCountText"),
  fpsText: document.querySelector("#fpsText"),
  maxFacesSelect: document.querySelector("#maxFacesSelect"),
  modelStatus: document.querySelector("#modelStatus"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  peopleList: document.querySelector("#peopleList"),
  qualityText: document.querySelector("#qualityText"),
  resolutionText: document.querySelector("#resolutionText"),
  runtimeText: document.querySelector("#runtimeText"),
  sampleCanvas: document.querySelector("#sampleCanvas"),
  sampleRateText: document.querySelector("#sampleRateText"),
  stagePlaceholder: document.querySelector("#stagePlaceholder"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  windowSelect: document.querySelector("#windowSelect"),
};

const overlayCtx = el.overlayCanvas.getContext("2d");
const sampleCtx = el.sampleCanvas.getContext("2d", { willReadFrequently: true });

let faceLandmarker;
let stream;
let rafId = 0;
let running = false;
let appStartMs = 0;
let nextTrackId = 1;
let tracks = new Map();
let fpsMeter = resetFpsMeter();
let lastUiRenderMs = 0;
let lastAnalysisMs = 0;
let lastVideoWidth = 0;
let lastVideoHeight = 0;
let cameraWarmupUntilMs = 0;
let currentFacingMode = "user";
let frameLoopMode = "raf";
let frameLoopCancelId = 0;

window.lucide?.createIcons();

el.startButton.addEventListener("click", start);
el.stopButton.addEventListener("click", stop);
el.cameraSelect.addEventListener("change", () => {
  if (running) {
    restartCamera().catch((error) => {
      console.error(error);
      stop();
      setStatus(cameraErrorMessage(error), "error");
    });
  }
});
el.maxFacesSelect.addEventListener("change", async () => {
  if (faceLandmarker) {
    await faceLandmarker.setOptions({ numFaces: getMaxFaces() });
  }
});

window.addEventListener("resize", () => fitCanvasToVideo());

initCameraList();

async function initCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    fillCameraSelect(devices);
  } catch {
    // Device labels are often unavailable before permission; start() refreshes them.
  }
}

async function start() {
  if (running) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("当前浏览器不支持摄像头", "error");
    return;
  }

  try {
    setButtons(true);
    setStatus("加载人脸模型", "busy");
    await ensureFaceLandmarker();

    setStatus("打开摄像头", "busy");
    await openCamera();

    tracks = new Map();
    nextTrackId = 1;
    appStartMs = performance.now();
    fpsMeter = resetFpsMeter(appStartMs);
    lastUiRenderMs = 0;
    lastAnalysisMs = 0;
    running = true;
    el.stagePlaceholder.classList.add("is-hidden");
    setStatus("实时检测中", "ready");
    startFrameLoop();
  } catch (error) {
    console.error(error);
    stop();
    setStatus(cameraErrorMessage(error), "error");
  } finally {
    setButtons(false);
  }
}

function stop() {
  running = false;
  stopFrameLoop();
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  el.cameraFeed.srcObject = null;
  el.stagePlaceholder.classList.remove("is-hidden");
  tracks.clear();
  clearCanvas(overlayCtx, el.overlayCanvas);
  renderPeople();
  updateStats(0);
  fpsMeter = resetFpsMeter();
  lastAnalysisMs = 0;
  lastVideoWidth = 0;
  lastVideoHeight = 0;
  cameraWarmupUntilMs = 0;
  el.stopButton.disabled = true;
  el.startButton.disabled = false;
  setStatus("待启动", "idle");
}

async function restartCamera() {
  const wasRunning = running;
  running = false;
  stopFrameLoop();
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  tracks.clear();
  if (wasRunning) {
    await openCamera();
    fpsMeter = resetFpsMeter();
    lastAnalysisMs = 0;
    running = true;
    startFrameLoop();
  }
}

async function ensureFaceLandmarker() {
  if (faceLandmarker) return;
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    numFaces: getMaxFaces(),
    minFaceDetectionConfidence: 0.72,
    minFacePresenceConfidence: 0.68,
    minTrackingConfidence: 0.55,
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
  };

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, options);
  } catch (gpuError) {
    console.warn("GPU delegate failed, falling back to CPU.", gpuError);
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      ...options,
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "CPU",
      },
    });
  }
}

async function openCamera() {
  const selectedDeviceId = el.cameraSelect.value;
  const constraints = {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 },
      ...(selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId } }
        : { facingMode: { ideal: currentFacingMode } }),
    },
  };

  stream = await navigator.mediaDevices.getUserMedia(constraints);
  el.cameraFeed.srcObject = stream;
  await waitForVideo(el.cameraFeed);

  const videoTrack = stream.getVideoTracks()[0];
  const settings = videoTrack?.getSettings?.() ?? {};
  currentFacingMode = settings.facingMode || currentFacingMode;

  fitCanvasToVideo(true);
  const devices = await navigator.mediaDevices.enumerateDevices();
  fillCameraSelect(devices, settings.deviceId);
}

function waitForVideo(video) {
  return new Promise((resolve) => {
    let settled = false;

    const finish = async () => {
      try {
        await video.play();
      } catch {
        // Muted autoplay should normally work after a user gesture, but browsers vary.
      }

      if (!video.videoWidth || !video.videoHeight) {
        requestAnimationFrame(finish);
        return;
      }

      if (settled) return;
      settled = true;
      resolve();
    };

    if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
      finish();
      return;
    }

    const onReady = () => {
      finish();
    };

    video.addEventListener("loadedmetadata", onReady, { once: true });
    video.addEventListener("loadeddata", onReady, { once: true });
  });
}

function fillCameraSelect(devices, activeDeviceId = el.cameraSelect.value) {
  const videoDevices = devices.filter((device) => device.kind === "videoinput");
  const options = videoDevices.map((device, index) => {
    const label = device.label || `摄像头 ${index + 1}`;
    const selected = device.deviceId === activeDeviceId ? "selected" : "";
    return `<option value="${escapeAttr(device.deviceId)}" ${selected}>${escapeHtml(label)}</option>`;
  });

  if (!options.length) {
    el.cameraSelect.innerHTML = '<option value="">默认摄像头</option>';
    return;
  }

  el.cameraSelect.innerHTML = options.join("");
}

function processFrame(nowMs) {
  if (!running) return;

  const video = el.cameraFeed;

  if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
    fitCanvasToVideo(false, nowMs);

    if (nowMs >= cameraWarmupUntilMs && nowMs - lastAnalysisMs >= MIN_INFERENCE_INTERVAL_MS) {
      lastAnalysisMs = nowMs;
      drawSampleFrame();
      analyzeFrame(nowMs);
    }
  }

  if (nowMs - lastUiRenderMs > 220) {
    renderPeople();
    updateStats(nowMs);
    lastUiRenderMs = nowMs;
  }

  if (frameLoopMode === "raf") {
    rafId = requestAnimationFrame(processFrame);
  }
}

function startFrameLoop() {
  frameLoopMode = "raf";
  frameLoopCancelId = 0;

  if (typeof el.cameraFeed.requestVideoFrameCallback === "function") {
    frameLoopMode = "video";
    scheduleVideoFrame();
    return;
  }

  rafId = requestAnimationFrame(processFrame);
}

function stopFrameLoop() {
  if (
    frameLoopMode === "video" &&
    typeof el.cameraFeed.cancelVideoFrameCallback === "function" &&
    frameLoopCancelId
  ) {
    el.cameraFeed.cancelVideoFrameCallback(frameLoopCancelId);
  }
  frameLoopCancelId = 0;
  cancelAnimationFrame(rafId);
  rafId = 0;
}

function scheduleVideoFrame() {
  if (!running || frameLoopMode !== "video") return;
  frameLoopCancelId = el.cameraFeed.requestVideoFrameCallback(onVideoFrame);
}

function onVideoFrame(nowMs) {
  if (!running || frameLoopMode !== "video") return;
  processFrame(typeof nowMs === "number" ? nowMs : performance.now());
  scheduleVideoFrame();
}

function analyzeFrame(nowMs) {
  try {
    const results = faceLandmarker.detectForVideo(el.sampleCanvas, nowMs);
    const timestamp = nowMs / 1000;
    const detections = collectDetections(results.faceLandmarks ?? []);
    matchTracks(detections, timestamp, nowMs);
    drawOverlay(timestamp);
    tickFps(nowMs);
  } catch (error) {
    console.warn("Face detection failed for this frame.", error);
  }
}

function drawSampleFrame() {
  const { width: canvasWidth, height: canvasHeight } = el.sampleCanvas;
  if (!canvasWidth || !canvasHeight) return;
  const { videoWidth, videoHeight } = el.cameraFeed;
  if (!videoWidth || !videoHeight) return;
  sampleCtx.drawImage(el.cameraFeed, 0, 0, canvasWidth, canvasHeight);
}

function fitCanvasToVideo(force = false, nowMs = performance.now()) {
  const { videoWidth, videoHeight } = el.cameraFeed;
  if (!videoWidth || !videoHeight) return;
  if (!force && videoWidth === lastVideoWidth && videoHeight === lastVideoHeight) {
    return;
  }
  const size = processingSize(videoWidth, videoHeight);
  lastVideoWidth = videoWidth;
  lastVideoHeight = videoHeight;
  el.overlayCanvas.width = size.width;
  el.overlayCanvas.height = size.height;
  el.sampleCanvas.width = size.width;
  el.sampleCanvas.height = size.height;
  el.resolutionText.textContent = `${videoWidth} x ${videoHeight}`;
  tracks.clear();
  nextTrackId = 1;
  clearCanvas(overlayCtx, el.overlayCanvas);
  renderPeople();
  cameraWarmupUntilMs = nowMs + CAMERA_WARMUP_MS;
  lastAnalysisMs = 0;
}

function processingSize(width, height) {
  const scale = Math.min(1, MAX_PROCESSING_SIDE / Math.max(width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

function collectDetections(faceLandmarks) {
  return faceLandmarks
    .map((landmarks) => {
      const box = landmarksToBox(landmarks);
      if (!box || box.width < 28 || box.height < 28 || !isLikelyFaceBox(box)) {
        return null;
      }
      const rects = pulseRects(box);
      const rgb = sampleRgb(rects);
      return {
        box,
        center: {
          x: box.x + box.width / 2,
          y: box.y + box.height / 2,
        },
        landmarks,
        rects,
        rgb,
      };
    })
    .filter(Boolean);
}

function landmarksToBox(landmarks) {
  const width = el.sampleCanvas.width;
  const height = el.sampleCanvas.height;
  if (!width || !height || !landmarks.length) return null;

  const xs = [];
  const ys = [];

  for (const point of landmarks) {
    const x = point.x * width;
    const y = point.y * height;
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    xs.push(x);
    ys.push(y);
  }

  if (xs.length < 80 || ys.length < 80) return null;

  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);

  const minX = percentile(xs, 0.03);
  const maxX = percentile(xs, 0.97);
  const minY = percentile(ys, 0.03);
  const maxY = percentile(ys, 0.97);

  const rawWidth = maxX - minX;
  const rawHeight = maxY - minY;
  const padX = rawWidth * 0.08;
  const padY = rawHeight * 0.12;

  return clampBox({
    x: minX - padX,
    y: minY - padY,
    width: rawWidth + padX * 2,
    height: rawHeight + padY * 2,
  });
}

function isLikelyFaceBox(box) {
  const width = el.sampleCanvas.width || 1;
  const height = el.sampleCanvas.height || 1;
  const areaRatio = (box.width * box.height) / (width * height);
  const aspectRatio = box.width / box.height;

  if (areaRatio < MIN_FACE_BOX_AREA_RATIO || areaRatio > MAX_FACE_BOX_AREA_RATIO) {
    return false;
  }

  return (
    aspectRatio >= MIN_FACE_BOX_ASPECT_RATIO &&
    aspectRatio <= MAX_FACE_BOX_ASPECT_RATIO
  );
}

function pulseRects(box) {
  const rects = [
    {
      role: "forehead",
      x: box.x + box.width * 0.28,
      y: box.y + box.height * 0.16,
      width: box.width * 0.44,
      height: box.height * 0.17,
    },
    {
      role: "left-cheek",
      x: box.x + box.width * 0.18,
      y: box.y + box.height * 0.48,
      width: box.width * 0.22,
      height: box.height * 0.2,
    },
    {
      role: "right-cheek",
      x: box.x + box.width * 0.6,
      y: box.y + box.height * 0.48,
      width: box.width * 0.22,
      height: box.height * 0.2,
    },
  ];

  return rects.map(clampBox).filter((rect) => rect.width >= 6 && rect.height >= 6);
}

function sampleRgb(rects) {
  let rTotal = 0;
  let gTotal = 0;
  let bTotal = 0;
  let count = 0;

  for (const rect of rects) {
    const x = Math.round(rect.x);
    const y = Math.round(rect.y);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    let imageData;

    try {
      imageData = sampleCtx.getImageData(x, y, width, height);
    } catch {
      continue;
    }

    const data = imageData.data;
    const stride = Math.max(1, Math.floor(Math.sqrt((width * height) / 900)));

    for (let row = 0; row < height; row += stride) {
      for (let col = 0; col < width; col += stride) {
        const index = (row * width + col) * 4;
        const red = data[index];
        const green = data[index + 1];
        const blue = data[index + 2];
        const brightness = red + green + blue;

        if (brightness < 55 || brightness > 740) continue;
        if (green < 22) continue;

        rTotal += red;
        gTotal += green;
        bTotal += blue;
        count += 1;
      }
    }
  }

  if (!count) return null;
  return {
    r: rTotal / count,
    g: gTotal / count,
    b: bTotal / count,
    pixels: count,
  };
}

function matchTracks(detections, timestamp, nowMs) {
  const assignments = [];
  const usedTracks = new Set();
  const usedDetections = new Set();
  const activeTracks = [...tracks.values()].filter(
    (track) => timestamp - track.lastSeen < TRACK_TTL_SECONDS,
  );

  for (const track of activeTracks) {
    detections.forEach((detection, index) => {
      const distance = pointDistance(track.center, detection.center);
      const gate = Math.max(70, track.box.width * 0.8, detection.box.width * 0.8);
      if (distance < gate) {
        assignments.push({ track, detection, index, distance });
      }
    });
  }

  assignments.sort((a, b) => a.distance - b.distance);

  for (const item of assignments) {
    if (usedTracks.has(item.track.id) || usedDetections.has(item.index)) continue;
    updateTrack(item.track, item.detection, timestamp, nowMs);
    usedTracks.add(item.track.id);
    usedDetections.add(item.index);
  }

  detections.forEach((detection, index) => {
    if (usedDetections.has(index)) return;
    const track = createTrack(detection, timestamp);
    updateTrack(track, detection, timestamp, nowMs);
    tracks.set(track.id, track);
  });

  for (const [id, track] of tracks) {
    if (timestamp - track.lastSeen > TRACK_TTL_SECONDS) {
      tracks.delete(id);
    }
  }
}

function createTrack(detection, timestamp) {
  const id = nextTrackId++;
  return {
    id,
    box: detection.box,
    center: detection.center,
    color: palette[(id - 1) % palette.length],
    confidence: 0,
    samples: [],
    firstSeen: timestamp,
    lastEstimateMs: 0,
    lastSeen: timestamp,
    motion: 0,
    bpmHistory: [],
    pulseWave: [],
    sampleRate: 0,
    status: "采集中",
  };
}

function updateTrack(track, detection, timestamp, nowMs) {
  const motionPixels = pointDistance(track.center, detection.center);
  const scale = Math.max(detection.box.width, detection.box.height, 1);
  const normalizedMotion = motionPixels / scale;

  track.motion = track.motion * 0.82 + normalizedMotion * 0.18;
  track.box = detection.box;
  track.center = detection.center;
  track.rects = detection.rects;
  track.lastSeen = timestamp;

  if (detection.rgb) {
    track.samples.push({
      t: timestamp,
      r: detection.rgb.r,
      g: detection.rgb.g,
      b: detection.rgb.b,
      pixels: detection.rgb.pixels,
      motion: normalizedMotion,
    });
    trimSamples(track, timestamp);
  }

  if (nowMs - track.lastEstimateMs > ESTIMATE_INTERVAL_MS) {
    estimateHeartRate(track);
    track.lastEstimateMs = nowMs;
  }
}

function trimSamples(track, timestamp) {
  const oldest = timestamp - MAX_SIGNAL_SECONDS;
  while (track.samples.length && track.samples[0].t < oldest) {
    track.samples.shift();
  }
}

function estimateHeartRate(track) {
  const windowSeconds = Number(el.windowSelect.value);
  const samples = recentSamples(track.samples, windowSeconds);

  if (samples.length < 45) {
    track.status = "采集中";
    track.confidence = 0;
    track.pulseWave = [];
    track.sampleRate = 0;
    return;
  }

  const duration = samples[samples.length - 1].t - samples[0].t;
  track.sampleRate = samples.length / Math.max(duration, 0.1);

  if (duration < MIN_ANALYSIS_SECONDS) {
    track.status = "采集中";
    track.confidence = clamp(duration / MIN_ANALYSIS_SECONDS, 0, 0.42);
    track.pulseWave = ecgShape(normalize(makeChromSignal(samples).slice(-80)));
    return;
  }

  const signal = removeTrend(makeChromSignal(samples), samples);
  if (signal.length < 20) {
    track.status = "信号弱";
    track.confidence = 0;
    return;
  }

  const spectrum = scanSpectrum(samples, signal);
  const ratio = spectrum.bestPower / Math.max(spectrum.medianPower, 1e-9);
  const avgMotion =
    samples.reduce((sum, sample) => sum + sample.motion, 0) / samples.length;
  const avgPixels =
    samples.reduce((sum, sample) => sum + sample.pixels, 0) / samples.length;

  const motionFactor = clamp(1 - avgMotion * 1.35, 0.12, 1);
  const areaFactor = clamp((avgPixels - 60) / 500, 0.35, 1);
  const durationFactor = clamp(duration / 18, 0.55, 1);
  const confidence = clamp(((ratio - 1.8) / 6) * motionFactor * areaFactor * durationFactor, 0, 1);

  track.confidence = confidence;

  if (confidence >= 0.18 && Number.isFinite(spectrum.bpm)) {
    updateBpmTrack(track, spectrum.bpm, confidence, samples[samples.length - 1].t);
  }

  track.pulseWave = ecgShape(normalize(signal.slice(-96)));
  if (confidence > 0.62) {
    track.status = "稳定";
  } else if (confidence > 0.32) {
    track.status = "可参考";
  } else {
    track.status = "信号弱";
  }
}

function recentSamples(samples, seconds) {
  if (!samples.length) return [];
  const cutoff = samples[samples.length - 1].t - seconds;
  return samples.filter((sample) => sample.t >= cutoff);
}

function updateBpmTrack(track, candidateBpm, confidence, timestamp) {
  if (!track.bpmHistory) track.bpmHistory = [];
  track.bpmHistory.push({
    bpm: candidateBpm,
    confidence,
    t: timestamp,
  });

  const historyWindow = 24;
  track.bpmHistory = track.bpmHistory.filter(
    (item) => timestamp - item.t <= historyWindow && item.confidence >= 0.12,
  );

  const reliable = track.bpmHistory.filter((item) => item.confidence >= 0.3);
  if (!reliable.length) {
    if (!Number.isFinite(track.bpm) && confidence >= 0.55) {
      track.bpm = candidateBpm;
    }
    return;
  }

  const sorted = reliable.map((item) => item.bpm).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const band = reliable.filter((item) => Math.abs(item.bpm - median) <= 5);
  const pool = band.length >= 3 ? band : reliable;
  const weighted = pool.reduce(
    (sum, item) => {
      sum.weight += item.confidence;
      sum.value += item.bpm * item.confidence;
      return sum;
    },
    { weight: 0, value: 0 },
  );
  const target = weighted.weight ? weighted.value / weighted.weight : median;

  if (!Number.isFinite(track.bpm)) {
    if (pool.length < 2 && confidence < 0.72) return;
    track.bpm = target;
    return;
  }

  if (confidence < 0.38 && Math.abs(target - track.bpm) > 6) {
    return;
  }

  const trust = clamp((confidence - 0.3) / 0.5, 0, 1);
  const maxStep = 1.5 + trust * 3.0;
  const limitedTarget = clamp(target, track.bpm - maxStep, track.bpm + maxStep);
  const alpha = 0.05 + trust * 0.18;
  track.bpm = clamp(track.bpm + (limitedTarget - track.bpm) * alpha, MIN_BPM, MAX_BPM);
}

function makeChromSignal(samples) {
  const mean = samples.reduce(
    (sum, sample) => {
      sum.r += sample.r;
      sum.g += sample.g;
      sum.b += sample.b;
      return sum;
    },
    { r: 0, g: 0, b: 0 },
  );

  mean.r /= samples.length;
  mean.g /= samples.length;
  mean.b /= samples.length;

  const x = [];
  const y = [];
  for (const sample of samples) {
    const rn = sample.r / mean.r - 1;
    const gn = sample.g / mean.g - 1;
    const bn = sample.b / mean.b - 1;
    x.push(3 * rn - 2 * gn);
    y.push(1.5 * rn + gn - 1.5 * bn);
  }

  const stdX = standardDeviation(x);
  const stdY = standardDeviation(y) || 1;
  const alpha = stdX / stdY;
  return x.map((value, index) => value - alpha * y[index]);
}

function removeTrend(values, samples) {
  const n = values.length;
  const t0 = samples[0].t;
  let sumT = 0;
  let sumV = 0;
  let sumTT = 0;
  let sumTV = 0;

  for (let i = 0; i < n; i += 1) {
    const t = samples[i].t - t0;
    sumT += t;
    sumV += values[i];
    sumTT += t * t;
    sumTV += t * values[i];
  }

  const denom = n * sumTT - sumT * sumT || 1;
  const slope = (n * sumTV - sumT * sumV) / denom;
  const intercept = (sumV - slope * sumT) / n;
  return values.map((value, index) => {
    const t = samples[index].t - t0;
    return value - (slope * t + intercept);
  });
}

function scanSpectrum(samples, signal) {
  const n = signal.length;
  const t0 = samples[0].t;
  const powers = [];
  let bestPower = 0;
  let bestBpm = 0;
  const bpmStep = 0.5;

  for (let bpm = MIN_BPM; bpm <= MAX_BPM; bpm += bpmStep) {
    const freq = bpm / 60;
    let re = 0;
    let im = 0;

    for (let i = 0; i < n; i += 1) {
      const t = samples[i].t - t0;
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
      const angle = 2 * Math.PI * freq * t;
      const value = signal[i] * window;
      re += value * Math.cos(angle);
      im -= value * Math.sin(angle);
    }

    const power = re * re + im * im;
    powers.push(power);
    if (power > bestPower) {
      bestPower = power;
      bestBpm = bpm;
    }
  }

  const sorted = [...powers].sort((a, b) => a - b);
  const medianPower = sorted[Math.floor(sorted.length / 2)] || 0;
  const bestIndex = powers.indexOf(bestPower);
  let refinedBpm = bestBpm;

  if (bestIndex > 0 && bestIndex < powers.length - 1) {
    const left = powers[bestIndex - 1];
    const center = powers[bestIndex];
    const right = powers[bestIndex + 1];
    const denominator = left - 2 * center + right;
    if (Math.abs(denominator) > 1e-9) {
      const offset = 0.5 * (left - right) / denominator;
      refinedBpm = bestBpm + offset * bpmStep;
    }
  }

  return { bestPower, bpm: clamp(refinedBpm, MIN_BPM, MAX_BPM), medianPower };
}

function drawOverlay(timestamp) {
  clearCanvas(overlayCtx, el.overlayCanvas);
  overlayCtx.lineWidth = Math.max(2, el.overlayCanvas.width / 480);
  overlayCtx.font = `${Math.max(15, el.overlayCanvas.width / 78)}px sans-serif`;
  overlayCtx.textBaseline = "middle";

  for (const track of tracks.values()) {
    if (timestamp - track.lastSeen > TRACK_TTL_SECONDS) continue;
    const { x, y, width, height } = track.box;
    const displayBpm = Number.isFinite(track.bpm) ? Math.round(track.bpm) : null;
    const label = displayBpm ? `#${track.id} ${displayBpm} BPM` : `#${track.id} ${track.status}`;

    overlayCtx.strokeStyle = track.color;
    overlayCtx.fillStyle = withAlpha(track.color, 0.13);
    roundedRect(overlayCtx, x, y, width, height, 8);
    overlayCtx.fill();
    overlayCtx.stroke();

    for (const rect of track.rects ?? []) {
      overlayCtx.fillStyle = withAlpha(track.color, 0.22);
      roundedRect(overlayCtx, rect.x, rect.y, rect.width, rect.height, 5);
      overlayCtx.fill();
    }

    const labelWidth = overlayCtx.measureText(label).width + 16;
    const labelHeight = 26;
    const labelX = clamp(x, 4, el.overlayCanvas.width - labelWidth - 4);
    const labelY = clamp(y - labelHeight - 6, 4, el.overlayCanvas.height - labelHeight - 4);
    overlayCtx.fillStyle = track.color;
    roundedRect(overlayCtx, labelX, labelY, labelWidth, labelHeight, 6);
    overlayCtx.fill();
    overlayCtx.fillStyle = "#ffffff";
    overlayCtx.fillText(label, labelX + 8, labelY + labelHeight / 2);
  }
}

function renderPeople() {
  const active = [...tracks.values()].sort((a, b) => a.id - b.id);
  el.activeTracksText.textContent = `${active.length} active`;

  if (!active.length) {
    el.peopleList.innerHTML = '<div class="empty-state">未检测到人脸</div>';
    return;
  }

  el.peopleList.innerHTML = active
    .map((track) => {
      const statusClass =
        track.status === "稳定" ? "" : track.status === "可参考" ? "weak" : "waiting";
      const bpm = Number.isFinite(track.bpm) && track.confidence > 0.18 ? Math.round(track.bpm) : "--";
      const confidence = Math.round(track.confidence * 100);
      const duration = track.samples.length
        ? Math.round(track.samples[track.samples.length - 1].t - track.samples[0].t)
        : 0;
      const sampleRate = track.sampleRate ? track.sampleRate.toFixed(1) : "0.0";

      return `
        <article class="person-card" style="--person-color: ${track.color}">
          <div class="person-head">
            <span class="person-name">人员 ${track.id}</span>
            <span class="person-state ${statusClass}">${track.status}</span>
          </div>
          <div class="bpm-row">
            <span class="bpm-value">${bpm}</span>
            <span class="bpm-unit">BPM</span>
          </div>
          ${ecgChart(track)}
          <div class="detail-grid">
            <span>置信 ${confidence}%</span>
            <span>${duration}s</span>
            <span>${sampleRate} Hz</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function ecgChart(track) {
  const values = track.pulseWave?.length ? track.pulseWave : [];
  const width = 260;
  const height = 58;
  const baseline = Math.round(height * 0.62);
  const wave = values;

  if (wave.length < 2) {
    return `<svg class="ecg-chart" viewBox="0 0 ${width} ${height}" aria-hidden="true" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="#04130f" />
      ${ecgGrid(width, height)}
      <line x1="0" y1="${baseline}" x2="${width}" y2="${baseline}" stroke="rgba(95, 255, 170, 0.12)" stroke-width="1.5" />
      <line x1="0" y1="${baseline}" x2="${width}" y2="${baseline}" stroke="#5fffa9" stroke-width="2" stroke-linecap="round" opacity="0.72" />
    </svg>`;
  }

  const points = wave
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = baseline - value * (height * 0.34);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return `<svg class="ecg-chart" viewBox="0 0 ${width} ${height}" aria-hidden="true" preserveAspectRatio="none">
    <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="#04130f" />
    ${ecgGrid(width, height)}
    <path d="M0 ${baseline} H ${width}" stroke="rgba(95, 255, 170, 0.12)" stroke-width="1.5" fill="none" />
    <polyline points="${points}" class="ecg-wave" fill="none" stroke="#5fffa9" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round" />
  </svg>`;
}

function ecgGrid(width, height) {
  const minor = 13;
  const major = 52;
  const lines = [];

  for (let x = 0; x <= width; x += minor) {
    const majorLine = x % major === 0;
    lines.push(
      `<line x1="${x}" y1="0" x2="${x}" y2="${height}" stroke="${majorLine ? "rgba(95, 255, 170, 0.16)" : "rgba(95, 255, 170, 0.07)"}" stroke-width="${majorLine ? 1.2 : 0.8}" />`,
    );
  }

  for (let y = 0; y <= height; y += minor) {
    const majorLine = y % major === 0;
    lines.push(
      `<line x1="0" y1="${y}" x2="${width}" y2="${y}" stroke="${majorLine ? "rgba(95, 255, 170, 0.16)" : "rgba(95, 255, 170, 0.07)"}" stroke-width="${majorLine ? 1.2 : 0.8}" />`,
    );
  }

  return `<g>${lines.join("")}</g>`;
}

function updateStats(nowMs) {
  const active = [...tracks.values()];
  const activeCount = active.length;
  const avgConfidence = activeCount
    ? active.reduce((sum, track) => sum + track.confidence, 0) / activeCount
    : 0;
  const avgSampleRate = activeCount
    ? active.reduce((sum, track) => sum + track.sampleRate, 0) / activeCount
    : 0;

  el.fpsText.textContent = `${fpsMeter.fps} fps`;
  el.faceCountText.textContent = `${activeCount} 人`;
  el.sampleRateText.textContent = avgSampleRate ? avgSampleRate.toFixed(1) : "0";
  el.qualityText.textContent = activeCount ? `${Math.round(avgConfidence * 100)}%` : "--";

  if (!appStartMs || !nowMs) {
    el.runtimeText.textContent = "00:00";
    return;
  }
  el.runtimeText.textContent = formatRuntime((nowMs - appStartMs) / 1000);
}

function resetFpsMeter(nowMs = performance.now()) {
  return { frames: 0, lastMs: nowMs, fps: 0 };
}

function tickFps(nowMs) {
  fpsMeter.frames += 1;
  if (nowMs - fpsMeter.lastMs < 1000) return;

  fpsMeter.fps = Math.round((fpsMeter.frames * 1000) / (nowMs - fpsMeter.lastMs));
  fpsMeter.frames = 0;
  fpsMeter.lastMs = nowMs;
}

function setStatus(text, state) {
  const dot = el.modelStatus.querySelector(".status-dot");
  const label = el.modelStatus.querySelector("span:last-child");
  dot.className = `status-dot ${state}`;
  label.textContent = text;
}

function setButtons(isBusy) {
  el.startButton.disabled = isBusy || running;
  el.stopButton.disabled = isBusy || !running;
}

function getMaxFaces() {
  return Number(el.maxFacesSelect.value);
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") return "摄像头权限被拒绝";
  if (error?.name === "NotFoundError") return "未找到摄像头";
  if (error?.name === "NotReadableError") return "摄像头被占用";
  if (location.protocol !== "https:" && location.hostname !== "localhost") {
    return "请使用 localhost 或 HTTPS 打开";
  }
  return "启动失败";
}

function formatRuntime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60)
    .toString()
    .padStart(2, "0");
  const secs = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${secs}`;
}

function pointDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function percentile(sortedValues, ratio) {
  if (!sortedValues.length) return 0;
  const index = clamp(
    Math.round((sortedValues.length - 1) * ratio),
    0,
    sortedValues.length - 1,
  );
  return sortedValues[index];
}

function clampBox(box) {
  const width = el.sampleCanvas.width;
  const height = el.sampleCanvas.height;
  const x = clamp(box.x, 0, width - 1);
  const y = clamp(box.y, 0, height - 1);
  const right = clamp(box.x + box.width, 1, width);
  const bottom = clamp(box.y + box.height, 1, height);
  return {
    x,
    y,
    width: Math.max(1, right - x),
    height: Math.max(1, bottom - y),
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clearCanvas(context, canvas) {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function withAlpha(hex, alpha) {
  const normalized = hex.replace("#", "");
  const bigint = Number.parseInt(normalized, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function standardDeviation(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(1, values.length - 1);
  return Math.sqrt(variance);
}

function normalize(values) {
  if (!values.length) return [];
  const maxAbs = Math.max(...values.map((value) => Math.abs(value))) || 1;
  return values.map((value) => clamp(value / maxAbs, -1, 1));
}

function ecgShape(values) {
  if (values.length < 2) return values.slice();

  const smoothed = values.map((value, index) => {
    const prev = values[index - 1] ?? value;
    const next = values[index + 1] ?? value;
    return value * 0.62 + prev * 0.2 + next * 0.18;
  });

  return smoothed.map((value, index) => {
    const prev = smoothed[index - 1] ?? value;
    const next = smoothed[index + 1] ?? value;
    const rise = Math.max(0, value - prev);
    const fall = Math.max(0, next - value);
    return clamp(value * 0.72 + rise * 0.88 - fall * 0.26, -1, 1);
  });
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return entities[char];
  });
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}
