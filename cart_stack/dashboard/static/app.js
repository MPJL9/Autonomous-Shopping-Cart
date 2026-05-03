const DEFAULT_TARGET = "10.65.142.17";
const STORAGE_KEY = "veda-cart-target";

const state = {
  status: null,
  remoteCameraKey: null,
  joystickActive: false,
  lastDriveSentAt: 0,
  activeMode: "live",
  activeView: "live",
  cameraHeightPx: 360,
  resizingCamera: false,
  autoConnectAttempted: false,
  autoReconnectEnabled: true,
  lastReconnectAttemptAt: 0,
  recording: false,
  recordingOwner: null, // "standalone" | "rl" | null
  bundledAutoInstallTried: false,
  mediaRecorder: null,
  recordedChunks: [],
  recordingCanvas: null,
  recordingInterval: null,
  recordingStartedAt: 0,
  recordingFrameCount: 0,
  recordingTimer: null,
  rlBundle: {
    active: false,
    sessionId: null,
    browserStartedAtMs: null,
    browserStartedAtIso: null,
    policy: null,
    maxSteps: 0,
    stepHz: 0,
    actionScale: 0,
    videoBlob: null,
    videoDurationSec: 0,
  },
};

const els = {
  clock: document.getElementById("clock"),
  connectForm: document.getElementById("connectForm"),
  disconnectButton: document.getElementById("disconnectButton"),
  refreshButton: document.getElementById("refreshButton"),
  targetInput: document.getElementById("targetInput"),
  noticeBanner: document.getElementById("noticeBanner"),
  connectionPill: document.getElementById("connectionPill"),
  connectionPillText: document.getElementById("connectionPillText"),
  deviceAgent: document.getElementById("deviceAgent"),
  deviceCamera: document.getElementById("deviceCamera"),
  deviceMode: document.getElementById("deviceMode"),
  deviceIp: document.getElementById("deviceIp"),
  deviceLeft1Pin: document.getElementById("deviceLeft1Pin"),
  deviceLeft2Pin: document.getElementById("deviceLeft2Pin"),
  deviceRight1Pin: document.getElementById("deviceRight1Pin"),
  deviceRight2Pin: document.getElementById("deviceRight2Pin"),
  sessionLastCommand: document.getElementById("sessionLastCommand"),
  sessionDistance: document.getElementById("sessionDistance"),
  sessionGap: document.getElementById("sessionGap"),
  sessionSpeed: document.getElementById("sessionSpeed"),
  sessionTurn: document.getElementById("sessionTurn"),
  sessionLatency: document.getElementById("sessionLatency"),
  cameraViewport: document.getElementById("cameraViewport"),
  cameraResizeHandle: document.getElementById("cameraResizeHandle"),
  remoteCamera: document.getElementById("remoteCamera"),
  reticle: document.getElementById("guideReticle"),
  overlayDistance: document.getElementById("overlayDistance"),
  overlayHeading: document.getElementById("overlayHeading"),
  overlayCommand: document.getElementById("overlayCommand"),
  pinForm: document.getElementById("pinForm"),
  left1PinInput: document.getElementById("left1PinInput"),
  left2PinInput: document.getElementById("left2PinInput"),
  right1PinInput: document.getElementById("right1PinInput"),
  right2PinInput: document.getElementById("right2PinInput"),
  invertForm: document.getElementById("invertForm"),
  left1InvertInput: document.getElementById("left1InvertInput"),
  left2InvertInput: document.getElementById("left2InvertInput"),
  right1InvertInput: document.getElementById("right1InvertInput"),
  right2InvertInput: document.getElementById("right2InvertInput"),
  trimForm: document.getElementById("trimForm"),
  left1TrimInput: document.getElementById("left1TrimInput"),
  left2TrimInput: document.getElementById("left2TrimInput"),
  right1TrimInput: document.getElementById("right1TrimInput"),
  right2TrimInput: document.getElementById("right2TrimInput"),
  deadbandInput: document.getElementById("deadbandInput"),
  viewTabs: [...document.querySelectorAll("[data-view-tab]")],
  viewPanels: [...document.querySelectorAll("[data-view-panel]")],
  recordStartButton: document.getElementById("recordStartButton"),
  recordStopButton: document.getElementById("recordStopButton"),
  recordBadge: document.getElementById("recordBadge"),
  recordDuration: document.getElementById("recordDuration"),
  recordFrames: document.getElementById("recordFrames"),
  recordStatusValue: document.getElementById("recordStatusValue"),
  rlForm: document.getElementById("rlForm"),
  rlPolicyInput: document.getElementById("rlPolicyInput"),
  rlMaxStepsInput: document.getElementById("rlMaxStepsInput"),
  rlStepHzInput: document.getElementById("rlStepHzInput"),
  rlActionScaleInput: document.getElementById("rlActionScaleInput"),
  rlStartButton: document.getElementById("rlStartButton"),
  rlStopButton: document.getElementById("rlStopButton"),
  rlDownloadLogButton: document.getElementById("rlDownloadLogButton"),
  rlUploadPolicyButton: document.getElementById("rlUploadPolicyButton"),
  rlPolicyFileInput: document.getElementById("rlPolicyFileInput"),
  rlInstallBundledButton: document.getElementById("rlInstallBundledButton"),
  rlPolicyCombinedInfo: document.getElementById("rlPolicyCombinedInfo"),
  rlPolicy51OnlyInfo: document.getElementById("rlPolicy51OnlyInfo"),
  rlPolicyPreFixInfo: document.getElementById("rlPolicyPreFixInfo"),
  rlPolicy4_29Info: document.getElementById("rlPolicy4_29Info"),
  rlPolicy4_29MirrorInfo: document.getElementById("rlPolicy4_29MirrorInfo"),
  rlPolicyAwrInfo: document.getElementById("rlPolicyAwrInfo"),
  rlBadge: document.getElementById("rlBadge"),
  rlStepValue: document.getElementById("rlStepValue"),
  rlRewardValue: document.getElementById("rlRewardValue"),
  rlPolicyValue: document.getElementById("rlPolicyValue"),
  leftMotorValue: document.getElementById("leftMotorValue"),
  rightMotorValue: document.getElementById("rightMotorValue"),
  fpsValue: document.getElementById("fpsValue"),
  followBadge: document.getElementById("followBadge"),
  manualBadge: document.getElementById("manualBadge"),
  targetGapDisplay: document.getElementById("targetGapDisplay"),
  gapForm: document.getElementById("gapForm"),
  gapInput: document.getElementById("gapInput"),
  engageFollowButton: document.getElementById("engageFollowButton"),
  holdPositionButton: document.getElementById("holdPositionButton"),
  captureFrameButton: document.getElementById("captureFrameButton"),
  recordButton: document.getElementById("recordButton"),
  measureDistanceButton: document.getElementById("measureDistanceButton"),
  draftDistanceValue: document.getElementById("draftDistanceValue"),
  draftHeadingValue: document.getElementById("draftHeadingValue"),
  draftDriveValue: document.getElementById("draftDriveValue"),
  distanceNote: document.getElementById("distanceNote"),
  snapshotPreview: document.getElementById("snapshotPreview"),
  snapshotPlaceholder: document.getElementById("snapshotPlaceholder"),
  snapshotStatus: document.getElementById("snapshotStatus"),
  modeTabs: [...document.querySelectorAll("[data-mode-tab]")],
  modePanels: [...document.querySelectorAll("[data-mode-panel]")],
  joystick: document.getElementById("joystick"),
  joystickHandle: document.getElementById("joystickHandle"),
  joystickReadout: document.getElementById("joystickReadout"),
  rlJoystick: document.getElementById("rlJoystick"),
  rlJoystickHandle: document.getElementById("rlJoystickHandle"),
  rlJoystickReadout: document.getElementById("rlJoystickReadout"),
  terminalLog: document.getElementById("terminalLog"),
  terminalForm: document.getElementById("terminalForm"),
  terminalInput: document.getElementById("terminalInput"),
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function savedTarget() {
  return window.localStorage.getItem(STORAGE_KEY) || DEFAULT_TARGET;
}

function saveTarget(target) {
  window.localStorage.setItem(STORAGE_KEY, target);
}

function showNotice(message) {
  els.noticeBanner.textContent = message;
  els.noticeBanner.hidden = false;
}

function clearNotice() {
  els.noticeBanner.hidden = true;
  els.noticeBanner.textContent = "";
}

function friendlyErrorMessage(error) {
  if (!error) {
    return "Something went wrong.";
  }
  if (String(error.message || error) === "Failed to fetch") {
    return "The local dashboard server was not reachable yet. It is usually still starting up. Wait about 15 to 20 seconds, then refresh the page.";
  }
  return error.message || String(error);
}

function updateClock() {
  els.clock.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function mergeLogs(status) {
  const lines = [...(status.dashboard_logs || []), ...(status.robot.logs || [])];
  const seen = new Set();
  return lines.filter((line) => {
    if (seen.has(line)) {
      return false;
    }
    seen.add(line);
    return true;
  });
}

function renderTerminal(status) {
  els.terminalLog.innerHTML = "";
  const items = mergeLogs(status).slice(0, 18);
  const lines = items.length ? items : ["No robot activity yet."];

  lines.forEach((line, index) => {
    const row = document.createElement("p");
    row.className = `terminal-line ${index < 4 ? "system" : ""}`.trim();
    row.textContent = line;
    els.terminalLog.appendChild(row);
  });
}

function setMode(mode) {
  state.activeMode = mode;
  els.modeTabs.forEach((button) => button.classList.toggle("a", button.dataset.modeTab === mode));
  els.modePanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.modePanel === mode));
}

function setCameraHeight(heightPx) {
  const clamped = Math.max(280, Math.min(720, Math.round(heightPx)));
  state.cameraHeightPx = clamped;
  els.cameraViewport.style.height = `${clamped}px`;
}

function syncInputValue(input, value) {
  if (!input) return;
  if (document.activeElement !== input) {
    input.value = value;
  }
}

function syncCheckbox(input, value) {
  if (!input) return;
  if (document.activeElement !== input) {
    input.checked = Boolean(value);
  }
}

function updateRlStatus(robot) {
  if (!els.rlStepValue) return;
  const running = Boolean(robot.rl_running);
  els.rlBadge.textContent = running ? "Running" : "Idle";
  els.rlStepValue.textContent = `${robot.rl_step || 0} / ${robot.rl_max_steps || 0}`;
  els.rlRewardValue.textContent = Number(robot.rl_total_reward || 0).toFixed(2);
  els.rlPolicyValue.textContent = robot.rl_policy || "idle";

  // Vision-mode readout, if the element exists in the DOM.
  const _visionReadout = document.getElementById("visionModeReadout");
  if (_visionReadout) {
    _visionReadout.textContent = `mode: ${robot.vision_mode || "aruco"}`;
  }
  els.rlStartButton.disabled = running;
  els.rlStopButton.disabled = !running;
  // Pi auto-ended the session (hit max_steps, or errored) while we were recording — flush the video.
  if (state.rlBundle.active && !running) {
    finalizeRlRecording();
  }
}

function updateGuideOverlay() {
  const connected = Boolean(state.status?.connection?.connected);
  const visionEnabled = Boolean(state.status?.robot?.vision_enabled);
  const visionLocked = Boolean(state.status?.robot?.vision_locked);
  const followMode = state.status?.robot?.controller_mode === "auto";
  const showGuide = connected && visionEnabled && !visionLocked;
  els.reticle.hidden = !showGuide;
  els.reticle.classList.toggle("measure-mode", showGuide && !followMode);
}

function updateDistanceDraftView() {
  const robot = state.status?.robot;
  const visionEnabled = Boolean(robot?.vision_enabled);
  const visionLocked = Boolean(robot?.vision_locked);
  const followMode = robot?.controller_mode === "auto";

  els.draftDistanceValue.textContent = visionEnabled && visionLocked ? `${robot.distance_m.toFixed(2)} m` : "0.00 m";
  els.draftHeadingValue.textContent = visionEnabled && visionLocked ? `${robot.heading_rad.toFixed(2)} rad` : "0.00 rad";

  if (visionEnabled && followMode && visionLocked) {
    els.draftDriveValue.textContent = "follow";
  } else if (visionEnabled && visionLocked) {
    els.draftDriveValue.textContent = "ready";
  } else if (visionEnabled) {
    els.draftDriveValue.textContent = "search";
  } else {
    els.draftDriveValue.textContent = "stop";
  }

  if (!visionEnabled) {
    els.distanceNote.textContent =
      "Press Calibrate Distance to let the Pi look for a person and build a live box directly from the camera feed.";
  } else if (!visionLocked) {
    els.distanceNote.textContent =
      "Calibration is armed. Step into frame so the Pi can lock onto a person and estimate distance from the moving box.";
  } else if (followMode) {
    els.distanceNote.textContent =
      "Person lock is active. Follow mode is using the live box to hold the target gap and heading.";
  } else {
    els.distanceNote.textContent =
      "Person lock is active. Use Follow Person when you want the cart to start driving from the live box.";
  }

  els.measureDistanceButton.textContent = visionEnabled ? "Clear Calibration" : "Calibrate Distance";
  updateGuideOverlay();
}

function showOfflineCamera() {
  els.remoteCamera.hidden = true;
  els.remoteCamera.removeAttribute("src");
}

function syncCamera(status) {
  if (!status.connection.connected || status.connection.camera_mode !== "remote") {
    state.remoteCameraKey = null;
    showOfflineCamera();
    return;
  }

  els.remoteCamera.hidden = false;
  const sourceUrl = status.connection.camera_stream_url;
  const sourceKey = `${status.connection.target}|${sourceUrl}`;
  if (state.remoteCameraKey !== sourceKey) {
    state.remoteCameraKey = sourceKey;
    els.remoteCamera.src = `${sourceUrl}?t=${Date.now()}`;
  }
}

function renderStatus(status) {
  state.status = status;
  const { connection, robot } = status;
  const connected = connection.connected;
  const targetLabel =
    connected && String(connection.target).startsWith("http")
      ? new URL(connection.target).hostname
      : connected
        ? connection.target
        : savedTarget();
  const visionEnabled = Boolean(robot.vision_enabled);
  const visionLocked = Boolean(robot.vision_locked);

  els.connectionPill.className = `conn-pill ${connected ? "ok" : "err"}`;
  els.connectionPillText.textContent = connected ? "Pi Online" : "Pi Offline";
  els.deviceAgent.innerHTML = connected ? '<span class="status-good">●</span> Connected' : '<span class="status-bad">●</span> Offline';
  if (!connected) {
    els.deviceCamera.innerHTML = '<span class="status-bad">●</span> Waiting';
  } else if (visionEnabled && visionLocked) {
    els.deviceCamera.innerHTML = '<span class="status-good">●</span> Tracking';
  } else if (visionEnabled) {
    els.deviceCamera.innerHTML = '<span class="status-bad">●</span> Searching';
  } else {
    els.deviceCamera.innerHTML = '<span class="status-good">●</span> Live';
  }
  if (!connected) {
    els.deviceMode.textContent = "Offline";
  } else if (robot.controller_mode === "auto") {
    els.deviceMode.textContent = "Follow Live";
  } else if (visionEnabled) {
    els.deviceMode.textContent = "Calibrating";
  } else {
    els.deviceMode.textContent = "Manual";
  }
  els.deviceIp.textContent = targetLabel;
  els.deviceLeft1Pin.textContent = String(robot.left1_motor_pin);
  els.deviceLeft2Pin.textContent = String(robot.left2_motor_pin);
  els.deviceRight1Pin.textContent = String(robot.right1_motor_pin);
  els.deviceRight2Pin.textContent = String(robot.right2_motor_pin);
  syncInputValue(els.left1PinInput, String(robot.left1_motor_pin));
  syncInputValue(els.left2PinInput, String(robot.left2_motor_pin));
  syncInputValue(els.right1PinInput, String(robot.right1_motor_pin));
  syncInputValue(els.right2PinInput, String(robot.right2_motor_pin));
  syncCheckbox(els.left1InvertInput, Boolean(robot.left1_invert));
  syncCheckbox(els.left2InvertInput, Boolean(robot.left2_invert));
  syncCheckbox(els.right1InvertInput, Boolean(robot.right1_invert));
  syncCheckbox(els.right2InvertInput, Boolean(robot.right2_invert));
  syncInputValue(els.left1TrimInput, Number(robot.left1_trim).toFixed(2));
  syncInputValue(els.left2TrimInput, Number(robot.left2_trim).toFixed(2));
  syncInputValue(els.right1TrimInput, Number(robot.right1_trim).toFixed(2));
  syncInputValue(els.right2TrimInput, Number(robot.right2_trim).toFixed(2));
  syncInputValue(els.deadbandInput, Number(robot.stop_deadband).toFixed(2));
  updateRlStatus(robot);
  els.sessionLastCommand.textContent = robot.last_command;
  els.sessionDistance.textContent = `${robot.distance_m.toFixed(2)} m`;
  els.sessionGap.textContent = `${robot.target_gap_m.toFixed(2)} m`;
  els.sessionSpeed.textContent = `${robot.linear_velocity_mps.toFixed(2)} m/s`;
  els.sessionTurn.textContent = `${robot.angular_velocity_rad_s.toFixed(2)} rad/s`;
  els.sessionLatency.textContent = `${robot.command_latency_ms.toFixed(0)} ms`;
  els.overlayDistance.textContent = `${robot.distance_m.toFixed(2)} m`;
  els.overlayHeading.textContent = `${robot.heading_rad.toFixed(2)} rad`;
  els.overlayCommand.textContent = visionEnabled && visionLocked ? (robot.controller_mode === "auto" ? "follow" : "lock") : robot.last_command;
  els.leftMotorValue.textContent = robot.left_command.toFixed(2);
  els.rightMotorValue.textContent = robot.right_command.toFixed(2);
  els.fpsValue.textContent = robot.fps.toFixed(1);
  els.targetGapDisplay.textContent = `${robot.target_gap_m.toFixed(2)} m`;
  els.gapInput.value = robot.target_gap_m.toFixed(2);
  if (!connected) {
    els.followBadge.textContent = "Offline";
  } else if (robot.controller_mode === "auto" && visionLocked) {
    els.followBadge.textContent = "Following";
  } else if (robot.controller_mode === "auto") {
    els.followBadge.textContent = "Searching";
  } else if (visionLocked) {
    els.followBadge.textContent = "Locked";
  } else {
    els.followBadge.textContent = "Ready";
  }
  els.manualBadge.textContent = connected ? "Connected" : "Offline";
  if (connected) {
    clearNotice();
  }
  renderTerminal(status);
  syncCamera(status);
  updateDistanceDraftView();
}

async function refreshStatus() {
  renderStatus(await request("/api/status"));
}

async function connect(target) {
  const trimmed = target.trim() || DEFAULT_TARGET;
  state.remoteCameraKey = null;
  const payload = await request("/api/connection/connect", {
    method: "POST",
    body: JSON.stringify({ target: trimmed }),
  });
  state.autoReconnectEnabled = true;
  saveTarget(trimmed);
  renderStatus(payload.status);
  // Auto-install bundled policies once the Pi is reachable, but only if
  // neither slot is filled already — don't trample a user-uploaded checkpoint.
  autoInstallBundledIfEmpty().catch(() => {});
}

async function autoInstallBundledIfEmpty() {
  if (state.bundledAutoInstallTried) return;
  state.bundledAutoInstallTried = true;
  try {
    const info = await request("/api/rl/policy/info");
    const allFilled = [
      "bc_2d_combined_v2", "bc_2d_5_1only_v2",
      "bc_2d_combined_pre_fix", "bc_2d", "bc_2d_mirror",
      "bc_2d_awr",
    ].every((s) => info?.[s]?.installed);
    if (allFilled) {
      // All slots already filled (probably persisted from a previous session
      // on the Pi); leave them alone.
      return;
    }
    await request("/api/rl/policy/install-builtin?name=all", { method: "POST" });
    await refreshRlPolicyInfo();
  } catch (_) {
    // Auto-install is best-effort; surface failures only via the manual button.
  }
}

async function disconnect() {
  state.autoReconnectEnabled = false;
  state.remoteCameraKey = null;
  const payload = await request("/api/connection/disconnect", { method: "POST" });
  renderStatus(payload.status);
  showOfflineCamera();
}

async function sendCommand(command) {
  const payload = await request("/api/command", {
    method: "POST",
    body: JSON.stringify({ command }),
  });
  renderStatus(payload.status);
}

function _joystickReadouts() {
  return [els.joystickReadout, els.rlJoystickReadout].filter(Boolean);
}

async function sendDrive(left, right) {
  const now = performance.now();
  const text = `L ${left.toFixed(2)} / R ${right.toFixed(2)}`;
  _joystickReadouts().forEach((el) => (el.textContent = text));
  if (state.joystickActive && now - state.lastDriveSentAt < 100) {
    return;
  }
  state.lastDriveSentAt = now;
  renderStatus(
    await request("/api/drive", {
      method: "POST",
      body: JSON.stringify({ left, right }),
    }),
  );
}

async function sendStop() {
  renderStatus(await request("/api/stop", { method: "POST" }));
  _joystickReadouts().forEach((el) => (el.textContent = "L 0.00 / R 0.00"));
}

function captureCurrentFrame() {
  if (els.remoteCamera.hidden || !els.remoteCamera.complete || !els.remoteCamera.naturalWidth) {
    window.alert("The live camera frame is not ready yet.");
    return;
  }

  const canvas = document.createElement("canvas");
  canvas.width = els.remoteCamera.naturalWidth;
  canvas.height = els.remoteCamera.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(els.remoteCamera, 0, 0);
  const dataUrl = canvas.toDataURL("image/png");
  els.snapshotPreview.src = dataUrl;
  els.snapshotPreview.hidden = false;
  els.snapshotPlaceholder.hidden = true;
  els.snapshotStatus.textContent = `Frame captured at ${new Date().toLocaleTimeString()}.`;
}

function setupModeTabs() {
  els.modeTabs.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.modeTab));
  });
}

function setupCameraResize() {
  let startY = 0;
  let startHeight = state.cameraHeightPx;

  const stop = () => {
    state.resizingCamera = false;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", stop);
    window.removeEventListener("pointercancel", stop);
  };

  const onMove = (event) => {
    if (!state.resizingCamera) {
      return;
    }
    const deltaY = event.clientY - startY;
    setCameraHeight(startHeight + deltaY);
  };

  els.cameraResizeHandle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    state.resizingCamera = true;
    startY = event.clientY;
    startHeight = els.cameraViewport.getBoundingClientRect().height || state.cameraHeightPx;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  });

  els.cameraResizeHandle.addEventListener("dblclick", () => {
    setCameraHeight(360);
  });

  setCameraHeight(360);
}

function setupConnection() {
  els.targetInput.value = savedTarget();

  els.connectForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await connect(els.targetInput.value || DEFAULT_TARGET).catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.disconnectButton.addEventListener("click", async () => {
    await disconnect().catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.refreshButton.addEventListener("click", async () => {
    await refreshStatus().catch((error) => showNotice(friendlyErrorMessage(error)));
  });
}

function setupPinControls() {
  els.pinForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const pins = [
      Number(els.left1PinInput.value),
      Number(els.left2PinInput.value),
      Number(els.right1PinInput.value),
      Number(els.right2PinInput.value),
    ];
    if (pins.some((v) => !Number.isInteger(v) || v < 0 || v > 27)) {
      showNotice("Enter whole BCM GPIO pin numbers (0-27) for all four wheels.");
      return;
    }
    await sendCommand(`pins ${pins.join(" ")}`).catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.invertForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      left1_invert: els.left1InvertInput.checked,
      left2_invert: els.left2InvertInput.checked,
      right1_invert: els.right1InvertInput.checked,
      right2_invert: els.right2InvertInput.checked,
    };
    try {
      const result = await request("/api/motor-inverts", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderStatus(result);
    } catch (error) {
      showNotice(friendlyErrorMessage(error));
    }
  });

  els.trimForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const trims = [
      Number(els.left1TrimInput.value),
      Number(els.left2TrimInput.value),
      Number(els.right1TrimInput.value),
      Number(els.right2TrimInput.value),
    ];
    const deadband = Number(els.deadbandInput.value);
    if ([...trims, deadband].some((v) => Number.isNaN(v))) {
      showNotice("Enter valid trim values before applying servo calibration.");
      return;
    }
    await sendCommand(`trim ${trims.join(" ")} ${deadband}`).catch((error) =>
      showNotice(friendlyErrorMessage(error)),
    );
  });
}

function setupViewTabs() {
  const setView = (name) => {
    state.activeView = name;
    els.viewTabs.forEach((btn) => btn.classList.toggle("active", btn.dataset.viewTab === name));
    els.viewPanels.forEach((panel) => panel.classList.toggle("active", panel.dataset.viewPanel === name));
  };
  els.viewTabs.forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.viewTab));
  });
  setView("live");
}

function setupRecordControls() {
  if (!els.recordStartButton) return;
  els.recordStartButton.addEventListener("click", () => startStandaloneRecording());
  els.recordStopButton.addEventListener("click", () => stopStandaloneRecording());
}

function beginCanvasRecording(owner, onComplete) {
  if (state.recording) {
    showNotice(
      state.recordingOwner === "rl"
        ? "An RL session recording is in progress. Stop it first."
        : "A recording is already in progress.",
    );
    return false;
  }
  if (els.remoteCamera.hidden || !els.remoteCamera.complete || !els.remoteCamera.naturalWidth) {
    showNotice("The live camera frame is not ready yet. Connect to the Pi first.");
    return false;
  }
  const canvas = document.createElement("canvas");
  canvas.width = els.remoteCamera.naturalWidth;
  canvas.height = els.remoteCamera.naturalHeight;
  const ctx = canvas.getContext("2d");
  state.recordingCanvas = canvas;
  state.recordedChunks = [];
  state.recordingFrameCount = 0;
  state.recordingStartedAt = performance.now();

  const stream = canvas.captureStream(15);
  let recorder;
  try {
    recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
  } catch (err) {
    showNotice("This browser does not support WebM recording. Try Chrome or Firefox.");
    return false;
  }
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) state.recordedChunks.push(event.data);
  };
  recorder.onstop = () => {
    const blob = new Blob(state.recordedChunks, { type: "video/webm" });
    const durationSec = (performance.now() - state.recordingStartedAt) / 1000;
    if (typeof onComplete === "function") {
      onComplete(blob, durationSec);
    }
  };

  state.recordingInterval = window.setInterval(() => {
    if (els.remoteCamera.complete && els.remoteCamera.naturalWidth) {
      ctx.drawImage(els.remoteCamera, 0, 0, canvas.width, canvas.height);
      state.recordingFrameCount += 1;
      if (els.recordFrames) els.recordFrames.textContent = String(state.recordingFrameCount);
    }
  }, 66);

  state.recordingTimer = window.setInterval(() => {
    const seconds = Math.floor((performance.now() - state.recordingStartedAt) / 1000);
    const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
    const ss = String(seconds % 60).padStart(2, "0");
    if (els.recordDuration) els.recordDuration.textContent = `${mm}:${ss}`;
  }, 500);

  recorder.start(500);
  state.mediaRecorder = recorder;
  state.recording = true;
  state.recordingOwner = owner;
  return true;
}

function stopCanvasRecording() {
  if (!state.recording) return;
  if (state.recordingInterval) {
    window.clearInterval(state.recordingInterval);
    state.recordingInterval = null;
  }
  if (state.recordingTimer) {
    window.clearInterval(state.recordingTimer);
    state.recordingTimer = null;
  }
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  state.mediaRecorder = null;
  state.recordingCanvas = null;
  state.recording = false;
  state.recordingOwner = null;
}

function startStandaloneRecording() {
  const ok = beginCanvasRecording("standalone", (blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    a.href = url;
    a.download = `cart-recording-${timestamp}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  });
  if (!ok) return;
  els.recordBadge.textContent = "Recording";
  els.recordStatusValue.textContent = "recording";
  els.recordStartButton.disabled = true;
  els.recordStopButton.disabled = false;
}

function stopStandaloneRecording() {
  if (!state.recording || state.recordingOwner !== "standalone") return;
  stopCanvasRecording();
  els.recordBadge.textContent = "Idle";
  els.recordStatusValue.textContent = "saved";
  els.recordStartButton.disabled = false;
  els.recordStopButton.disabled = true;
}

function setupRlControls() {
  if (!els.rlForm) return;
  els.rlForm.addEventListener("submit", handleRlStart);
  els.rlStopButton.addEventListener("click", handleRlStop);
  els.rlDownloadLogButton.addEventListener("click", handleRlDownloadBundle);
  if (els.rlUploadPolicyButton && els.rlPolicyFileInput) {
    els.rlUploadPolicyButton.addEventListener("click", () => els.rlPolicyFileInput.click());
    els.rlPolicyFileInput.addEventListener("change", handleRlPolicyUpload);
  }
  if (els.rlInstallBundledButton) {
    els.rlInstallBundledButton.addEventListener("click", handleRlInstallBundled);
  }
  refreshRlPolicyInfo().catch(() => {});
}

async function handleRlInstallBundled() {
  const btn = els.rlInstallBundledButton;
  if (btn) { btn.disabled = true; btn.textContent = "Installing..."; }
  try {
    const result = await request("/api/rl/policy/install-builtin?name=all", {
      method: "POST",
    });
    const lines = (result.installed || []).map((r) =>
      r.ok
        ? `${r.file} → ${r.slot} (${(r.size_bytes / 1024).toFixed(1)} KB)`
        : `${r.file} FAILED: ${r.error}`,
    );
    showNotice(`Installed ${result.installed.length} policy file(s):\n${lines.join("\n")}`);
    await refreshRlPolicyInfo();
  } catch (error) {
    showNotice(friendlyErrorMessage(error));
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Install Bundled Policies (6 slots)"; }
  }
}

function _renderSlot(el, info) {
  if (!el) return;
  if (!info || !info.installed) {
    el.textContent = "not installed";
    el.style.color = "#888";
    return;
  }
  const kb = (info.size_bytes / 1024).toFixed(1);
  el.textContent = `${kb} KB · ${info.modified_at.slice(11, 19)}`;
  el.style.color = "";
}

async function refreshRlPolicyInfo() {
  const slotEls = [
    els.rlPolicyCombinedInfo, els.rlPolicy51OnlyInfo,
    els.rlPolicyPreFixInfo, els.rlPolicy4_29Info, els.rlPolicy4_29MirrorInfo,
    els.rlPolicyAwrInfo,
  ];
  if (slotEls.every((e) => !e)) return;
  try {
    const info = await request("/api/rl/policy/info");
    _renderSlot(els.rlPolicyCombinedInfo,   info.bc_2d_combined_v2);
    _renderSlot(els.rlPolicy51OnlyInfo,     info.bc_2d_5_1only_v2);
    _renderSlot(els.rlPolicyPreFixInfo,     info.bc_2d_combined_pre_fix);
    _renderSlot(els.rlPolicy4_29Info,       info.bc_2d);
    _renderSlot(els.rlPolicy4_29MirrorInfo, info.bc_2d_mirror);
    _renderSlot(els.rlPolicyAwrInfo,        info.bc_2d_awr);
  } catch (_) {
    slotEls.forEach((el) => {
      if (el) {
        el.textContent = "(connect Pi)";
        el.style.color = "#888";
      }
    });
  }
}

async function handleRlPolicyUpload(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    const buf = await file.arrayBuffer();
    const response = await fetch("/api/rl/policy/upload", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: buf,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `Upload failed (HTTP ${response.status})`);
    }
    const slot = payload.slot || "?";
    const kb = (payload.size_bytes / 1024).toFixed(1);
    showNotice(`Policy uploaded to ${slot} slot: ${kb} KB`);
    await refreshRlPolicyInfo();
  } catch (error) {
    showNotice(friendlyErrorMessage(error));
  } finally {
    event.target.value = "";  // allow re-uploading the same filename
  }
}

async function handleRlStart(event) {
  event.preventDefault();
  if (state.recording && state.recordingOwner !== "rl") {
    showNotice("A standalone recording is running. Stop it before starting an RL session.");
    return;
  }

  const policy = els.rlPolicyInput.value;
  const maxSteps = Number(els.rlMaxStepsInput.value);
  const stepHz = Number(els.rlStepHzInput.value);
  const actionScale = Number(els.rlActionScaleInput.value);

  const browserStartedAtMs = Date.now();
  const browserStartedAtIso = new Date(browserStartedAtMs).toISOString();
  const sessionId = browserStartedAtIso.replace(/[:.]/g, "-").slice(0, 19);

  const recStarted = beginCanvasRecording("rl", (blob, durationSec) => {
    // Only save if this completion belongs to the *current* session. On start
    // failure we null out sessionId, so the late onStop callback is ignored.
    if (state.rlBundle.sessionId !== sessionId) return;
    state.rlBundle.videoBlob = blob;
    state.rlBundle.videoDurationSec = durationSec;
  });
  if (!recStarted) {
    return; // beginCanvasRecording already surfaced the reason
  }

  state.rlBundle = {
    active: true,
    sessionId,
    browserStartedAtMs,
    browserStartedAtIso,
    policy,
    maxSteps,
    stepHz,
    actionScale,
    videoBlob: null,
    videoDurationSec: 0,
  };

  try {
    const result = await request("/api/rl/start", {
      method: "POST",
      body: JSON.stringify({
        policy,
        max_steps: maxSteps,
        step_hz: stepHz,
        action_scale: actionScale,
      }),
    });
    renderStatus(result);
  } catch (error) {
    // Pi rejected the start: orphan the sessionId so the recorder's onStop
    // callback won't write to rlBundle, then tear down.
    state.rlBundle.sessionId = null;
    state.rlBundle.active = false;
    stopCanvasRecording();
    showNotice(friendlyErrorMessage(error));
    return;
  }

  els.recordBadge.textContent = "Recording (RL)";
  els.recordStatusValue.textContent = "recording (rl)";
  els.recordStartButton.disabled = true;
  els.recordStopButton.disabled = true; // standalone stop must not stop an RL recording
}

async function handleRlStop() {
  try {
    const result = await request("/api/rl/stop", { method: "POST" });
    renderStatus(result);
  } catch (error) {
    showNotice(friendlyErrorMessage(error));
    // Keep the recording running; the user can retry stop.
    return;
  }
  finalizeRlRecording();
}

function finalizeRlRecording() {
  if (!state.rlBundle.active) return;
  state.rlBundle.active = false;
  if (state.recording && state.recordingOwner === "rl") {
    stopCanvasRecording();
  }
  els.recordBadge.textContent = "Idle";
  els.recordStatusValue.textContent = "saved";
  els.recordStartButton.disabled = false;
  els.recordStopButton.disabled = true;
}

async function handleRlDownloadBundle() {
  // 1. Fetch the JSONL from the Pi.
  let logBlob = null;
  let logFilename = null;
  try {
    const response = await fetch("/api/rl/log");
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Log download failed (HTTP ${response.status})`);
    }
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^";]+)"?/);
    logFilename = match ? match[1] : "rl_session.jsonl";
    logBlob = await response.blob();
  } catch (error) {
    showNotice(friendlyErrorMessage(error));
  }

  // Use the Pi's log filename as the stem so all three files share a name.
  const stem = (logFilename || `rl_session_${state.rlBundle.sessionId || Date.now()}`).replace(/\.jsonl$/, "");
  const videoFilename = `${stem}.webm`;
  const metaFilename = `${stem}.meta.json`;

  // 2. Build the meta sidecar.
  const meta = {
    session_id: state.rlBundle.sessionId,
    browser_started_at_ms: state.rlBundle.browserStartedAtMs,
    browser_started_at_iso: state.rlBundle.browserStartedAtIso,
    policy: state.rlBundle.policy,
    max_steps: state.rlBundle.maxSteps,
    step_hz: state.rlBundle.stepHz,
    action_scale: state.rlBundle.actionScale,
    video_filename: state.rlBundle.videoBlob ? videoFilename : null,
    video_duration_sec: state.rlBundle.videoDurationSec || null,
    log_filename: logFilename,
    alignment_notes:
      "To align video to JSONL: the video frame at t=0s corresponds to browser_started_at_iso. " +
      "The JSONL header 'started_at' is the Pi wall-clock at session start; use " +
      "(pi_started_at_ms - browser_started_at_ms) as the clock offset. " +
      "Round-trip latency (~20-100ms LAN) is the residual error.",
  };
  const metaBlob = new Blob([JSON.stringify(meta, null, 2)], { type: "application/json" });

  // 3. Sequential download (brief spacing so Chrome's multi-download prompt fires once).
  const downloads = [];
  if (logBlob) downloads.push([logBlob, logFilename]);
  if (state.rlBundle.videoBlob) downloads.push([state.rlBundle.videoBlob, videoFilename]);
  downloads.push([metaBlob, metaFilename]);

  for (const [blob, filename] of downloads) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    await new Promise((r) => setTimeout(r, 150));
  }

  if (!state.rlBundle.videoBlob) {
    showNotice("No RL video cached (page may have refreshed). Downloaded log + meta only.");
  }
}

function setupLiveControls() {
  els.engageFollowButton.addEventListener("click", async () => {
    await sendCommand("follow_person").catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.holdPositionButton.addEventListener("click", async () => {
    await sendCommand("manual").catch((error) => showNotice(friendlyErrorMessage(error)));
    await sendCommand("stop").catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.captureFrameButton.addEventListener("click", captureCurrentFrame);
  // The in-Live "Record" button just jumps to the Record tab; actual recording lives there.
  els.recordButton.addEventListener("click", () => {
    const recordTab = els.viewTabs.find((btn) => btn.dataset.viewTab === "record");
    if (recordTab) recordTab.click();
  });

  els.measureDistanceButton.addEventListener("click", async () => {
    const command = state.status?.robot?.vision_enabled ? "clear_calibration" : "calibrate_distance";
    await sendCommand(command).catch((error) => showNotice(friendlyErrorMessage(error)));
    setMode("manual");
  });

  els.gapForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const gap = Number(els.gapInput.value);
    if (Number.isNaN(gap)) {
      showNotice("Enter a valid follow distance.");
      return;
    }
    await sendCommand(`gap ${gap}`).catch((error) => showNotice(friendlyErrorMessage(error)));
  });
}

function setupQuickButtons() {
  document.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", async () => {
      await sendCommand(button.dataset.command).catch((error) => showNotice(friendlyErrorMessage(error)));
    });
  });
}

function setupTerminal() {
  els.terminalForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const command = els.terminalInput.value.trim();
    if (!command) {
      return;
    }
    els.terminalInput.value = "";
    await sendCommand(command).catch((error) => showNotice(friendlyErrorMessage(error)));
  });
}

// Keyboard teleop:
//   w  -> forward 0.4   (hold = drive, release = stop)
//   s  -> reverse 0.4
//   a  -> spin_left 0.15
//   d  -> spin_right 0.15
//   space -> stop  (one-shot, does not auto-undo on keyup)
//
// Routes through sendCommand exactly like the dashboard buttons, so motor
// state, recording, and command parsing all behave identically.
const _KEY_TO_COMMAND = {
  w: "forward 0.4",
  s: "reverse 0.4",
  a: "spin_left 0.15",
  d: "spin_right 0.15",
  " ": "stop",
};
const _AUTO_STOP_KEYS = new Set(["w", "s", "a", "d"]);

function _isTypingInInput() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
}

function setupKeyboardControls() {
  document.addEventListener("keydown", async (event) => {
    if (event.repeat) return;          // ignore OS auto-repeat
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (_isTypingInInput()) return;    // don't hijack the terminal / number inputs
    const key = event.key.toLowerCase();
    const cmd = _KEY_TO_COMMAND[key];
    if (!cmd) return;
    event.preventDefault();
    await sendCommand(cmd).catch((error) => showNotice(friendlyErrorMessage(error)));
  });
  document.addEventListener("keyup", async (event) => {
    if (_isTypingInInput()) return;
    const key = event.key.toLowerCase();
    if (!_AUTO_STOP_KEYS.has(key)) return;
    event.preventDefault();
    await sendCommand("stop").catch((error) => showNotice(friendlyErrorMessage(error)));
  });
}

function _attachJoystick(joystickEl, handleEl) {
  if (!joystickEl || !handleEl) return;

  const moveHandle = (x, y) => {
    handleEl.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
  };

  const update = async (clientX, clientY) => {
    const rect = joystickEl.getBoundingClientRect();
    const radius = rect.width / 2;
    const maxDistance = radius * 0.68;
    const centerX = rect.left + radius;
    const centerY = rect.top + radius;
    let dx = clientX - centerX;
    let dy = clientY - centerY;
    const distance = Math.hypot(dx, dy);
    if (distance > maxDistance) {
      const scale = maxDistance / distance;
      dx *= scale;
      dy *= scale;
    }

    moveHandle(dx, dy);
    const xNorm = dx / maxDistance;
    const yNorm = dy / maxDistance;
    const left = Math.max(-1, Math.min(1, -yNorm + xNorm));
    const right = Math.max(-1, Math.min(1, -yNorm - xNorm));
    await sendDrive(left, right).catch(() => {});
  };

  const end = async () => {
    if (!state.joystickActive) return;
    state.joystickActive = false;
    moveHandle(0, 0);
    await sendStop().catch(() => {});
  };

  joystickEl.addEventListener("pointerdown", async (event) => {
    state.joystickActive = true;
    joystickEl.setPointerCapture(event.pointerId);
    await update(event.clientX, event.clientY);
  });

  joystickEl.addEventListener("pointermove", async (event) => {
    if (!state.joystickActive) return;
    await update(event.clientX, event.clientY);
  });

  joystickEl.addEventListener("pointerup", end);
  joystickEl.addEventListener("pointercancel", end);
}

function setupJoystick() {
  _attachJoystick(els.joystick, els.joystickHandle);
  _attachJoystick(els.rlJoystick, els.rlJoystickHandle);
}

function attachCameraHandlers() {
  els.remoteCamera.addEventListener("error", () => {
    state.remoteCameraKey = null;
    showOfflineCamera();
    showNotice("Camera stream is retrying. If it stays blank, click Connect once to refresh the feed.");
  });
}

async function maybeAutoConnect() {
  if (state.autoConnectAttempted) {
    return;
  }
  state.autoConnectAttempted = true;
  const status = await request("/api/status");
  renderStatus(status);
  if (!status.connection.connected) {
    await connect(savedTarget());
  }
}

async function maybeReconnect() {
  if (!state.autoReconnectEnabled || !state.status || state.status.connection.connected) {
    return;
  }
  const now = performance.now();
  if (now - state.lastReconnectAttemptAt < 5000) {
    return;
  }
  state.lastReconnectAttemptAt = now;
  await connect(savedTarget());
}

async function bootstrap() {
  updateClock();
  window.setInterval(updateClock, 30000);
  setupViewTabs();
  setupModeTabs();
  setupCameraResize();
  setupConnection();
  setupPinControls();
  setupLiveControls();
  setupQuickButtons();
  setupTerminal();
  setupJoystick();
  setupKeyboardControls();
  setupRecordControls();
  setupRlControls();
  attachCameraHandlers();
  updateDistanceDraftView();
  setMode("live");
  await maybeAutoConnect().catch((error) => {
    showNotice(friendlyErrorMessage(error));
  });
  await refreshStatus().catch((error) => {
    showNotice(friendlyErrorMessage(error));
  });
  window.setInterval(async () => {
    await refreshStatus().catch(() => {});
    await maybeReconnect().catch(() => {});
  }, 1200);
}

bootstrap().catch((error) => showNotice(friendlyErrorMessage(error)));
