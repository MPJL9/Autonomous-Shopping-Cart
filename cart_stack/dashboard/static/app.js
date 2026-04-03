const DEFAULT_TARGET = "10.65.142.17";
const STORAGE_KEY = "veda-cart-target";

const state = {
  status: null,
  remoteCameraKey: null,
  joystickActive: false,
  lastDriveSentAt: 0,
  activeMode: "live",
  cameraHeightPx: 360,
  resizingCamera: false,
  autoConnectAttempted: false,
  autoReconnectEnabled: true,
  lastReconnectAttemptAt: 0,
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
  deviceLeftPin: document.getElementById("deviceLeftPin"),
  deviceRightPin: document.getElementById("deviceRightPin"),
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
  leftPinInput: document.getElementById("leftPinInput"),
  rightPinInput: document.getElementById("rightPinInput"),
  trimForm: document.getElementById("trimForm"),
  leftTrimInput: document.getElementById("leftTrimInput"),
  rightTrimInput: document.getElementById("rightTrimInput"),
  deadbandInput: document.getElementById("deadbandInput"),
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
  if (document.activeElement !== input) {
    input.value = value;
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
  els.deviceLeftPin.textContent = String(robot.left_motor_pin);
  els.deviceRightPin.textContent = String(robot.right_motor_pin);
  syncInputValue(els.leftPinInput, String(robot.left_motor_pin));
  syncInputValue(els.rightPinInput, String(robot.right_motor_pin));
  syncInputValue(els.leftTrimInput, robot.left_trim.toFixed(2));
  syncInputValue(els.rightTrimInput, robot.right_trim.toFixed(2));
  syncInputValue(els.deadbandInput, robot.stop_deadband.toFixed(2));
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

async function sendDrive(left, right) {
  const now = performance.now();
  els.joystickReadout.textContent = `L ${left.toFixed(2)} / R ${right.toFixed(2)}`;
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
  els.joystickReadout.textContent = "L 0.00 / R 0.00";
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
    const leftPin = Number(els.leftPinInput.value);
    const rightPin = Number(els.rightPinInput.value);
    if (!Number.isInteger(leftPin) || !Number.isInteger(rightPin)) {
      showNotice("Enter whole BCM GPIO pin numbers for the left and right motor channels.");
      return;
    }
    await sendCommand(`pins ${leftPin} ${rightPin}`).catch((error) => showNotice(friendlyErrorMessage(error)));
  });

  els.trimForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const leftTrim = Number(els.leftTrimInput.value);
    const rightTrim = Number(els.rightTrimInput.value);
    const deadband = Number(els.deadbandInput.value);
    if ([leftTrim, rightTrim, deadband].some((value) => Number.isNaN(value))) {
      showNotice("Enter valid trim values before applying servo calibration.");
      return;
    }
    await sendCommand(`trim ${leftTrim} ${rightTrim} ${deadband}`).catch((error) =>
      showNotice(friendlyErrorMessage(error)),
    );
  });
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

function setupJoystick() {
  const moveHandle = (x, y) => {
    els.joystickHandle.style.transform = `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`;
  };

  const update = async (clientX, clientY) => {
    const rect = els.joystick.getBoundingClientRect();
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
    if (!state.joystickActive) {
      return;
    }
    state.joystickActive = false;
    moveHandle(0, 0);
    await sendStop().catch(() => {});
  };

  els.joystick.addEventListener("pointerdown", async (event) => {
    state.joystickActive = true;
    els.joystick.setPointerCapture(event.pointerId);
    await update(event.clientX, event.clientY);
  });

  els.joystick.addEventListener("pointermove", async (event) => {
    if (!state.joystickActive) {
      return;
    }
    await update(event.clientX, event.clientY);
  });

  els.joystick.addEventListener("pointerup", end);
  els.joystick.addEventListener("pointercancel", end);
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
  setupModeTabs();
  setupCameraResize();
  setupConnection();
  setupPinControls();
  setupLiveControls();
  setupQuickButtons();
  setupTerminal();
  setupJoystick();
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
