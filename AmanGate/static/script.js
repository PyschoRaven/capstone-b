const cameraBtn       = document.getElementById("cameraBtn");
const resultText      = document.getElementById("result-text");
const statusDiv       = document.getElementById("status");
const video           = document.getElementById("video");
const livePill        = document.getElementById("livePill");
const statusDot       = document.getElementById("statusDot");
const detailAccessType = document.getElementById("detailAccessType");
const detailPersonName = document.getElementById("detailPersonName");
const detailStatus    = document.getElementById("detailStatus");
const detailGateAction = document.getElementById("detailGateAction");
const cameraMessage   = document.getElementById("cameraMessage");
const progressText    = document.getElementById("progressText");
const progressFill    = document.getElementById("progressFill");
const accessCard      = document.querySelector(".access-card");
const cameraContainer = document.querySelector(".camera-container");

// --------------------------
// State
// --------------------------

let cameraStream        = null;
let cameraActive        = false;
let isProcessing        = false;
let isPaused            = false;
let lockdownActive      = false;
let processInterval     = null;
let lockdownInterval    = null;
let centerCheckInterval = null;
let isCheckingCenter    = false;
let centeredFrames      = 0;
let lastScanAt          = 0;

// --------------------------
// This terminal's gate
// --------------------------
// One server, one iPad per gate. The gate comes from the link (?gate=female)
// or the picker in the header, and is remembered on this device.
const GATE_NAMES = { main: "Main Gate", male: "Male Gate", female: "Female Gate", vehicle: "Vehicle Gate" };
const GATE_KEY   = "amangate.terminalGate";
const gateSelect = document.getElementById("gateSelect");

function pickGate() {
    const fromLink = new URLSearchParams(location.search).get("gate");
    let saved = null;
    try { saved = localStorage.getItem(GATE_KEY); } catch (e) { /* ignore */ }
    const gate = GATE_NAMES[fromLink] ? fromLink : GATE_NAMES[saved] ? saved : "main";
    try { localStorage.setItem(GATE_KEY, gate); } catch (e) { /* ignore */ }
    return gate;
}

const gateId    = pickGate();
let gateType    = "walk";     // "walk" | "vehicle" — set by the admin, read from the server
let gateLocked  = false;      // this gate closed by the admin (other gates may be open)
if (gateSelect) {
    gateSelect.value = gateId;
    gateSelect.addEventListener("change", () => {
        try { localStorage.setItem(GATE_KEY, gateSelect.value); } catch (e) { /* ignore */ }
        location.search = "?gate=" + gateSelect.value;   // reload as the chosen gate
    });
}


// --------------------------
// Constants
// --------------------------

const CENTER_CHECK_INTERVAL   = 500;
const PROCESS_INTERVAL        = 500;
const CENTERED_FRAMES_REQUIRED = 2;
const SCAN_COOLDOWN_TIME      = 2500;
const SUCCESS_HOLD_TIME       = 5000;
const LOCKDOWN_CHECK_INTERVAL = 5000; // reduced from 1000ms to avoid console spam

// --------------------------
// Blink detection (liveness)
// --------------------------
// Google MediaPipe Face Landmarker runs in the browser (on the iPad) and tracks
// the eyelids ~30 times a second. A scan is only allowed after one real blink
// (eyes open -> closed -> open), so a printed photo or still picture can't pass.
// Each scan needs a fresh blink, and the blink is forgotten if the face leaves.

const MEDIAPIPE_URL   = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const BLINK_CLOSED    = 0.5;    // eyeBlink score above this = eyes closed
const BLINK_OPEN      = 0.25;   // ...and below this = eyes open again
const BLINK_MAX_MS    = 800;    // a real blink closes the eyes for well under a second
const BLINK_VALID_MS  = 5000;   // a blink allows a scan within this time
// After the blink, wait until the eyes have been fully open for a moment before
// taking the photos — half-open eyes right after a blink lower ArcFace's match.
const EYES_OPEN_SCORE = 0.2;    // eyeBlink score below this = eyes fully open
const EYES_SETTLE_MS  = 500;    // eyes must stay open this long before the photos
// ~15 checks a second is plenty to catch a blink (100-300 ms). Running on every
// video frame used a lot of processor time — on a laptop that also runs the
// server it slowed the face scan itself down.
const BLINK_CHECK_MS  = 66;

let faceLandmarker  = null;
let blinkStatus     = "loading";  // "loading" | "ready" | "failed"
let lastBlinkAt     = 0;
let eyesClosedAt    = 0;
// Every scan is treated as a new person: the "Please blink" prompt is shown
// first, and only a blink AFTER the prompt counts (natural blinks during the
// previous result screen are ignored).
let blinkPromptAt   = 0;      // when "Please blink" was shown for the current person (0 = not yet)
let noFaceSince     = 0;      // when the face last disappeared from the camera
let eyesOpenSince   = 0;      // since when the eyes have been fully open (0 = not open now)
let lastVideoTime   = -1;
let lastBlinkCheck  = 0;

async function loadBlinkDetector() {
    try {
        const vision  = await import(`${MEDIAPIPE_URL}/vision_bundle.mjs`);
        const fileset = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE_URL}/wasm`);
        const options = delegate => ({
            baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true
        });
        try {
            faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options("GPU"));
        } catch (e) {
            faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options("CPU"));
        }
        blinkStatus = "ready";
        if (cameraActive) requestAnimationFrame(blinkLoop);
    } catch (error) {
        console.error("Blink detector failed to load", error);
        blinkStatus = "failed";
        if (cameraActive && !lockdownActive) showBlinkUnavailable();
    }
}

function blinkScore(result) {
    const shapes = result.faceBlendshapes?.[0]?.categories;
    if (!shapes) return null;
    const get = name => shapes.find(c => c.categoryName === name)?.score ?? 0;
    return (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2;
}

function blinkLoop() {
    if (!cameraActive || !faceLandmarker) return;
    // Pause while a scan is running or a result is on screen — no blink is needed
    // then, and the processor is better spent on the face scan.
    const busy = isProcessing || isPaused;
    if (!busy && performance.now() - lastBlinkCheck >= BLINK_CHECK_MS
            && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastBlinkCheck = performance.now();
        lastVideoTime = video.currentTime;
        const score = blinkScore(faceLandmarker.detectForVideo(video, performance.now()));
        const now = Date.now();
        if (score === null) {
            eyesOpenSince = 0;
            // Face gone for over a second: treat whoever comes next as a new person
            if (!noFaceSince) noFaceSince = now;
            else if (now - noFaceSince > 1000) resetBlink();
            requestAnimationFrame(blinkLoop);
            return;
        }
        noFaceSince = 0;
        if (score < EYES_OPEN_SCORE) { if (!eyesOpenSince) eyesOpenSince = now; }
        else eyesOpenSince = 0;
        if (!eyesClosedAt && score > BLINK_CLOSED) {
            eyesClosedAt = now;
        } else if (eyesClosedAt && score < BLINK_OPEN) {
            if (now - eyesClosedAt < BLINK_MAX_MS) lastBlinkAt = now;
            eyesClosedAt = 0;
        }
    }
    requestAnimationFrame(blinkLoop);
}

function hasRecentBlink() {
    return blinkStatus === "ready"
        && blinkPromptAt > 0
        && lastBlinkAt > blinkPromptAt
        && Date.now() - lastBlinkAt < BLINK_VALID_MS;
}

// Eyes fully open and settled after the blink — the right moment for the photos
function eyesReady() {
    const settled = eyesOpenSince > 0 && Date.now() - eyesOpenSince >= EYES_SETTLE_MS;
    // Some eyes never score as "fully open" — don't keep them waiting forever
    return settled || Date.now() - lastBlinkAt > 1500;
}

// Start a fresh liveness check (new person / new scan)
function resetBlink() {
    lastBlinkAt = eyesClosedAt = blinkPromptAt = 0;
}

function showBlinkUnavailable() {
    setTone("danger");
    setCameraMessage("Blink check unavailable", "danger");
    resultText.innerHTML   = "Liveness check could not load";
    resultText.style.color = "var(--danger)";
    statusDiv.innerHTML    = "Check the internet connection, then reload the page";
    statusDiv.style.color  = "var(--danger)";
}

// Face tracking overlay removed

// --------------------------
// UI helpers
// --------------------------

function setProgress(scan, percent) {
    const safe = Math.max(0, Math.min(100, percent));
    if (progressText) progressText.textContent = `Scanning ${scan}/3 - ${safe}%`;
    if (progressFill) progressFill.style.width  = `${safe}%`;
}

function setCameraMessage(message, tone = "") {
    // Audio guidance (audio-only.js) speaks each message when it is switched on
    window.AmanAudio?.prompt(message === "Please blink" ? "Please blink once to continue." : message);
    if (!cameraMessage) return;
    cameraMessage.textContent = message;
    cameraMessage.classList.remove("is-success", "is-danger", "is-warning");
    if (tone) cameraMessage.classList.add(`is-${tone}`);
}

function getPositionMessage(message = "") {
    const n = message.toLowerCase();
    if (n.includes("turn") && n.includes("right")) return "Turn head right";
    if (n.includes("turn") && n.includes("left"))  return "Turn head left";
    if (n.includes("right")) return "Move right";
    if (n.includes("left"))  return "Move left";
    if (n.includes("down"))  return "Move down";
    if (n.includes("up"))    return "Move up";
    if (n.includes("closer")) return "Move closer";
    if (n.includes("back"))   return "Move back";
    return "Center your face";
}

function setTone(tone = "") {
    if (accessCard) {
        accessCard.classList.remove("status-success", "status-danger", "status-warning");
        if (tone) accessCard.classList.add(`status-${tone}`);
    }
    [detailStatus, detailGateAction].forEach(node => {
        if (!node) return;
        node.classList.remove("status-success", "status-danger", "status-warning");
        if (tone) node.classList.add(`status-${tone}`);
    });
}

function setDetails({ accessType, personName, gateAction, status } = {}) {
    // Only update fields that are explicitly passed
    if (detailAccessType && accessType !== undefined) detailAccessType.textContent = accessType;
    if (detailPersonName  && personName  !== undefined) detailPersonName.textContent  = personName;
    if (detailStatus      && status      !== undefined) detailStatus.textContent      = status;
    if (detailGateAction  && gateAction  !== undefined) detailGateAction.textContent  = gateAction;
}

function setUiState(label, alert = false) {
    if (livePill) {
        livePill.textContent = label;
        livePill.classList.toggle("is-alert", alert);
    }
    if (statusDot) statusDot.classList.toggle("is-alert", alert);
}

function setScanAnimation(active) {
    cameraContainer?.classList.toggle("is-scanning", active);
}

// --------------------------
// Lockdown UI
// --------------------------

function showLockdownMessage() {
    window.AmanAudio?.prompt(gateLocked && !emergencyLockdown ? "This gate is closed. Please use another gate." : "Emergency lockdown is active. Please follow security instructions.", "gate-closed", true);
    // Either the emergency lockdown (all gates) or only this gate closed by the admin
    const onlyThisGate = gateLocked && !emergencyLockdown;
    setUiState(onlyThisGate ? "Closed" : "Lockdown", true);
    setTone("danger");
    setProgress(0, 0);
    setCameraMessage(onlyThisGate ? "Gate closed" : "Access denied", "danger");
    setDetails({
        accessType: onlyThisGate ? "Gate closed" : "Lockdown Active",
        personName: "System Restricted",
        gateAction: onlyThisGate ? "Use another gate" : "Contact Security",
        status:     onlyThisGate ? "Closed" : "Lockdown"
    });
    resultText.innerHTML   = onlyThisGate ? `${GATE_NAMES[gateId].toUpperCase()} CLOSED` : "EMERGENCY LOCKDOWN";
    resultText.style.color = "var(--danger)";
    statusDiv.innerHTML    = onlyThisGate
        ? "This gate is closed by the admin. Please use another gate."
        : "Gate is closed. Access is disabled by admin.";
    statusDiv.style.color  = "var(--danger)";
}

function showReadyMessage() {
    if (cameraActive) window.AmanAudio?.ready();
    else window.AmanAudio?.stop();
    setScanAnimation(cameraActive);
    setUiState(cameraActive ? "Scanning" : "Standby");
    setTone("");
    setProgress(cameraActive ? 1 : 0, cameraActive ? 20 : 0);
    setCameraMessage(cameraActive ? "Look straight" : "Center your face");
    setDetails({
        accessType: cameraActive ? "Scanning..."             : "Students, Staff & Faculty",
        personName: cameraActive ? "Detecting face"          : "Not verified yet",
        gateAction: "Waiting",
        status:     cameraActive ? "Scanning"                : "Standby"
    });
    resultText.innerHTML   = cameraActive ? "Scanning gate pass" : "System ready";
    resultText.style.color = "var(--ink)";
    statusDiv.innerHTML    = cameraActive
        ? (gateType === "vehicle" ? "Driver: please lower your window and look at the camera" : "Looking for a university member...")
        : 'Click "Open Camera" to start';
    statusDiv.style.color  = "var(--muted)";
    if (gateType === "vehicle") setDetails({ accessType: "Vehicle access" });
}

// --------------------------
// Lockdown polling
// --------------------------

let emergencyLockdown = false;   // all gates locked by the admin

// Walk-in vs vehicle look of the screen (title, subtitle, header chip)
const heroTitle = document.querySelector(".gt-title");
const heroSub   = document.querySelector(".gt-sub");
const chipState = document.getElementById("terminalClock");

function applyGateTypeUI() {
    const vehicle = gateType === "vehicle";
    if (heroTitle) heroTitle.textContent = vehicle ? "Welcome, driver" : "Welcome to campus";
    if (heroSub)   heroSub.textContent   = vehicle
        ? "Lower your window and look at the camera."
        : "Look at the camera to enter. Safe. Smart. Secure.";
    if (chipState) chipState.textContent = vehicle ? "🚗 Online" : "Online";
}

// Emergency lockdown + this gate's own lock and type, all from the server
async function checkLockdownStatus() {
    try {
        const response  = await fetch("/gates");
        const data      = await response.json();
        const mine      = (data.gates || []).find(g => g.id === gateId) || {};
        const wasLocked = lockdownActive;
        const oldType   = gateType;
        emergencyLockdown = data.lockdown === true;
        gateLocked        = mine.locked === true;
        gateType          = mine.type === "vehicle" ? "vehicle" : "walk";
        lockdownActive    = emergencyLockdown || gateLocked;
        if (oldType !== gateType) applyGateTypeUI();

        if (lockdownActive) {
            isPaused = true;
            showLockdownMessage();
        } else if (wasLocked) {
            isPaused = false;
            showReadyMessage();
        } else if (oldType !== gateType && !isProcessing && !isPaused) {
            showReadyMessage();   // admin switched walk-in <-> vehicle
        }
    } catch (error) {
        console.error("Gate status check failed", error);
    }
}

// --------------------------
// Camera control
// --------------------------

async function startCamera() {
    setScanAnimation(true);
    try {
        // Ask for 1280x720: wide-angle front cameras (iPads) make the face small,
        // so a sharper picture gives ArcFace enough detail. Devices that can't
        // do it simply give their best size.
        cameraStream        = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject     = cameraStream;
        cameraActive        = true;
        cameraBtn.textContent = "Close Camera";

        await checkLockdownStatus();
        if (!lockdownActive) showReadyMessage();
        resetBlink();
        if (blinkStatus === "ready") requestAnimationFrame(blinkLoop);
        if (blinkStatus === "failed" && !lockdownActive) showBlinkUnavailable();
        startCenterCheck();
        startProcessing();
    } catch (error) {
        setScanAnimation(false);
        window.AmanAudio?.prompt("Camera access is unavailable. Please allow camera access and try again.", "camera-error", true);
        statusDiv.innerHTML   = "Camera access denied";
        statusDiv.style.color = "var(--danger)";
    }
}

function stopCamera() {
    window.AmanAudio?.stop();
    if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
    cameraActive  = false;
    isProcessing  = false;
    isPaused      = false;
    centeredFrames = 0;
    setScanAnimation(false);
    cameraBtn.textContent = "Open Camera";
    if (processInterval) { clearInterval(processInterval); processInterval = null; }
    if (centerCheckInterval) { clearInterval(centerCheckInterval); centerCheckInterval = null; }
    lockdownActive ? showLockdownMessage() : showReadyMessage();
}

// --------------------------
// Frame capture
// --------------------------

function captureFrame(maxWidth = 480, quality = 0.7) {
    const canvas   = document.createElement("canvas");
    const scale    = Math.min(1, maxWidth / video.videoWidth);
    canvas.width   = video.videoWidth  * scale;
    canvas.height  = video.videoHeight * scale;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
}

// --------------------------
// Fast center guidance loop
// --------------------------

function startCenterCheck() {
    clearInterval(centerCheckInterval);
    centeredFrames = 0;

    centerCheckInterval = setInterval(() => {
        if (
            cameraActive &&
            !lockdownActive &&
            !isPaused &&
            !isProcessing &&
            !isCheckingCenter &&
            video.videoWidth > 0
        ) {
            checkFaceCenter();
        }
    }, CENTER_CHECK_INTERVAL);
}

async function checkFaceCenter() {
    isCheckingCenter = true;

    try {
        const latestFrame = captureFrame(480, 0.7);   // a bit larger so small (wide-angle) faces are still found
        const response = await fetch("/check-center", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ image: latestFrame, gate: gateId })
        });
        const data = await response.json();

        // A scan (or a result / lockdown) may have started while this check was in
        // flight — don't overwrite its "Verifying identity" with "Look straight".
        if (isProcessing || isPaused || lockdownActive || !cameraActive) return;

        if (data.centered) {
            centeredFrames = Math.min(centeredFrames + 1, CENTERED_FRAMES_REQUIRED);

            // Liveness first: no scan until a real blink has been seen
            if (blinkStatus === "failed") { showBlinkUnavailable(); return; }
            if (!hasRecentBlink()) {
                const loading = blinkStatus === "loading";
                // Only blinks after this prompt appears will count
                if (!loading && !blinkPromptAt) blinkPromptAt = Date.now();
                setTone("");
                setProgress(1, 50);
                setCameraMessage(loading ? "Getting ready..." : "Please blink", loading ? "" : "warning");
                setDetails({ accessType: "Aligned", personName: "Face detected", gateAction: "Waiting", status: "Blink to continue" });
                resultText.innerHTML   = loading ? "Preparing liveness check" : "Please blink once";
                resultText.style.color = "var(--ink)";
                statusDiv.innerHTML    = loading
                    ? "Loading the blink check..."
                    : "Blink naturally to confirm you are a real person";
                statusDiv.style.color  = "var(--muted)";
                return;
            }

            setTone("");
            setProgress(centeredFrames >= CENTERED_FRAMES_REQUIRED ? 2 : 1, centeredFrames >= CENTERED_FRAMES_REQUIRED ? 70 : 45);
            setCameraMessage("Look straight", "success");
            setDetails({
                accessType: "Aligned",
                personName: "Face detected",
                gateAction: centeredFrames >= CENTERED_FRAMES_REQUIRED ? "Verifying" : "Hold still",
                status: "Centered"
            });
            resultText.innerHTML = centeredFrames >= CENTERED_FRAMES_REQUIRED ? "Verifying Identity" : "Face Aligned";
            resultText.style.color = "var(--ink)";
            statusDiv.innerHTML = centeredFrames >= CENTERED_FRAMES_REQUIRED
                ? "Hold still for verification..."
                : "Keep your face in the middle";
            statusDiv.style.color = "var(--muted)";
            return;
        }

        centeredFrames = 0;
        if (data.reason === "multiple_faces") {
            resultText.innerHTML   = "ACCESS DENIED";
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(1, 30);
            setCameraMessage("One person only", "danger");
            setDetails({ accessType: "Violation", personName: "Multiple Faces", gateAction: "Contact Security", status: "Denied" });
            statusDiv.innerHTML   = "Only one person at a time";
            statusDiv.style.color = "var(--danger)";
            return;
        }

        resultText.innerHTML   = "Please position your face";
        resultText.style.color = "var(--amber)";
        setTone(data.reason === "no_face" ? "" : "warning");
        setProgress(1, data.reason === "no_face" ? 20 : 35);
        setCameraMessage(data.reason === "no_face" ? "Look straight at the camera" : getPositionMessage(data.message), data.reason === "no_face" ? "" : "warning");
        setDetails({ accessType: "Aligning", personName: "Waiting for face", gateAction: "Waiting", status: "Center your face" });
        statusDiv.innerHTML   = data.message || "Please position your face in the middle";
        statusDiv.style.color = data.reason === "no_face" ? "var(--muted)" : "var(--amber)";
    } catch (error) {
        console.error("Center check failed", error);
    } finally {
        isCheckingCenter = false;
    }
}

// --------------------------
// Scan loop
// --------------------------

function startProcessing() {
    clearInterval(processInterval);
    processInterval = setInterval(() => {
        const scanReady  = centeredFrames >= CENTERED_FRAMES_REQUIRED;
        const cooledDown = Date.now() - lastScanAt > SCAN_COOLDOWN_TIME;
        if (cameraActive && !lockdownActive && !isProcessing && !isPaused && scanReady && cooledDown && hasRecentBlink() && eyesReady() && video.videoWidth > 0) {
            resetBlink();   // each scan uses up the blink — the next person must blink again after a new prompt
            processFace();
        }
    }, PROCESS_INTERVAL);
}

// --------------------------
// Main scan function
// --------------------------

async function processFace() {
    if (lockdownActive) return;
    isProcessing = true;
    window.AmanAudio?.prompt("Verifying your identity. Please hold still.", "verifying", true);
    lastScanAt = Date.now();
    setUiState("Verifying");
    setTone("");
    setProgress(3, 85);
    setCameraMessage("Verifying identity...", "success");
    setDetails({ personName: "Checking...", gateAction: "Verifying", status: "Verifying" });
    resultText.innerHTML   = "Verifying identity...";
    resultText.style.color = "var(--ink)";
    statusDiv.innerHTML    = "Hold still — this can take a few seconds";
    statusDiv.style.color  = "var(--muted)";

    try {
        // Capture a short burst of frames instead of a single shot —
        // averages out one bad frame (blink, motion blur, angle).
        const frames = [];
        for (let i = 0; i < 3; i++) {
            // Up to 1280 px at good JPEG quality: keeps the face detailed even on
            // wide-angle cameras (the face search itself still runs at the same size)
            frames.push(captureFrame(1280, 0.85));
            if (i < 2) await new Promise(r => setTimeout(r, 120));
        }

        const response = await fetch("/scan", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ images: frames, gate: gateId })
        });
        let data;
        try { data = await response.json(); } catch (e) { data = null; }
        if (!data) throw new Error(`Unreadable server reply (HTTP ${response.status})`);
        window.AmanAudio?.result(data);

        // Server busy (too many attempts from this gate in a minute)
        if (data.status === "busy") {
            showScanProblem("Please wait a moment", data.message || "Too many attempts - please wait a moment");
            return;
        }

        // Positioning / photo-quality guidance from the scan route — not a denial,
        // nothing is logged: show what to fix and scan again once it's fixed.
        if (data.status === "no_face" || data.status === "position_error" || data.status === "adjust") {
            centeredFrames = 0;
            resultText.innerHTML   = "Please adjust";
            resultText.style.color = "var(--amber)";
            setTone("warning");
            setProgress(1, 35);
            setCameraMessage(
                data.status === "no_face" ? "Look straight at the camera"
                : data.status === "adjust" && !/turn|left|right|closer/i.test(data.message) ? data.message
                : getPositionMessage(data.message), "warning");
            setDetails({ accessType: "Aligning", personName: "Waiting for a clear face", gateAction: "Waiting", status: "Adjust" });
            statusDiv.innerHTML   = data.status === "no_face"
                ? "No face found - please look straight at the camera"
                : data.message || "Please position your face in the middle";
            statusDiv.style.color = "var(--amber)";
            isProcessing = false;
            return;
        }

        if (data.status === "multiple_faces") {
            centeredFrames = 0;
            resultText.innerHTML   = "ACCESS DENIED";
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(1, 35);
            setCameraMessage("One person only", "danger");
            setDetails({ accessType: "Violation", personName: "Multiple Faces", gateAction: "Contact Security", status: "Denied" });
            statusDiv.innerHTML   = "Only one person at a time";
            statusDiv.style.color = "var(--danger)";
            isProcessing = false;
            return;
        }

        if (data.status === "success") {
            isPaused = true;
            setUiState("Granted");
            setTone("success");
            setProgress(3, 100);
            setCameraMessage("Access granted", "success");

            // FIX: explicitly set all four identity card fields
            setDetails({
                accessType: data.role,
                personName: data.user,
                gateAction: "Gate Opening",
                status:     "Granted"
            });

            resultText.textContent = `Access granted — welcome, ${data.user}!`;
            resultText.style.color = "var(--success)";
            statusDiv.innerHTML    = `Role: ${data.role} · ${data.confidence}% match`;
            statusDiv.style.color  = "var(--success)";
            setTimeout(() => {
                isPaused             = false;
                lockdownActive ? showLockdownMessage() : showReadyMessage();
            }, SUCCESS_HOLD_TIME);

        } else if (data.status === "lockdown" || data.status === "gate_locked") {
            if (data.status === "lockdown") emergencyLockdown = true; else gateLocked = true;
            lockdownActive = true;
            isPaused       = true;
            showLockdownMessage();

        } else if (data.status === "spoof") {
            // Anti-spoofing: a screen or printed photo instead of a real face
            resultText.textContent = "Fake face detected";
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(3, 100);
            setCameraMessage("Fake face detected", "danger");
            setDetails({
                accessType: "Spoof attempt",
                personName: "Not a real face",
                gateAction: "Contact Security",
                status:     "Denied"
            });
            statusDiv.textContent = "A screen or printed photo was detected. Please use your real face.";
            statusDiv.style.color = "var(--danger)";

        } else if (data.status === "not_permitted") {
            // Recognised, but this person has no access at this gate
            resultText.innerHTML   = `Not permitted at the ${GATE_NAMES[gateId]}`;
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(3, 100);
            setCameraMessage("Wrong gate", "danger");
            setDetails({
                accessType: "No access at this gate",
                personName: data.user,
                gateAction: "Use your permitted gate",
                status:     "Denied"
            });
            statusDiv.textContent = `${data.user} is registered, but not for this gate.`;
            statusDiv.style.color = "var(--danger)";

        } else if (data.status === "outside_hours") {
            // Recognised, but outside the access hours set for their role
            resultText.textContent = "Outside access hours";
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(3, 100);
            setCameraMessage("Outside access hours", "danger");
            setDetails({
                accessType: `Allowed ${data.hours}`,
                personName: data.user,
                gateAction: "Come back during your hours",
                status:     "Denied"
            });
            statusDiv.textContent = `${data.user} may enter ${data.hours} only.`;
            statusDiv.style.color = "var(--danger)";

        } else {
            // Denied or unknown
            resultText.innerHTML   = "Unknown user";
            resultText.style.color = "var(--danger)";
            setTone("danger");
            setProgress(3, 100);
            setCameraMessage("Access denied", "danger");
            setDetails({
                accessType: "Not recognized",
                personName: "Unknown Profile",
                gateAction: "Contact Security",
                status:     "Denied"
            });
            statusDiv.innerHTML   = "Access Denied — Contact Security";
            statusDiv.style.color = "var(--danger)";
        }

    } catch (error) {
        // Network drop or unreadable reply: don't leave "Verifying identity..." on screen
        console.error("Scan request failed", error);
        showScanProblem("Connection problem", "Could not reach the server - please try again");
    } finally {
        isProcessing = false;
    }
}

// A scan that could not be completed (not a denial): say so, then the normal
// loop takes over again — the next scan needs a new blink, as always.
function showScanProblem(title, message) {
    window.AmanAudio?.prompt(message, "scan-problem", true);
    centeredFrames = 0;
    setUiState("Scanning");
    setTone("warning");
    setProgress(1, 20);
    setCameraMessage(title, "warning");
    setDetails({ personName: "Not verified yet", gateAction: "Waiting", status: "Try again" });
    resultText.textContent = title;
    resultText.style.color = "var(--amber)";
    statusDiv.textContent  = message;
    statusDiv.style.color  = "var(--amber)";
}

// --------------------------
// Button + init
// --------------------------

cameraBtn.addEventListener("click", () => {
    cameraActive ? stopCamera() : startCamera();
});

// Start loading the blink detector straight away so it is ready by the first scan
loadBlinkDetector();

// Poll lockdown every 5 seconds instead of every 1 second
checkLockdownStatus();
lockdownInterval = setInterval(checkLockdownStatus, LOCKDOWN_CHECK_INTERVAL);
