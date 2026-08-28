/**
 * MediaPipe Camera Pipeline & Real-Time Orchestrator
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elementos do DOM
    const videoElement = document.getElementById("videoElement");
    const outputCanvas = document.getElementById("outputCanvas");
    const canvasCtx = outputCanvas.getContext("2d");

    const btnStartCamera = document.getElementById("btnStartCamera");
    const btnFlipCamera = document.getElementById("btnFlipCamera");
    const btnTogglePose = document.getElementById("btnTogglePose");
    const btnToggleFace = document.getElementById("btnToggleFace");
    const btnToggleAudio = document.getElementById("btnToggleAudio");
    const btnClearLog = document.getElementById("btnClearLog");
    const startOverlay = document.getElementById("startOverlay");

    const alertBanner = document.getElementById("alertBanner");
    const alertIcon = document.getElementById("alertIcon");
    const alertTitle = document.getElementById("alertTitle");
    const alertDesc = document.getElementById("alertDesc");
    const connectionBadge = document.getElementById("connectionBadge");
    const fpsValue = document.getElementById("fpsValue");

    // HUD Bars
    const hudFlappingBar = document.getElementById("hudFlappingBar");
    const hudSensoryBar = document.getElementById("hudSensoryBar");
    const hudRockingBar = document.getElementById("hudRockingBar");

    // Metrics Card Elements
    const flappingScore = document.getElementById("flappingScore");
    const flappingHz = document.getElementById("flappingHz");
    const sensoryScore = document.getElementById("sensoryScore");
    const sensoryStatus = document.getElementById("sensoryStatus");
    const rockingScore = document.getElementById("rockingScore");
    const rockingStatus = document.getElementById("rockingStatus");
    const eventList = document.getElementById("eventList");

    // Instância do Detector de Estereotipias
    const detector = new StimmingDetector();

    // Configurações e Estados
    let isRunning = false;
    let facingMode = "user"; // 'user' (frontal) ou 'environment' (traseira)
    let showPose = true;
    let showFace = true;
    let audioAlertEnabled = false;
    let currentStream = null;

    // Métricas de FPS
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Instâncias MediaPipe
    let poseModel = null;
    let faceModel = null;
    let latestPoseLandmarks = null;
    let latestFaceLandmarks = null;

    // Web Audio API para alertas discretos
    let audioCtx = null;

    function playBeep(freq = 587.33, duration = 0.15) { // D5 note
        if (!audioAlertEnabled) return;
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = "sine";
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        } catch (e) {
            console.error("Audio beep error:", e);
        }
    }

    /**
     * Inicializa os Modelos MediaPipe
     */
    async function initMediaPipeModels() {
        connectionBadge.textContent = "Carregando Modelos IA...";

        // 1. Pose
        poseModel = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });
        poseModel.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        poseModel.onResults(onPoseResults);

        // 2. FaceMesh
        faceModel = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        faceModel.setOptions({
            maxNumFaces: 1,
            refineLandmarks: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        faceModel.onResults(onFaceResults);

        connectionBadge.textContent = "Modelos Prontos";
        connectionBadge.className = "status-badge status-ready";
    }

    function onPoseResults(results) {
        latestPoseLandmarks = results.poseLandmarks || null;
    }

    function onFaceResults(results) {
        latestFaceLandmarks = (results.multiFaceLandmarks && results.multiFaceLandmarks[0]) || null;
    }

    /**
     * Inicia a Câmera do Dispositivo
     */
    async function startCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }

        try {
            connectionBadge.textContent = "Conectando Câmera...";
            const constraints = {
                video: {
                    facingMode: facingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };

            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = currentStream;

            await new Promise((resolve) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play();
                    resolve();
                };
            });

            outputCanvas.width = videoElement.videoWidth || 640;
            outputCanvas.height = videoElement.videoHeight || 480;

            startOverlay.style.display = "none";
            isRunning = true;
            connectionBadge.textContent = "Monitorando";
            connectionBadge.className = "status-badge status-monitoring";

            requestAnimationFrame(processLoop);
        } catch (err) {
            console.error("Erro ao acessar câmera:", err);
            alert("Erro ao acessar a câmera. Certifique-se de que autorizou as permissões de vídeo no navegador (HTTPS requerido).");
            connectionBadge.textContent = "Erro na Câmera";
        }
    }

    /**
     * Loop Principal de Processamento e Renderização
     */
    async function processLoop() {
        if (!isRunning) return;

        // Envia frame de vídeo para MediaPipe
        if (videoElement.readyState >= 2) {
            if (poseModel) await poseModel.send({ image: videoElement });
            if (faceModel && showFace) await faceModel.send({ image: videoElement });

            // Renderiza no Canvas
            renderCanvas();

            // Processa inferência biomecânica
            const analysis = detector.processFrame(latestPoseLandmarks, latestFaceLandmarks);
            updateUI(analysis);

            // Se for evento crítico, despacha para backend
            if (analysis.shouldTriggerEvent && analysis.eventPayload) {
                logEventToBackend(analysis.eventPayload);
                playBeep(analysis.severity === "critical" ? 880 : 587);
            }

            // Cálculo de FPS
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                fpsValue.textContent = Math.round((frameCount * 1000) / (now - lastFpsTime));
                frameCount = 0;
                lastFpsTime = now;
            }
        }

        requestAnimationFrame(processLoop);
    }

    /**
     * Renderiza o vídeo e os esqueletos/malhas no Canvas
     */
    function renderCanvas() {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

        // Se estiver em câmera frontal, espelha para sensação natural
        if (facingMode === "user") {
            canvasCtx.scale(-1, 1);
            canvasCtx.translate(-outputCanvas.width, 0);
        }

        // Desenha imagem da câmera
        canvasCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

        // Desenha Esqueleto de Pose
        if (showPose && latestPoseLandmarks && window.drawConnectors && window.drawLandmarks) {
            drawConnectors(canvasCtx, latestPoseLandmarks, POSE_CONNECTIONS, {
                color: "#38bdf8",
                lineWidth: 3
            });
            drawLandmarks(canvasCtx, latestPoseLandmarks, {
                color: "#f43f5e",
                lineWidth: 1,
                radius: 4
            });
        }

        // Desenha Malha Facial
        if (showFace && latestFaceLandmarks && window.drawConnectors) {
            drawConnectors(canvasCtx, latestFaceLandmarks, FACEMESH_TESSELATION, {
                color: "rgba(255, 255, 255, 0.15)",
                lineWidth: 1
            });
        }

        canvasCtx.restore();
    }

    /**
     * Atualiza o HUD e os Cards com as Métricas
     */
    function updateUI(analysis) {
        // Banner de Alerta
        alertBanner.className = `alert-banner alert-${analysis.severity}`;
        alertTitle.textContent = analysis.title;
        alertDesc.textContent = analysis.desc;

        if (analysis.severity === "critical") alertIcon.textContent = "🔴";
        else if (analysis.severity === "warning") alertIcon.textContent = "🟡";
        else alertIcon.textContent = "🟢";

        // HUD Bars
        const flapPct = analysis.flapping.score;
        const sensoryPct = analysis.sensoryCovering.score;
        const rockPct = analysis.rocking.score;

        hudFlappingBar.style.width = `${flapPct}%`;
        hudFlappingBar.style.backgroundColor = flapPct > 60 ? "var(--accent-orange)" : "var(--accent-blue)";

        hudSensoryBar.style.width = `${sensoryPct}%`;
        hudSensoryBar.style.backgroundColor = sensoryPct > 60 ? "var(--accent-red)" : "var(--accent-blue)";

        hudRockingBar.style.width = `${rockPct}%`;
        hudRockingBar.style.backgroundColor = rockPct > 60 ? "var(--accent-yellow)" : "var(--accent-blue)";

        // Metrics Grid Cards
        flappingScore.textContent = flapPct;
        flappingHz.textContent = `${analysis.flapping.hz} Hz`;

        sensoryScore.textContent = sensoryPct;
        sensoryStatus.textContent = analysis.sensoryCovering.status;

        rockingScore.textContent = rockPct;
        rockingStatus.textContent = analysis.rocking.status;
    }

    /**
     * Envia o evento de sobrecarga/stimming para a API do Servidor
     */
    async function logEventToBackend(payload) {
        addEventToDOM(payload);
        try {
            await fetch("/api/events", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } catch (e) {
            console.warn("Backend offline, armazenado apenas no cliente:", e);
        }
    }

    function addEventToDOM(event) {
        const emptyState = eventList.querySelector(".empty-state");
        if (emptyState) emptyState.remove();

        const time = new Date().toLocaleTimeString();
        const item = document.createElement("div");
        item.className = `event-item ${event.severity === "critical" ? "critical" : ""}`;
        item.innerHTML = `
            <div>
                <strong>${event.label}</strong>
                <div style="color: #94a3b8; font-size: 10px;">Confiança: ${(event.confidence * 100).toFixed(0)}%</div>
            </div>
            <span style="font-family: monospace; color: #38bdf8;">${time}</span>
        `;
        eventList.prepend(item);
    }

    // --- Listeners de Ações e Botões ---

    btnStartCamera.addEventListener("click", () => {
        startCamera();
    });

    btnFlipCamera.addEventListener("click", () => {
        facingMode = facingMode === "user" ? "environment" : "user";
        startCamera();
    });

    btnTogglePose.addEventListener("click", () => {
        showPose = !showPose;
        btnTogglePose.classList.toggle("active", showPose);
    });

    btnToggleFace.addEventListener("click", () => {
        showFace = !showFace;
        btnToggleFace.classList.toggle("active", showFace);
    });

    btnToggleAudio.addEventListener("click", () => {
        audioAlertEnabled = !audioAlertEnabled;
        btnToggleAudio.classList.toggle("active", audioAlertEnabled);
        if (audioAlertEnabled) playBeep(440, 0.1);
    });

    btnClearLog.addEventListener("click", async () => {
        eventList.innerHTML = `<div class="empty-state">Histórico limpo.</div>`;
        try { await fetch("/api/events", { method: "DELETE" }); } catch (_) {}
    });

    // Inicia carregamento dos modelos MediaPipe na inicialização
    initMediaPipeModels();
});
