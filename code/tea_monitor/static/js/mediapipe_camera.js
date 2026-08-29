/**
 * MediaPipe Camera Pipeline & Real-Time Orchestrator
 * Bulletproof Mobile & Desktop Support
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elementos do DOM
    const videoElement = document.getElementById("videoElement");
    const outputCanvas = document.getElementById("outputCanvas");
    const canvasCtx = outputCanvas ? outputCanvas.getContext("2d") : null;

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

    // Configuração do vídeo para mobile (iOS / Android)
    videoElement.setAttribute("playsinline", "");
    videoElement.setAttribute("webkit-playsinline", "");
    videoElement.muted = true;
    videoElement.autoplay = true;

    // Instância do Detector de Estereotipias
    const detector = window.StimmingDetector ? new window.StimmingDetector() : null;

    let isRunning = false;
    let isProcessing = false;
    let facingMode = "user"; // 'user' (frontal) ou 'environment' (traseira)
    let showPose = true;
    let showFace = true;
    let audioAlertEnabled = false;
    let currentStream = null;

    // FPS
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Modelos MediaPipe
    let poseModel = null;
    let faceModel = null;
    let latestPoseLandmarks = null;
    let latestFaceLandmarks = null;
    let modelsReady = false;

    // Web Audio
    let audioCtx = null;

    function playBeep(freq = 587.33, duration = 0.15) {
        if (!audioAlertEnabled) return;
        try {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === "suspended") audioCtx.resume();
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
            console.error("Audio error:", e);
        }
    }

    /**
     * Inicializa os Modelos MediaPipe de forma segura
     */
    async function initMediaPipeModels() {
        try {
            if (typeof Pose !== "undefined") {
                poseModel = new Pose({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
                });
                poseModel.setOptions({
                    modelComplexity: 0, // 0 = mais rápido para mobile
                    smoothLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                poseModel.onResults((results) => {
                    latestPoseLandmarks = results.poseLandmarks || null;
                });
            }

            if (typeof FaceMesh !== "undefined") {
                faceModel = new FaceMesh({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
                });
                faceModel.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: false,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                faceModel.onResults((results) => {
                    latestFaceLandmarks = (results.multiFaceLandmarks && results.multiFaceLandmarks[0]) || null;
                });
            }

            modelsReady = true;
            if (connectionBadge && !isRunning) {
                connectionBadge.textContent = "Pronto";
                connectionBadge.className = "status-badge status-ready";
            }
        } catch (e) {
            console.warn("Aviso ao carregar MediaPipe, continuando com câmera:", e);
        }
    }

    /**
     * Inicia a Câmera do Dispositivo com fallback robusto
     */
    async function startCamera() {
        btnStartCamera.disabled = true;
        btnStartCamera.textContent = "⏳ Conectando câmera...";
        if (connectionBadge) connectionBadge.textContent = "Acessando câmera...";

        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }

        // Tenta constraints ideais e faz fallback progressivo
        const constraintTiers = [
            { video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
            { video: { facingMode: facingMode }, audio: false },
            { video: true, audio: false }
        ];

        let stream = null;
        let lastErr = null;

        for (const constraints of constraintTiers) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (stream) break;
            } catch (err) {
                lastErr = err;
                console.warn("Tentativa de câmera falhou, tentando fallback:", err);
            }
        }

        if (!stream) {
            btnStartCamera.disabled = false;
            btnStartCamera.textContent = "▶️ Tentar Novamente";
            const errMsg = lastErr ? (lastErr.name || lastErr.message) : "Desconhecido";
            alert(`Não foi possível acessar a câmera (${errMsg}).\n\nCertifique-se de autorizar a câmera nas configurações do navegador.`);
            if (connectionBadge) connectionBadge.textContent = "Permissão Negada";
            return;
        }

        currentStream = stream;
        videoElement.srcObject = stream;

        try {
            await videoElement.play();
        } catch (playErr) {
            console.warn("Aguardando evento de play:", playErr);
        }

        // Configura dimensões do canvas
        const w = videoElement.videoWidth || 640;
        const h = videoElement.videoHeight || 480;
        outputCanvas.width = w;
        outputCanvas.height = h;

        startOverlay.style.display = "none";
        isRunning = true;
        if (connectionBadge) {
            connectionBadge.textContent = "Monitorando";
            connectionBadge.className = "status-badge status-monitoring";
        }

        // Garante que o modelo começou a carregar se ainda não iniciou
        if (!modelsReady) {
            initMediaPipeModels();
        }

        requestAnimationFrame(processLoop);
    }

    /**
     * Loop Principal de Processamento e Renderização
     */
    async function processLoop() {
        if (!isRunning) return;

        if (videoElement.readyState >= 2) {
            // 1. Atualiza dimensões se mudaram
            if (outputCanvas.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
                outputCanvas.width = videoElement.videoWidth;
                outputCanvas.height = videoElement.videoHeight;
            }

            // 2. Renderiza no Canvas primeiro para feed imediato
            renderCanvas();

            // 3. Processa IA se não estiver sobrecarregado
            if (!isProcessing) {
                isProcessing = true;
                try {
                    if (poseModel) await poseModel.send({ image: videoElement });
                    if (faceModel && showFace) await faceModel.send({ image: videoElement });

                    if (detector) {
                        const analysis = detector.processFrame(latestPoseLandmarks, latestFaceLandmarks);
                        updateUI(analysis);

                        if (analysis.shouldTriggerEvent && analysis.eventPayload) {
                            logEventToBackend(analysis.eventPayload);
                            playBeep(analysis.severity === "critical" ? 880 : 587);
                        }
                    }
                } catch (procErr) {
                    console.warn("Erro no processamento do frame MediaPipe:", procErr);
                } finally {
                    isProcessing = false;
                }
            }

            // FPS
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                if (fpsValue) fpsValue.textContent = Math.round((frameCount * 1000) / (now - lastFpsTime));
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
        if (!canvasCtx) return;
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

        // Se for frontal, espelha
        if (facingMode === "user") {
            canvasCtx.scale(-1, 1);
            canvasCtx.translate(-outputCanvas.width, 0);
        }

        // Imagem da câmera
        canvasCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

        // Esqueleto Pose
        if (showPose && latestPoseLandmarks && window.drawConnectors && window.drawLandmarks && typeof POSE_CONNECTIONS !== "undefined") {
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

        // Malha Facial
        if (showFace && latestFaceLandmarks && window.drawConnectors && typeof FACEMESH_TESSELATION !== "undefined") {
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
        if (!analysis) return;

        if (alertBanner) {
            alertBanner.className = `alert-banner alert-${analysis.severity}`;
            if (alertTitle) alertTitle.textContent = analysis.title;
            if (alertDesc) alertDesc.textContent = analysis.desc;

            if (alertIcon) {
                if (analysis.severity === "critical") alertIcon.textContent = "🔴";
                else if (analysis.severity === "warning") alertIcon.textContent = "🟡";
                else alertIcon.textContent = "🟢";
            }
        }

        // HUD Bars
        const flapPct = (analysis.flapping && analysis.flapping.score) || 0;
        const sensoryPct = (analysis.sensoryCovering && analysis.sensoryCovering.score) || 0;
        const rockPct = (analysis.rocking && analysis.rocking.score) || 0;

        if (hudFlappingBar) {
            hudFlappingBar.style.width = `${flapPct}%`;
            hudFlappingBar.style.backgroundColor = flapPct > 60 ? "var(--accent-orange)" : "var(--accent-blue)";
        }
        if (hudSensoryBar) {
            hudSensoryBar.style.width = `${sensoryPct}%`;
            hudSensoryBar.style.backgroundColor = sensoryPct > 60 ? "var(--accent-red)" : "var(--accent-blue)";
        }
        if (hudRockingBar) {
            hudRockingBar.style.width = `${rockPct}%`;
            hudRockingBar.style.backgroundColor = rockPct > 60 ? "var(--accent-yellow)" : "var(--accent-blue)";
        }

        // Metrics Grid Cards
        if (flappingScore) flappingScore.textContent = flapPct;
        if (flappingHz && analysis.flapping) flappingHz.textContent = `${analysis.flapping.hz} Hz`;

        if (sensoryScore) sensoryScore.textContent = sensoryPct;
        if (sensoryStatus && analysis.sensoryCovering) sensoryStatus.textContent = analysis.sensoryCovering.status;

        if (rockingScore) rockingScore.textContent = rockPct;
        if (rockingStatus && analysis.rocking) rockingStatus.textContent = analysis.rocking.status;
    }

    /**
     * Envia o evento para a API do Servidor
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
            // Falha silenciosa
        }
    }

    function addEventToDOM(event) {
        if (!eventList) return;
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

    // --- Listeners de Ações ---
    if (btnStartCamera) {
        btnStartCamera.addEventListener("click", () => {
            startCamera();
        });
    }

    if (btnFlipCamera) {
        btnFlipCamera.addEventListener("click", () => {
            facingMode = facingMode === "user" ? "environment" : "user";
            startCamera();
        });
    }

    if (btnTogglePose) {
        btnTogglePose.addEventListener("click", () => {
            showPose = !showPose;
            btnTogglePose.classList.toggle("active", showPose);
        });
    }

    if (btnToggleFace) {
        btnToggleFace.addEventListener("click", () => {
            showFace = !showFace;
            btnToggleFace.classList.toggle("active", showFace);
        });
    }

    if (btnToggleAudio) {
        btnToggleAudio.addEventListener("click", () => {
            audioAlertEnabled = !audioAlertEnabled;
            btnToggleAudio.classList.toggle("active", audioAlertEnabled);
            if (audioAlertEnabled) playBeep(440, 0.1);
        });
    }

    if (btnClearLog) {
        btnClearLog.addEventListener("click", async () => {
            if (eventList) eventList.innerHTML = `<div class="empty-state">Histórico limpo.</div>`;
            try { await fetch("/api/events", { method: "DELETE" }); } catch (_) {}
        });
    }

    // Inicializa carregamento assíncrono dos modelos
    initMediaPipeModels();
});
