/**
 * MediaPipe Camera Pipeline & Real-Time Orchestrator
 * Multimodal: Pose + FaceMesh + Web Audio Decibel Meter
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Camera & Canvas
    const videoElement = document.getElementById("videoElement");
    const outputCanvas = document.getElementById("outputCanvas");
    const canvasCtx = outputCanvas ? outputCanvas.getContext("2d") : null;
    const timelineCanvas = document.getElementById("timelineChart");
    const timelineCtx = timelineCanvas ? timelineCanvas.getContext("2d") : null;

    // Controls & Buttons
    const btnStartCamera = document.getElementById("btnStartCamera");
    const btnFlipCamera = document.getElementById("btnFlipCamera");
    const btnTogglePose = document.getElementById("btnTogglePose");
    const btnToggleFace = document.getElementById("btnToggleFace");
    const btnToggleAudio = document.getElementById("btnToggleAudio");
    const btnClearLog = document.getElementById("btnClearLog");
    const btnExportReport = document.getElementById("btnExportReport");
    const btnSaveSettings = document.getElementById("btnSaveSettings");
    const startOverlay = document.getElementById("startOverlay");

    // Status & Badges
    const alertBanner = document.getElementById("alertBanner");
    const alertIcon = document.getElementById("alertIcon");
    const alertStageName = document.getElementById("alertStageName");
    const alertDesc = document.getElementById("alertDesc");
    const teacherRecommendation = document.getElementById("teacherRecommendation");
    const connectionBadge = document.getElementById("connectionBadge");
    const fpsValue = document.getElementById("fpsValue");

    // HUD Bars
    const hudNoiseBar = document.getElementById("hudNoiseBar");
    const hudNoiseText = document.getElementById("hudNoiseText");
    const hudFlappingBar = document.getElementById("hudFlappingBar");
    const hudSensoryBar = document.getElementById("hudSensoryBar");
    const hudRockingBar = document.getElementById("hudRockingBar");

    // Metrics Cards
    const noiseDb = document.getElementById("noiseDb");
    const noiseStatus = document.getElementById("noiseStatus");
    const flappingScore = document.getElementById("flappingScore");
    const flappingHz = document.getElementById("flappingHz");
    const sensoryScore = document.getElementById("sensoryScore");
    const sensoryStatus = document.getElementById("sensoryStatus");
    const rockingScore = document.getElementById("rockingScore");
    const rockingStatus = document.getElementById("rockingStatus");
    const gazeScore = document.getElementById("gazeScore");
    const gazeStatus = document.getElementById("gazeStatus");
    const overallStressScore = document.getElementById("overallStressScore");
    const eventList = document.getElementById("eventList");

    // Clinical Report Elements
    const cntFlapping = document.getElementById("cntFlapping");
    const cntRocking = document.getElementById("cntRocking");
    const cntAuditory = document.getElementById("cntAuditory");
    const cntVisual = document.getElementById("cntVisual");
    const cntNodding = document.getElementById("cntNodding");
    const cntFreeze = document.getElementById("cntFreeze");
    const reportSummaryText = document.getElementById("reportSummaryText");

    // Mobile video setup
    videoElement.setAttribute("playsinline", "");
    videoElement.setAttribute("webkit-playsinline", "");
    videoElement.muted = true;
    videoElement.autoplay = true;

    // Detector Instance
    const detector = window.StimmingDetector ? new window.StimmingDetector() : null;

    let isRunning = false;
    let isProcessing = false;
    let facingMode = "user";
    let showPose = true;
    let showFace = true;
    let audioAlertEnabled = false;
    let currentStream = null;

    // Web Audio Analyser para Decibéis da Sala
    let micAudioCtx = null;
    let audioAnalyser = null;
    let audioDataArray = null;
    let currentDecibels = 50;

    // FPS
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // MediaPipe Models
    let poseModel = null;
    let faceModel = null;
    let latestPoseLandmarks = null;
    let latestFaceLandmarks = null;
    let modelsReady = false;

    // Web Audio Alert Beeper
    let beeperAudioCtx = null;

    function playBeep(freq = 587.33, duration = 0.15) {
        if (!audioAlertEnabled) return;
        try {
            if (!beeperAudioCtx) beeperAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (beeperAudioCtx.state === "suspended") beeperAudioCtx.resume();
            const osc = beeperAudioCtx.createOscillator();
            const gain = beeperAudioCtx.createGain();
            osc.type = "sine";
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.1, beeperAudioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, beeperAudioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(beeperAudioCtx.destination);
            osc.start();
            osc.stop(beeperAudioCtx.currentTime + duration);
        } catch (e) {
            console.error("Audio error:", e);
        }
    }

    /**
     * Configura o analisador de ruído do microfone (Decibéis)
     */
    function setupMicrophoneAnalysis(stream) {
        try {
            const audioTracks = stream.getAudioTracks();
            if (audioTracks.length > 0) {
                micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const source = micAudioCtx.createMediaStreamSource(stream);
                audioAnalyser = micAudioCtx.createAnalyser();
                audioAnalyser.fftSize = 256;
                audioAnalyser.smoothingTimeConstant = 0.8;
                source.connect(audioAnalyser);
                audioDataArray = new Uint8Array(audioAnalyser.frequencyBinCount);
            }
        } catch (e) {
            console.warn("Aviso ao configurar microfone de ruído:", e);
        }
    }

    function sampleCurrentDecibels() {
        if (!audioAnalyser || !audioDataArray) return 50;
        audioAnalyser.getByteTimeDomainData(audioDataArray);
        let sum = 0;
        for (let i = 0; i < audioDataArray.length; i++) {
            const val = (audioDataArray[i] - 128) / 128;
            sum += val * val;
        }
        const rms = Math.sqrt(sum / audioDataArray.length);
        // SPL Aproximado em dB calibrado para microfones comuns
        currentDecibels = Math.max(38, Math.min(100, Math.round(20 * Math.log10(rms + 1e-4) + 98)));
        return currentDecibels;
    }

    /**
     * Inicializa os Modelos MediaPipe
     */
    async function initMediaPipeModels() {
        try {
            if (typeof Pose !== "undefined") {
                poseModel = new Pose({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
                });
                poseModel.setOptions({
                    modelComplexity: 0,
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
            console.warn("Aviso MediaPipe:", e);
        }
    }

    /**
     * Inicia a Câmera e Microfone
     */
    async function startCamera() {
        btnStartCamera.disabled = true;
        btnStartCamera.textContent = "⏳ Conectando sensores...";
        if (connectionBadge) connectionBadge.textContent = "Acessando sensores...";

        if (currentStream) {
            currentStream.getTracks().forEach(t => t.stop());
            currentStream = null;
        }

        const constraintTiers = [
            { video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: true },
            { video: { facingMode: facingMode }, audio: true },
            { video: true, audio: true },
            { video: true, audio: false } // Fallback se o microfone for negado
        ];

        let stream = null;
        let lastErr = null;

        for (const c of constraintTiers) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(c);
                if (stream) break;
            } catch (err) {
                lastErr = err;
            }
        }

        if (!stream) {
            btnStartCamera.disabled = false;
            btnStartCamera.textContent = "▶️ Tentar Novamente";
            const msg = lastErr ? (lastErr.name || lastErr.message) : "Desconhecido";
            alert(`Não foi possível acessar os sensores (${msg}).\n\nAutorize câmera e microfone no navegador.`);
            if (connectionBadge) connectionBadge.textContent = "Permissão Negada";
            return;
        }

        currentStream = stream;
        videoElement.srcObject = stream;

        // Configura análise do microfone
        setupMicrophoneAnalysis(stream);

        try {
            await videoElement.play();
        } catch (playErr) {
            console.warn("Play:", playErr);
        }

        outputCanvas.width = videoElement.videoWidth || 640;
        outputCanvas.height = videoElement.videoHeight || 480;

        startOverlay.style.display = "none";
        isRunning = true;
        if (connectionBadge) {
            connectionBadge.textContent = "Monitorando";
            connectionBadge.className = "status-badge status-monitoring";
        }

        if (!modelsReady) initMediaPipeModels();

        requestAnimationFrame(processLoop);
    }

    /**
     * Loop Principal de Processamento Multimodal
     */
    async function processLoop() {
        if (!isRunning) return;

        if (videoElement.readyState >= 2) {
            if (outputCanvas.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
                outputCanvas.width = videoElement.videoWidth;
                outputCanvas.height = videoElement.videoHeight;
            }

            renderCanvas();

            // Mede decibéis atuais do ambiente
            const db = sampleCurrentDecibels();

            if (!isProcessing) {
                isProcessing = true;
                try {
                    if (poseModel) await poseModel.send({ image: videoElement });
                    if (faceModel && showFace) await faceModel.send({ image: videoElement });

                    if (detector) {
                        const analysis = detector.processFrame(latestPoseLandmarks, latestFaceLandmarks, db);
                        updateUI(analysis);
                        drawTimelineChart(detector.timelineHistory);

                        if (analysis.shouldTriggerEvent && analysis.eventPayload) {
                            logEventToBackend(analysis.eventPayload);
                            playBeep(analysis.severity === "critical" ? 880 : 587);
                        }
                    }
                } catch (procErr) {
                    console.warn("Proc error:", procErr);
                } finally {
                    isProcessing = false;
                }
            }

            // FPS Counter
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
     * Renderiza o Vídeo e Esqueleto no Canvas
     */
    function renderCanvas() {
        if (!canvasCtx) return;
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

        if (facingMode === "user") {
            canvasCtx.scale(-1, 1);
            canvasCtx.translate(-outputCanvas.width, 0);
        }

        canvasCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

        // Pose Skeleton
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

        // FaceMesh
        if (showFace && latestFaceLandmarks && window.drawConnectors && typeof FACEMESH_TESSELATION !== "undefined") {
            drawConnectors(canvasCtx, latestFaceLandmarks, FACEMESH_TESSELATION, {
                color: "rgba(255, 255, 255, 0.15)",
                lineWidth: 1
            });
        }

        canvasCtx.restore();
    }

    /**
     * Renderiza o Gráfico de Estresse & Ruído
     */
    function drawTimelineChart(timeline) {
        if (!timelineCtx || !timeline || timeline.length === 0) return;

        const w = timelineCanvas.width;
        const h = timelineCanvas.height;

        timelineCtx.clearRect(0, 0, w, h);

        // Linhas de Grade
        timelineCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        timelineCtx.lineWidth = 1;
        for (let y = 0; y <= h; y += 40) {
            timelineCtx.beginPath();
            timelineCtx.moveTo(0, y);
            timelineCtx.lineTo(w, y);
            timelineCtx.stroke();
        }

        // Linha Crítica de Sobrecarga (70%)
        const critY = h - (0.7 * h);
        timelineCtx.strokeStyle = "rgba(239, 68, 68, 0.5)";
        timelineCtx.setLineDash([4, 4]);
        timelineCtx.beginPath();
        timelineCtx.moveTo(0, critY);
        timelineCtx.lineTo(w, critY);
        timelineCtx.stroke();
        timelineCtx.setLineDash([]);

        // Curva de Estresse (Azul)
        timelineCtx.beginPath();
        const step = w / Math.max(1, timeline.length - 1);

        for (let i = 0; i < timeline.length; i++) {
            const x = i * step;
            const y = h - (timeline[i].stress / 100) * (h - 20) - 10;
            if (i === 0) timelineCtx.moveTo(x, y);
            else timelineCtx.lineTo(x, y);
        }

        timelineCtx.strokeStyle = "#38bdf8";
        timelineCtx.lineWidth = 2.5;
        timelineCtx.stroke();

        // Gradiente sob a curva
        timelineCtx.lineTo((timeline.length - 1) * step, h);
        timelineCtx.lineTo(0, h);
        const grad = timelineCtx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, "rgba(56, 189, 248, 0.35)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");
        timelineCtx.fillStyle = grad;
        timelineCtx.fill();
    }

    /**
     * Atualiza a Interface com Métricas Multimodais
     */
    function updateUI(analysis) {
        if (!analysis) return;

        // Banner e Estágio
        if (alertBanner) alertBanner.className = `alert-banner alert-${analysis.severity}`;
        if (alertStageName) alertStageName.textContent = analysis.stageName;
        if (alertDesc) alertDesc.textContent = analysis.desc;
        if (teacherRecommendation) teacherRecommendation.textContent = analysis.recommendation;

        if (alertIcon) {
            if (analysis.severity === "critical") alertIcon.textContent = "🔴";
            else if (analysis.severity === "alert") alertIcon.textContent = "🟠";
            else if (analysis.severity === "warning") alertIcon.textContent = "🟡";
            else alertIcon.textContent = "🟢";
        }

        // HUD Ruído & Decibéis
        const db = analysis.ambientDb || 50;
        if (noiseDb) noiseDb.textContent = db;
        if (hudNoiseText) hudNoiseText.textContent = `${db} dB`;
        if (hudNoiseBar) {
            const pct = Math.min(100, Math.max(0, ((db - 40) / 50) * 100));
            hudNoiseBar.style.width = `${pct}%`;
            hudNoiseBar.style.backgroundColor = db >= 78 ? "var(--accent-red)" : db >= 68 ? "var(--accent-yellow)" : "var(--accent-green)";
        }
        if (noiseStatus) {
            noiseStatus.textContent = db >= 78 ? "Muito Alto" : db >= 68 ? "Moderado" : "Silencioso";
            noiseStatus.style.color = db >= 78 ? "var(--accent-red)" : db >= 68 ? "var(--accent-yellow)" : "var(--accent-green)";
        }

        // HUD Bars
        const flapPct = (analysis.flapping && analysis.flapping.score) || 0;
        const sensoryPct = (analysis.auditoryDefense && analysis.auditoryDefense.score) || 0;
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
        if (sensoryStatus && analysis.auditoryDefense) sensoryStatus.textContent = analysis.auditoryDefense.type || "Normal";

        if (rockingScore) rockingScore.textContent = rockPct;
        if (rockingStatus && analysis.rocking) rockingStatus.textContent = analysis.rocking.axis || "Estável";

        if (gazeScore && analysis.gazeAttention) gazeScore.textContent = analysis.gazeAttention.focusScore;
        if (gazeStatus && analysis.gazeAttention) gazeStatus.textContent = analysis.gazeAttention.status;

        if (overallStressScore) overallStressScore.textContent = analysis.stressScore || 0;

        // Atualiza Contadores Clínicos
        if (detector) {
            if (cntFlapping) cntFlapping.textContent = detector.behaviorCounts.hand_flapping;
            if (cntRocking) cntRocking.textContent = detector.behaviorCounts.body_rocking;
            if (cntAuditory) cntAuditory.textContent = detector.behaviorCounts.sensory_auditory;
            if (cntVisual) cntVisual.textContent = detector.behaviorCounts.sensory_visual;
            if (cntNodding) cntNodding.textContent = detector.behaviorCounts.head_nodding;
            if (cntFreeze) cntFreeze.textContent = detector.behaviorCounts.shutdown_freeze;
            if (reportSummaryText) reportSummaryText.textContent = detector.generateReportSummary();
        }
    }

    /**
     * Envia evento para a API
     */
    async function logEventToBackend(payload) {
        addEventToDOM(payload);
        try {
            await fetch("/api/events", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
        } catch (e) {}
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
                <div style="color: #94a3b8; font-size: 10px;">${event.recommendation || ""}</div>
            </div>
            <span style="font-family: monospace; color: #38bdf8; font-size: 10px;">${time}</span>
        `;
        eventList.prepend(item);
    }

    // --- Navegação por Abas ---
    const tabButtons = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");
        });
    });

    // --- Listeners de Controles ---
    if (btnStartCamera) btnStartCamera.addEventListener("click", startCamera);

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

    // Exportar Relatório Clínico
    if (btnExportReport && detector) {
        btnExportReport.addEventListener("click", () => {
            const report = detector.getClinicalReport();
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `relatorio_tea_${new Date().toISOString().slice(0,10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            alert("Relatório clínico exportado com sucesso! Arquivo pronto para envio ao psicólogo.");
        });
    }

    // Salvar Configurações
    if (btnSaveSettings && detector) {
        btnSaveSettings.addEventListener("click", () => {
            const name = document.getElementById("cfgStudentName").value || "Aluno";
            const audSens = parseFloat(document.getElementById("cfgAuditorySens").value) || 1.0;
            const noiseThresh = parseFloat(document.getElementById("cfgNoiseThreshold").value) || 78.0;

            detector.setProfile({
                studentName: name,
                auditorySensitivity: audSens,
                noiseThresholdCriticalDb: noiseThresh
            });
            alert("Configurações acústicas e sensoriais salvas com sucesso!");
        });
    }

    initMediaPipeModels();
});
