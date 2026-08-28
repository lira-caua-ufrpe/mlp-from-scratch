/**
 * TEA Stimming & Sensory Overload Detector
 * Motor de análise biomecânica e visão computacional em tempo real
 */

class StimmingDetector {
    constructor() {
        this.historySize = 45; // ~1.5 segundos a 30 FPS
        this.poseHistory = [];
        this.faceHistory = [];

        // Estados e Temporizadores de Sustentação
        this.earCoverStartTime = null;
        this.headHoldStartTime = null;
        this.lastEventTimes = {};
        this.cooldownMs = 4000; // Tempo mínimo entre notificações repetidas do mesmo tipo
    }

    /**
     * Adiciona um novo frame de pose e processa as métricas
     */
    processFrame(poseLandmarks, faceLandmarks) {
        if (!poseLandmarks) return this.getDefaultResult();

        const timestamp = performance.now();
        const frameData = {
            timestamp,
            landmarks: poseLandmarks
        };

        this.poseHistory.push(frameData);
        if (this.poseHistory.length > this.historySize) {
            this.poseHistory.shift();
        }

        // 1. Escala de normalização (distância entre os ombros)
        const leftShoulder = poseLandmarks[11];
        const rightShoulder = poseLandmarks[12];
        const shoulderWidth = Math.max(0.1, this.distance(leftShoulder, rightShoulder));

        // 2. Análise de Hand Flapping
        const flapping = this.detectHandFlapping(shoulderWidth);

        // 3. Análise de Sobrecarga Sensorial (Mãos nos Ouvidos)
        const sensoryCovering = this.detectSensoryCovering(poseLandmarks, shoulderWidth, timestamp);

        // 4. Análise de Balanço de Tronco (Body Rocking)
        const rocking = this.detectBodyRocking(shoulderWidth);

        // 5. Determinação do Estado Global e Severidade
        return this.synthesizeState({
            flapping,
            sensoryCovering,
            rocking,
            timestamp
        });
    }

    /**
     * Detecta oscilação rápida e repetitiva das mãos (Hand Flapping)
     */
    detectHandFlapping(scale) {
        if (this.poseHistory.length < 15) return { score: 0, hz: 0, detected: false };

        const leftWristY = [];
        const rightWristY = [];
        const leftWristX = [];
        const rightWristX = [];

        for (const frame of this.poseHistory) {
            const lWrist = frame.landmarks[15];
            const rWrist = frame.landmarks[16];
            if (lWrist && rWrist) {
                leftWristY.push(lWrist.y / scale);
                rightWristY.push(rWrist.y / scale);
                leftWristX.push(lWrist.x / scale);
                rightWristX.push(rWrist.x / scale);
            }
        }

        const leftFlap = this.calculateOscillation(leftWristY, leftWristX);
        const rightFlap = this.calculateOscillation(rightWristY, rightWristX);

        const maxHz = Math.max(leftFlap.hz, rightFlap.hz);
        const maxEnergy = Math.max(leftFlap.energy, rightFlap.energy);

        // Flapping típico ocorre entre 2.0 e 6.0 Hz com alta variância
        const isFlappingHz = maxHz >= 2.0 && maxHz <= 7.0;
        const hasEnergy = maxEnergy > 0.08;

        let score = 0;
        if (isFlappingHz && hasEnergy) {
            score = Math.min(100, Math.round((maxEnergy / 0.25) * 70 + (maxHz / 5.0) * 30));
        } else if (hasEnergy) {
            score = Math.min(50, Math.round(maxEnergy * 200));
        }

        return {
            score: Math.min(100, score),
            hz: parseFloat(maxHz.toFixed(1)),
            detected: score >= 60
        };
    }

    /**
     * Detecta mãos cobrindo os ouvidos ou segurando a cabeça (Sinal de Sobrecarga Sensorial)
     */
    detectSensoryCovering(landmarks, scale, timestamp) {
        const leftWrist = landmarks[15];
        const rightWrist = landmarks[16];
        const leftEar = landmarks[7];
        const rightEar = landmarks[8];
        const nose = landmarks[0];

        if (!leftWrist || !rightWrist || !leftEar || !rightEar) {
            return { score: 0, status: 'Normal', type: 'NONE', detected: false };
        }

        // Distâncias normalizadas
        const distLeftToLeftEar = this.distance(leftWrist, leftEar) / scale;
        const distRightToRightEar = this.distance(rightWrist, rightEar) / scale;
        const distLeftToHead = this.distance(leftWrist, nose) / scale;
        const distRightToHead = this.distance(rightWrist, nose) / scale;

        // Limiares de proximidade
        const isCoveringEars = (distLeftToLeftEar < 0.45 && distRightToRightEar < 0.45) ||
                              (distLeftToLeftEar < 0.35) || (distRightToRightEar < 0.35);

        const isHoldingHead = (distLeftToHead < 0.5 && distRightToHead < 0.5);

        let type = 'NONE';
        let score = 0;

        if (isCoveringEars) {
            type = 'EARS';
            if (!this.earCoverStartTime) this.earCoverStartTime = timestamp;
            const duration = timestamp - this.earCoverStartTime;
            score = Math.min(100, Math.round(50 + Math.min(50, duration / 20))); // Sobe com duração sustentada
        } else {
            this.earCoverStartTime = null;
        }

        if (isHoldingHead && !isCoveringEars) {
            type = 'HEAD';
            if (!this.headHoldStartTime) this.headHoldStartTime = timestamp;
            const duration = timestamp - this.headHoldStartTime;
            score = Math.min(100, Math.round(40 + Math.min(50, duration / 25)));
        } else if (!isCoveringEars) {
            this.headHoldStartTime = null;
        }

        return {
            score,
            type,
            status: score >= 65 ? (type === 'EARS' ? 'Mãos nos Ouvidos' : 'Mãos na Cabeça') : 'Normal',
            detected: score >= 65
        };
    }

    /**
     * Detecta balanço repetitivo do tronco para frente/trás ou lados (Body Rocking)
     */
    detectBodyRocking(scale) {
        if (this.poseHistory.length < 25) return { score: 0, status: 'Estável', detected: false };

        const centerXs = [];
        for (const frame of this.poseHistory) {
            const nose = frame.landmarks[0];
            const ls = frame.landmarks[11];
            const rs = frame.landmarks[12];
            if (nose && ls && rs) {
                const midX = (ls.x + rs.x) / 2.0;
                centerXs.push(midX / scale);
            }
        }

        const osc = this.calculateOscillation(centerXs);
        const isRockingHz = osc.hz >= 0.7 && osc.hz <= 2.5;
        const hasRockingAmp = osc.energy > 0.035;

        let score = 0;
        if (isRockingHz && hasRockingAmp) {
            score = Math.min(100, Math.round((osc.energy / 0.1) * 80 + (osc.hz / 2.0) * 20));
        }

        return {
            score,
            hz: parseFloat(osc.hz.toFixed(1)),
            status: score >= 60 ? 'Balanço Detectado' : 'Estável',
            detected: score >= 60
        };
    }

    /**
     * Calcula a frequência dominante e energia de oscilação através de reversões de direção
     */
    calculateOscillation(seriesY, seriesX = null) {
        if (seriesY.length < 10) return { hz: 0, energy: 0 };

        let reversals = 0;
        let totalDisp = 0;

        // Combina movimento 2D se seriesX existir
        for (let i = 1; i < seriesY.length - 1; i++) {
            const prev = seriesY[i - 1];
            const curr = seriesY[i];
            const next = seriesY[i + 1];

            const diff1 = curr - prev;
            const diff2 = next - curr;

            if (diff1 * diff2 < 0) { // Mudança de sentido
                reversals++;
            }
            totalDisp += Math.abs(diff1);
        }

        const durationSeconds = (this.historySize / 30.0);
        const hz = (reversals / 2.0) / durationSeconds;
        const energy = totalDisp / seriesY.length;

        return { hz, energy };
    }

    /**
     * Consolida e determina o status geral
     */
    synthesizeState(results) {
        const { flapping, sensoryCovering, rocking, timestamp } = results;

        let state = 'NORMAL';
        let title = 'Estado Estável';
        let desc = 'Nenhum sinal de sobrecarga sensorial ou estereotipia motora.';
        let severity = 'normal'; // normal, warning, critical
        let shouldTriggerEvent = false;

        if (sensoryCovering.detected) {
            state = 'SENSORY_OVERLOAD';
            title = '⚠️ Alerta: Sobrecarga Sensorial';
            desc = sensoryCovering.type === 'EARS'
                ? 'Aluno protegendo os ouvidos (possível sobrecarga acústica/ambiente).'
                : 'Aluno segurando a cabeça (possível estresse sensorial ou fadiga).';
            severity = 'critical';
            shouldTriggerEvent = this.canEmitEvent('SENSORY_OVERLOAD', timestamp);
        } else if (flapping.detected) {
            state = 'HAND_FLAPPING';
            title = '⚡ Estereotipia Motora: Flapping de Mãos';
            desc = `Movimento repetitivo de mãos detectado a ${flapping.hz} Hz.`;
            severity = 'warning';
            shouldTriggerEvent = this.canEmitEvent('HAND_FLAPPING', timestamp);
        } else if (rocking.detected) {
            state = 'BODY_ROCKING';
            title = '🔄 Estereotipia: Balanço de Tronco';
            desc = `Movimento oscilatório rítmico do tronco detectado a ${rocking.hz} Hz.`;
            severity = 'warning';
            shouldTriggerEvent = this.canEmitEvent('BODY_ROCKING', timestamp);
        }

        return {
            state,
            title,
            desc,
            severity,
            flapping,
            sensoryCovering,
            rocking,
            shouldTriggerEvent,
            eventPayload: shouldTriggerEvent ? {
                type: state,
                label: title,
                confidence: Math.max(flapping.score, sensoryCovering.score, rocking.score) / 100.0,
                severity,
                metrics: {
                    flappingHz: flapping.hz,
                    sensoryScore: sensoryCovering.score,
                    rockingHz: rocking.hz
                }
            } : null
        };
    }

    canEmitEvent(type, timestamp) {
        const last = this.lastEventTimes[type] || 0;
        if (timestamp - last > this.cooldownMs) {
            this.lastEventTimes[type] = timestamp;
            return true;
        }
        return false;
    }

    distance(p1, p2) {
        if (!p1 || !p2) return 999;
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dz = (p1.z || 0) - (p2.z || 0);
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    getDefaultResult() {
        return {
            state: 'SEARCHING',
            title: 'Buscando Aluno na Câmera...',
            desc: 'Posicione a câmera para enquadrar tronco e face.',
            severity: 'normal',
            flapping: { score: 0, hz: 0, detected: false },
            sensoryCovering: { score: 0, status: 'Normal', detected: false },
            rocking: { score: 0, status: 'Estável', detected: false },
            shouldTriggerEvent: false
        };
    }
}

window.StimmingDetector = StimmingDetector;
