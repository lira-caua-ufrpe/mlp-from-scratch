/**
 * TEA Advanced Clinical & Behavioral Engine
 * Multimodal: Visão Espaço-Temporal (Pose + Face) + Análise Acústica Ambiental (dB)
 */

class StimmingDetector {
    constructor() {
        this.historySize = 60; // ~2 segundos a 30 FPS
        this.poseHistory = [];
        this.faceHistory = [];
        this.timelineHistory = [];

        // Estados Temporais
        this.earCoverStartTime = null;
        this.faceHideStartTime = null;
        this.freezeStartTime = null;
        this.sessionStartTime = Date.now();
        this.lastEventTimes = {};
        this.cooldownMs = 3500;

        // Histórico Clínico & Contadores
        this.clinicalIncidentLog = [];
        this.behaviorCounts = {
            hand_flapping: 0,
            body_rocking: 0,
            sensory_auditory: 0,
            sensory_visual: 0,
            head_nodding: 0,
            shutdown_freeze: 0
        };

        // Perfil de Sensibilidade
        this.profile = {
            auditorySensitivity: 1.0,
            motorSensitivity: 1.0,
            studentName: "Aluno Monitorado",
            classroomNoiseLevel: "Normal",
            noiseThresholdCriticalDb: 78 // dB a partir do qual o ruído vira gatilho crítico
        };
    }

    setProfile(newProfile) {
        this.profile = { ...this.profile, ...newProfile };
    }

    /**
     * Processa o frame com fusão multimodal (Pose + Face + Decibéis)
     */
    processFrame(poseLandmarks, faceLandmarks, ambientDb = 55) {
        const timestamp = performance.now();

        if (!poseLandmarks) {
            return this.getDefaultResult(ambientDb);
        }

        const frameData = {
            timestamp,
            pose: poseLandmarks,
            face: faceLandmarks,
            db: ambientDb
        };

        this.poseHistory.push(frameData);
        if (this.poseHistory.length > this.historySize) {
            this.poseHistory.shift();
        }

        // 1. Normalização Biomecânica (Distância Biacromial / Ombros)
        const leftShoulder = poseLandmarks[11];
        const rightShoulder = poseLandmarks[12];
        const shoulderWidth = Math.max(0.08, this.distance(leftShoulder, rightShoulder));

        // 2. Análise Comportamental Multivariada
        const flapping = this.detectHandFlapping(shoulderWidth);
        const rocking = this.detectBodyRocking(shoulderWidth);
        const auditoryDefense = this.detectAuditoryDefense(poseLandmarks, shoulderWidth, timestamp, ambientDb);
        const visualDefense = this.detectVisualDefense(poseLandmarks, faceLandmarks, shoulderWidth, timestamp);
        const headNodding = this.detectHeadNodding(poseLandmarks, shoulderWidth);
        const freezeShutdown = this.detectFreezeShutdown(poseLandmarks, shoulderWidth, timestamp);
        const gazeAttention = this.detectGazeAndAttention(faceLandmarks);

        // 3. Síntese do Nível de Sobrecarga e Estágio Clínico
        const clinicalState = this.synthesizeClinicalCascade({
            flapping,
            rocking,
            auditoryDefense,
            visualDefense,
            headNodding,
            freezeShutdown,
            gazeAttention,
            ambientDb,
            timestamp
        });

        // 4. Atualiza a timeline contínua
        this.recordTimelinePoint(clinicalState);

        return clinicalState;
    }

    /**
     * 1. Hand Flapping & Finger Motion
     */
    detectHandFlapping(scale) {
        if (this.poseHistory.length < 15) return { score: 0, hz: 0, detected: false, intensity: 'none' };

        const lWristY = [], rWristY = [], lWristX = [], rWristX = [];
        for (const f of this.poseHistory) {
            const lw = f.pose[15], rw = f.pose[16];
            if (lw && rw) {
                lWristY.push(lw.y / scale);
                rWristY.push(rw.y / scale);
                lWristX.push(lw.x / scale);
                rWristX.push(rw.x / scale);
            }
        }

        const lFlap = this.calculateOscillation(lWristY, lWristX);
        const rFlap = this.calculateOscillation(rWristY, rWristX);
        const maxHz = Math.max(lFlap.hz, rFlap.hz);
        const maxEnergy = Math.max(lFlap.energy, rFlap.energy) * this.profile.motorSensitivity;

        const isFlapRange = maxHz >= 2.0 && maxHz <= 6.5;
        const hasEnergy = maxEnergy > 0.07;

        let score = 0;
        if (isFlapRange && hasEnergy) {
            score = Math.min(100, Math.round((maxEnergy / 0.22) * 65 + (maxHz / 5.0) * 35));
        } else if (hasEnergy) {
            score = Math.min(45, Math.round(maxEnergy * 180));
        }

        return {
            score,
            hz: parseFloat(maxHz.toFixed(1)),
            detected: score >= 60,
            intensity: score >= 80 ? 'Alta' : score >= 60 ? 'Média' : 'Baixa'
        };
    }

    /**
     * 2. Body Rocking (Balanço de Tronco)
     */
    detectBodyRocking(scale) {
        if (this.poseHistory.length < 25) return { score: 0, hz: 0, detected: false, axis: 'none' };

        const centerXs = [], centerYs = [];
        for (const f of this.poseHistory) {
            const ls = f.pose[11], rs = f.pose[12], nose = f.pose[0];
            if (ls && rs && nose) {
                const midX = (ls.x + rs.x) / 2.0;
                const midY = (ls.y + rs.y) / 2.0;
                centerXs.push(midX / scale);
                centerYs.push(midY / scale);
            }
        }

        const oscX = this.calculateOscillation(centerXs);
        const oscY = this.calculateOscillation(centerYs);
        const maxOsc = oscX.energy > oscY.energy ? oscX : oscY;
        const axis = oscX.energy > oscY.energy ? 'Lateral' : 'Anteroposterior';

        const isRockingHz = maxOsc.hz >= 0.7 && maxOsc.hz <= 2.2;
        const hasEnergy = maxOsc.energy > 0.03 * this.profile.motorSensitivity;

        let score = 0;
        if (isRockingHz && hasEnergy) {
            score = Math.min(100, Math.round((maxOsc.energy / 0.09) * 75 + (maxOsc.hz / 2.0) * 25));
        }

        return {
            score,
            hz: parseFloat(maxOsc.hz.toFixed(1)),
            axis,
            detected: score >= 60
        };
    }

    /**
     * 3. Defesa Auditiva com Fusão Acústica (Pose + Ruído em dB)
     */
    detectAuditoryDefense(landmarks, scale, timestamp, ambientDb) {
        const lw = landmarks[15], rw = landmarks[16];
        const le = landmarks[7], re = landmarks[8];
        if (!lw || !rw || !le || !re) return { score: 0, detected: false, durationMs: 0, isNoiseTriggered: false };

        const dL = this.distance(lw, le) / scale;
        const dR = this.distance(rw, re) / scale;

        const isCovering = (dL < 0.42 && dR < 0.42) || (dL < 0.32) || (dR < 0.32);
        const isLoudNoise = ambientDb >= this.profile.noiseThresholdCriticalDb;

        if (isCovering) {
            if (!this.earCoverStartTime) this.earCoverStartTime = timestamp;
            const durationMs = timestamp - this.earCoverStartTime;
            
            // Se o ambiente estiver barulhento, a confiança e a gravidade aumentam rapidamente
            const noiseBoost = isLoudNoise ? 25 : 0;
            const score = Math.min(100, Math.round(55 + Math.min(45, durationMs / 15) * this.profile.auditorySensitivity + noiseBoost));

            return {
                score,
                detected: score >= 65,
                durationMs: Math.round(durationMs),
                type: (dL < 0.42 && dR < 0.42) ? 'Bilateral' : 'Unilateral',
                isNoiseTriggered: isLoudNoise,
                db: Math.round(ambientDb)
            };
        } else {
            this.earCoverStartTime = null;
            return { score: 0, detected: false, durationMs: 0, type: 'Nenhum', isNoiseTriggered: false, db: Math.round(ambientDb) };
        }
    }

    /**
     * 4. Defesa Visual / Esconder o Rosto
     */
    detectVisualDefense(poseLandmarks, faceLandmarks, scale, timestamp) {
        const lw = poseLandmarks[15], rw = poseLandmarks[16];
        const nose = poseLandmarks[0];
        if (!lw || !rw || !nose) return { score: 0, detected: false };

        const dL = this.distance(lw, nose) / scale;
        const dR = this.distance(rw, nose) / scale;

        const isHidingFace = (dL < 0.38 && dR < 0.38);

        if (isHidingFace) {
            if (!this.faceHideStartTime) this.faceHideStartTime = timestamp;
            const durationMs = timestamp - this.faceHideStartTime;
            const score = Math.min(100, Math.round(50 + Math.min(50, durationMs / 20)));
            return {
                score,
                detected: score >= 65,
                durationMs: Math.round(durationMs)
            };
        } else {
            this.faceHideStartTime = null;
            return { score: 0, detected: false, durationMs: 0 };
        }
    }

    /**
     * 5. Head Nodding
     */
    detectHeadNodding(poseLandmarks, scale) {
        if (this.poseHistory.length < 20) return { score: 0, hz: 0, detected: false };

        const noseYs = [];
        for (const f of this.poseHistory) {
            if (f.pose[0]) noseYs.push(f.pose[0].y / scale);
        }

        const osc = this.calculateOscillation(noseYs);
        const isNodding = osc.hz >= 1.2 && osc.hz <= 3.5 && osc.energy > 0.035;

        let score = 0;
        if (isNodding) {
            score = Math.min(100, Math.round(osc.energy * 250));
        }

        return {
            score,
            hz: parseFloat(osc.hz.toFixed(1)),
            detected: score >= 60
        };
    }

    /**
     * 6. Freeze / Shutdown
     */
    detectFreezeShutdown(poseLandmarks, scale, timestamp) {
        if (this.poseHistory.length < 35) return { score: 0, detected: false };

        let totalMotion = 0;
        for (let i = 1; i < this.poseHistory.length; i++) {
            const p1 = this.poseHistory[i - 1].pose[0];
            const p2 = this.poseHistory[i].pose[0];
            if (p1 && p2) totalMotion += this.distance(p1, p2) / scale;
        }

        const avgMotion = totalMotion / this.poseHistory.length;
        const isRigid = avgMotion < 0.003;

        if (isRigid) {
            if (!this.freezeStartTime) this.freezeStartTime = timestamp;
            const duration = timestamp - this.freezeStartTime;
            if (duration > 3000) {
                return {
                    score: Math.min(100, Math.round(60 + (duration / 100))),
                    detected: true,
                    durationSeconds: (duration / 1000).toFixed(1)
                };
            }
        } else {
            this.freezeStartTime = null;
        }

        return { score: 0, detected: false, durationSeconds: 0 };
    }

    /**
     * 7. Atenção e Contato Visual
     */
    detectGazeAndAttention(faceLandmarks) {
        if (!faceLandmarks || faceLandmarks.length < 10) {
            return { focusScore: 50, status: "Aguardando Rosto" };
        }

        const nose = faceLandmarks[1];
        const leftCheek = faceLandmarks[234];
        const rightCheek = faceLandmarks[454];

        if (!nose || !leftCheek || !rightCheek) return { focusScore: 50, status: "Face Parcial" };

        const dL = Math.abs(nose.x - leftCheek.x);
        const dR = Math.abs(nose.x - rightCheek.x);
        const asymmetry = Math.abs(dL - dR) / (dL + dR);

        const focusScore = Math.max(0, Math.min(100, Math.round((1 - asymmetry * 2.5) * 100)));
        return {
            focusScore,
            status: focusScore > 65 ? "Atenção Frontal" : "Atenção Desviada"
        };
    }

    /**
     * Síntese Clínica de 4 Fases com Gatilho Acústico
     */
    synthesizeClinicalCascade(data) {
        const { flapping, rocking, auditoryDefense, visualDefense, headNodding, freezeShutdown, gazeAttention, ambientDb, timestamp } = data;

        let stage = "CALM";
        let stageName = "🟢 Estágio 1: Calmo & Regulado";
        let title = "Comportamento Estável";
        let desc = `Ambiente em ${Math.round(ambientDb)} dB. Aluno com autorregulação basal.`;
        let severity = "normal";
        let recommendation = "Manter o ritmo habitual da aula e incentivar a participação.";
        let eventType = null;

        // Regras de Transição Clínica
        if (auditoryDefense.detected) {
            stage = "SENSORY_OVERLOAD";
            stageName = "🔴 Estágio 4: Sobrecarga Sensorial Acústica";
            title = auditoryDefense.isNoiseTriggered ? `⚠️ Sobrecarga Acústica (Ruído: ${Math.round(ambientDb)} dB)` : "⚠️ Defesa Sensorial Auditiva";
            desc = `Aluno cobrindo os ouvidos (${auditoryDefense.type}) há ${auditoryDefense.durationMs}ms com ruído de ${Math.round(ambientDb)} dB.`;
            severity = "critical";
            recommendation = `🚨 AÇÃO IMEDIATA: Ruído da sala atingiu ${Math.round(ambientDb)} dB. Oferecer abafador de ruído acústico imediatamente ou permitir ida ao cantinho sensorial.`;
            eventType = "SENSORY_AUDITORY";
        } else if (visualDefense.detected) {
            stage = "SENSORY_OVERLOAD";
            stageName = "🔴 Estágio 4: Sobrecarga Visual / Emocional";
            title = "⚠️ Defesa Visual / Rosto Coberto";
            desc = `Aluno protegendo os olhos/face há ${visualDefense.durationMs}ms.`;
            severity = "critical";
            recommendation = "🚨 AÇÃO IMEDIATA: Reduzir luminosidade/telas e fornecer suporte proprioceptivo suave.";
            eventType = "SENSORY_VISUAL";
        } else if (freezeShutdown.detected) {
            stage = "SHUTDOWN";
            stageName = "🟣 Estágio 4: Resposta de Congelamento (Shutdown)";
            title = "⚠️ Imobilidade / Sobrecarga Prolongada";
            desc = `Postura rígida sustentada por ${freezeShutdown.durationSeconds}s.`;
            severity = "critical";
            recommendation = "🚨 AÇÃO IMEDIATA: Não pressionar verbalmente. Ficar por perto, transmitir segurança e dar tempo de reprocessamento.";
            eventType = "SHUTDOWN_FREEZE";
        } else if (flapping.detected && rocking.detected) {
            stage = "ESCALATION";
            stageName = "🟠 Estágio 3: Agitação Comportamental em Cascata";
            title = "⚡ Flapping + Balanço Simultâneos";
            desc = "Multi-estereotipia ativa indicando alta demanda sensorial.";
            severity = "alert";
            recommendation = "⚠️ ATENÇÃO: Faça uma pausa estruturada de 3 minutos ou proponha uma atividade motora reguladora.";
            eventType = "MULTI_STIMMING";
        } else if (flapping.detected) {
            stage = "ESCALATION";
            stageName = "🟡 Estágio 2: Autorregulação Ativa (Flapping)";
            title = "⚡ Estereotipia Motora: Flapping";
            desc = `Movimento rítmico de punhos a ${flapping.hz} Hz (Intensidade ${flapping.intensity}).`;
            severity = "warning";
            recommendation = "💡 Observar sem interrupção abrupta (o flapping é um mecanismo natural de regulação). Identificar o que iniciou o movimento.";
            eventType = "HAND_FLAPPING";
        } else if (rocking.detected) {
            stage = "ESCALATION";
            stageName = "🟡 Estágio 2: Autorregulação Vestibular (Rocking)";
            title = "🔄 Estereotipia: Balanço de Tronco";
            desc = `Balanço ${rocking.axis} a ${rocking.hz} Hz.`;
            severity = "warning";
            recommendation = "💡 Permitir movimentação corporal. Pode indicar busca por estímulo vestibular para manter o foco.";
            eventType = "BODY_ROCKING";
        } else if (headNodding.detected) {
            stage = "RUMBLE";
            stageName = "🟡 Estágio 2: Inquietação / Head Nodding";
            title = "🔄 Movimento Pendular de Cabeça";
            desc = `Oscilação repetitiva de cabeça a ${headNodding.hz} Hz.`;
            severity = "warning";
            recommendation = "💡 Verificar se há desconforto postural ou fadiga visual.";
            eventType = "HEAD_NODDING";
        } else if (ambientDb >= this.profile.noiseThresholdCriticalDb) {
            stage = "RUMBLE";
            stageName = "🟠 Alerta Ambiental: Ruído Elevado";
            title = `🔊 Ruído Excessivo na Sala (${Math.round(ambientDb)} dB)`;
            desc = "Ambiente acústico ultrapassou o limiar de conforto sensorial.";
            severity = "warning";
            recommendation = "💡 O volume da sala está muito alto. Risco iminente de sobrecarga para alunos com hipersensibilidade auditiva.";
        }

        const shouldTriggerEvent = eventType && this.canEmitEvent(eventType, timestamp);
        if (shouldTriggerEvent) {
            this.recordClinicalIncident({
                type: eventType,
                title,
                stage,
                stageName,
                recommendation,
                ambientDb: Math.round(ambientDb),
                timestamp: new Date().toLocaleTimeString(),
                confidence: Math.max(flapping.score, auditoryDefense.score, rocking.score, visualDefense.score) / 100.0
            });
        }

        const stressScore = Math.min(100, Math.round(
            flapping.score * 0.35 +
            auditoryDefense.score * 0.45 +
            rocking.score * 0.25 +
            visualDefense.score * 0.35 +
            freezeShutdown.score * 0.40
        ));

        return {
            stage,
            stageName,
            title,
            desc,
            severity,
            recommendation,
            stressScore,
            ambientDb: Math.round(ambientDb),
            flapping,
            rocking,
            auditoryDefense,
            visualDefense,
            headNodding,
            freezeShutdown,
            gazeAttention,
            shouldTriggerEvent,
            eventPayload: shouldTriggerEvent ? {
                type: eventType,
                label: title,
                stage,
                recommendation,
                confidence: Math.max(flapping.score, auditoryDefense.score, rocking.score) / 100.0,
                severity,
                metrics: {
                    stressScore,
                    ambientDb: Math.round(ambientDb),
                    flappingHz: flapping.hz,
                    rockingHz: rocking.hz
                }
            } : null
        };
    }

    recordClinicalIncident(incident) {
        this.clinicalIncidentLog.unshift(incident);
        if (this.clinicalIncidentLog.length > 50) this.clinicalIncidentLog.pop();

        if (incident.type === 'HAND_FLAPPING') this.behaviorCounts.hand_flapping++;
        else if (incident.type === 'BODY_ROCKING') this.behaviorCounts.body_rocking++;
        else if (incident.type === 'SENSORY_AUDITORY') this.behaviorCounts.sensory_auditory++;
        else if (incident.type === 'SENSORY_VISUAL') this.behaviorCounts.sensory_visual++;
        else if (incident.type === 'HEAD_NODDING') this.behaviorCounts.head_nodding++;
        else if (incident.type === 'SHUTDOWN_FREEZE') this.behaviorCounts.shutdown_freeze++;
    }

    recordTimelinePoint(state) {
        const now = performance.now();
        if (!this.lastTimelineRecord || now - this.lastTimelineRecord >= 800) {
            this.timelineHistory.push({
                time: new Date().toLocaleTimeString(),
                stress: state.stressScore,
                db: state.ambientDb,
                stage: state.stage
            });
            if (this.timelineHistory.length > 40) this.timelineHistory.shift();
            this.lastTimelineRecord = now;
        }
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

    calculateOscillation(seriesY, seriesX = null) {
        if (seriesY.length < 8) return { hz: 0, energy: 0 };
        let reversals = 0;
        let totalDisp = 0;

        for (let i = 1; i < seriesY.length - 1; i++) {
            const diff1 = seriesY[i] - seriesY[i - 1];
            const diff2 = seriesY[i + 1] - seriesY[i];
            if (diff1 * diff2 < 0) reversals++;
            totalDisp += Math.abs(diff1);
        }

        const duration = (this.historySize / 30.0);
        const hz = (reversals / 2.0) / duration;
        const energy = totalDisp / seriesY.length;
        return { hz, energy };
    }

    getClinicalReport() {
        const sessionDurationMin = Math.max(1, Math.round((Date.now() - this.sessionStartTime) / 60000));
        return {
            student: this.profile.studentName,
            sessionDate: new Date().toLocaleDateString('pt-BR'),
            sessionDuration: `${sessionDurationMin} minutos`,
            totalIncidents: this.clinicalIncidentLog.length,
            counts: this.behaviorCounts,
            incidents: this.clinicalIncidentLog,
            timeline: this.timelineHistory,
            pedagogicalSummary: this.generateReportSummary()
        };
    }

    generateReportSummary() {
        const total = this.clinicalIncidentLog.length;
        if (total === 0) return "Sessão estável sem registros de sobrecarga ou estereotipias agudas.";
        const dominant = Object.entries(this.behaviorCounts).sort((a, b) => b[1] - a[1])[0];
        return `Durante a sessão de ${total} registros, o padrão mais frequente foi '${dominant[0]}' (${dominant[1]} ocorrências). Recomendado repassar esses horários para o psicólogo correlacionar com as atividades pedagógicas.`;
    }

    getDefaultResult(ambientDb = 50) {
        return {
            stage: 'CALM',
            stageName: 'Buscando Aluno...',
            title: 'Enquadrando Câmera',
            desc: `Ambiente em ${Math.round(ambientDb)} dB. Posicione o dispositivo para capturar tronco e rosto.`,
            severity: 'normal',
            recommendation: 'Aguardando detecção de pose.',
            stressScore: 0,
            ambientDb: Math.round(ambientDb),
            flapping: { score: 0, hz: 0, detected: false },
            rocking: { score: 0, hz: 0, detected: false },
            auditoryDefense: { score: 0, detected: false, durationMs: 0 },
            visualDefense: { score: 0, detected: false },
            headNodding: { score: 0, detected: false },
            freezeShutdown: { score: 0, detected: false },
            gazeAttention: { focusScore: 50, status: "Aguardando" },
            shouldTriggerEvent: false
        };
    }
}

window.StimmingDetector = StimmingDetector;
