"""
Teste Automatizado de Extração Geométrica de Action Units (FACS) no MediaPipe FaceMesh
Valida AU04 (Franzimento), AU24 (Tensão Labial), EAR (Fechamento Ocular) e Atenção
"""
import math
import pytest

def compute_facs_metrics(landmarks):
    # Têmporas (234 e 454)
    face_width = max(0.05, abs(landmarks[234]["x"] - landmarks[454]["x"]))
    
    # 1. AU04 (Sobrancelhas 105 e 334)
    brow_dist = abs(landmarks[105]["x"] - landmarks[334]["x"]) / face_width
    au04_score = max(0, min(100, round(((0.36 - brow_dist) / 0.12) * 100)))
    
    # 2. AU24 (Lábios 0 e 17, Cantos 61 e 291)
    mouth_h = abs(landmarks[0]["y"] - landmarks[17]["y"])
    mouth_w = max(0.01, abs(landmarks[61]["x"] - landmarks[291]["x"]))
    lip_ratio = mouth_h / mouth_w
    au24_score = max(0, min(100, round(((0.22 - lip_ratio) / 0.15) * 100)))
    
    # 3. EAR (Olhos 159, 145 e 33, 133)
    eye_h = abs(landmarks[159]["y"] - landmarks[145]["y"])
    eye_w = max(0.01, abs(landmarks[33]["x"] - landmarks[133]["x"]))
    ear = eye_h / eye_w
    
    return {
        "au04": au04_score,
        "au24": au24_score,
        "ear": ear
    }

def create_mock_face(brow_distance=0.36, lip_ratio=0.22, eye_ratio=0.30):
    lm = {}
    lm[234] = {"x": 0.30, "y": 0.40} # Têmpora Esq
    lm[454] = {"x": 0.70, "y": 0.40} # Têmpora Dir (largura = 0.40)
    
    # Sobrancelhas
    center = 0.50
    half_brow = (brow_distance * 0.40) / 2.0
    lm[105] = {"x": center - half_brow, "y": 0.30}
    lm[334] = {"x": center + half_brow, "y": 0.30}
    
    # Lábios
    lm[61] = {"x": 0.44, "y": 0.65}
    lm[291] = {"x": 0.56, "y": 0.65} # mouth_w = 0.12
    h = 0.12 * lip_ratio
    lm[0] = {"x": 0.50, "y": 0.65 - h/2}
    lm[17] = {"x": 0.50, "y": 0.65 + h/2}
    
    # Olhos
    lm[33] = {"x": 0.38, "y": 0.38}
    lm[133] = {"x": 0.44, "y": 0.38} # eye_w = 0.06
    eh = 0.06 * eye_ratio
    lm[159] = {"x": 0.41, "y": 0.38 - eh/2}
    lm[145] = {"x": 0.41, "y": 0.38 + eh/2}
    
    return lm

def test_facs_relaxed_face():
    """Rosto relaxado deve ter AU04 e AU24 baixos."""
    face = create_mock_face(brow_distance=0.36, lip_ratio=0.22, eye_ratio=0.30)
    metrics = compute_facs_metrics(face)
    
    assert metrics["au04"] <= 20, f"AU04 falso positivo: {metrics['au04']}"
    assert metrics["au24"] <= 20, f"AU24 falso positivo: {metrics['au24']}"
    assert metrics["ear"] >= 0.25, f"Olhos detectados como fechados: {metrics['ear']}"

def test_facs_brow_furrow_distress():
    """Sobrancelhas franzidas devem ativar AU04 com score alto."""
    face = create_mock_face(brow_distance=0.25, lip_ratio=0.22, eye_ratio=0.30)
    metrics = compute_facs_metrics(face)
    
    assert metrics["au04"] >= 75, f"AU04 não detectou franzimento: {metrics['au04']}"

def test_facs_lip_compression_tension():
    """Lábios comprimidos/travados devem ativar AU24."""
    face = create_mock_face(brow_distance=0.36, lip_ratio=0.10, eye_ratio=0.30)
    metrics = compute_facs_metrics(face)
    
    assert metrics["au24"] >= 70, f"AU24 não detectou tensão labial: {metrics['au24']}"

def test_facs_prolonged_eye_closure():
    """Olhos fechados (EAR < 0.12) devem ser identificados para fotofobia."""
    face = create_mock_face(brow_distance=0.36, lip_ratio=0.22, eye_ratio=0.08)
    metrics = compute_facs_metrics(face)
    
    assert metrics["ear"] < 0.12, f"EAR não capturou fechamento ocular: {metrics['ear']}"
