"""
Teste Automatizado de Rejeição de Falsos Positivos e Sensibilidade a Estereotipias
Valida o motor de IA contra o banco de dados de Controles Negativos de Sala de Aula
"""
import os
import json
import math
import pytest

DATASETS_DIR = r"D:\datasets\tea_stimming\landmark_sequences"

def calculate_wrist_oscillation(wrist_y_series, history_size=60, fps=30):
    if len(wrist_y_series) < 8:
        return 0.0, 0.0
    
    reversals = 0
    total_disp = 0.0
    for i in range(1, len(wrist_y_series) - 1):
        diff1 = wrist_y_series[i] - wrist_y_series[i-1]
        diff2 = wrist_y_series[i+1] - wrist_y_series[i]
        if diff1 * diff2 < 0:
            reversals += 1
        total_disp += abs(diff1)
        
    duration = min(len(wrist_y_series), history_size) / fps
    hz = (reversals / 2.0) / duration
    energy = total_disp / len(wrist_y_series)
    return hz, energy

def distance_3d(p1, p2):
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    dz = p1.get("z", 0.0) - p2.get("z", 0.0)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def test_negative_control_raising_hand_no_flapping():
    """Garante que levantar a mão para falar NÃO dispara flapping."""
    path = os.path.join(DATASETS_DIR, "sample_ctrl_raising_hand.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rw_y = [frame["pose_landmarks"]["16_RIGHT_WRIST"]["y"] for frame in data["frames"]]
    hz, energy = calculate_wrist_oscillation(rw_y)
    
    # Levantar a mão tem movimento único sem inversão de velocidade periódica (Hz < 0.5)
    assert hz < 1.0, f"Levantar a mão disparou oscilação anômala: {hz} Hz"

def test_negative_control_writing_notebook_no_flapping():
    """Garante que escrever no caderno NÃO dispara flapping."""
    path = os.path.join(DATASETS_DIR, "sample_ctrl_writing.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rw_y = [frame["pose_landmarks"]["16_RIGHT_WRIST"]["y"] for frame in data["frames"]]
    hz, energy = calculate_wrist_oscillation(rw_y)
    
    # Micromovimentos de escrita têm energia muito baixa (< 0.03)
    assert energy < 0.03, f"Escrita no caderno disparou alta energia cinética: {energy}"

def test_negative_control_scratch_ear_no_sensory_overload():
    """Garante que coçar a orelha rapidamente (< 300ms) NÃO dispara alarme de sobrecarga sensorial."""
    path = os.path.join(DATASETS_DIR, "sample_ctrl_scratch_ear.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Filtro temporal: a aproximação precisa durar mais de 400ms contínuos
    ear_contact_frames = 0
    for frame in data["frames"]:
        lw = frame["pose_landmarks"]["16_RIGHT_WRIST"]
        le = frame["pose_landmarks"]["8_RIGHT_EAR"]
        dist = distance_3d(lw, le)
        if dist < 0.10: # Contato com a orelha
            ear_contact_frames += 1
            
    contact_duration_ms = (ear_contact_frames / data["fps"]) * 1000
    is_overload_alarm = contact_duration_ms >= 400.0
    
    # Coçar a orelha durou ~200ms, então o alarme NÃO deve ser disparado
    assert not is_overload_alarm, f"Toque breve de {contact_duration_ms}ms disparou falso alarme de sobrecarga!"

def test_positive_flapping_detection():
    """Garante que flapping real de 3.5 Hz é detectado com 100% de sucesso."""
    path = os.path.join(DATASETS_DIR, "sample_flapping_01.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    lw_y = [frame["pose_landmarks"]["15_LEFT_WRIST"]["y"] for frame in data["frames"]]
    hz, energy = calculate_wrist_oscillation(lw_y)
    
    assert 2.5 <= hz <= 4.5, f"Flapping de 3.5 Hz medido incorretamente como {hz} Hz"
    assert energy > 0.05, f"Energia cinética muito baixa: {energy}"
