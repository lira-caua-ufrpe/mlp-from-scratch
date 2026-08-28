"""
TEA Monitor - Backend Server & HTTPS Tunnel Manager
Desenvolvido para Tecnologias na Educação - Monitoramento de Estereotipias e Sobrecarga Sensorial (TEA)
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Armazenamento em memoria dos eventos detectados
event_history = []
MAX_EVENTS = 100

CLOUDFLARED_URL_WINDOWS = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CLOUDFLARED_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "TEA Monitor API",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/events", methods=["GET", "POST", "DELETE"])
def handle_events():
    global event_history
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        event = {
            "id": len(event_history) + 1,
            "type": data.get("type", "UNKNOWN"),
            "label": data.get("label", "Evento Desconhecido"),
            "confidence": round(data.get("confidence", 0.0), 2),
            "severity": data.get("severity", "medium"), # low, medium, high
            "metrics": data.get("metrics", {}),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "datetime": datetime.now().isoformat()
        }
        event_history.insert(0, event)
        if len(event_history) > MAX_EVENTS:
            event_history.pop()
        return jsonify({"success": True, "event": event}), 201

    elif request.method == "GET":
        return jsonify({
            "total": len(event_history),
            "events": event_history
        })

    elif request.method == "DELETE":
        event_history = []
        return jsonify({"success": True, "message": "Historico de eventos limpo."})

def ensure_cloudflared():
    """Baixa o executavel do cloudflared caso nao exista para tunel HTTPS gratuito e sem login."""
    if os.path.exists(CLOUDFLARED_EXE):
        return CLOUDFLARED_EXE
    
    print("\n[Tunel] Baixando utilitario oficial do Cloudflare Tunnel para HTTPS...")
    try:
        r = requests.get(CLOUDFLARED_URL_WINDOWS, stream=True, timeout=30)
        with open(CLOUDFLARED_EXE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("[Tunel] Download concluido com sucesso!")
        return CLOUDFLARED_EXE
    except Exception as e:
        print(f"[Tunel] Erro ao baixar cloudflared: {e}")
        return None

def start_cloudflare_tunnel(port):
    """Inicia um tunel HTTPS rapido da Cloudflare (zero config, sem necessidade de cadastro)."""
    exe = ensure_cloudflared()
    if not exe or not os.path.exists(exe):
        return None

    cmd = [exe, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    tunnel_url = None
    # Cloudflared imprime a URL no stderr
    for line in iter(process.stderr.readline, ""):
        if "trycloudflare.com" in line:
            import re
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                break
    return tunnel_url

def start_ngrok_tunnel(port, authtoken=None):
    """Inicia tunel usando pyngrok."""
    try:
        from pyngrok import ngrok
        if authtoken:
            ngrok.set_auth_token(authtoken)
        tunnel = ngrok.connect(port, "http")
        return tunnel.public_url
    except Exception as e:
        print(f"[Ngrok] Erro ao iniciar ngrok: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="TEA Monitor Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Porta local do servidor (padrao: 5000)")
    parser.add_argument("--tunnel", type=str, choices=["cloudflare", "ngrok", "none"], default="cloudflare",
                        help="Tipo de tunel HTTPS (cloudflare [recomendado/gratuito], ngrok ou none)")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Token do Ngrok (opcional)")
    args = parser.parse_args()

    port = args.port

    print("=" * 65)
    print("  🧠 TEA MONITOR - VISÃO COMPUTACIONAL NA EDUCAÇÃO")
    print("  Monitoramento de Estereotipias e Sobrecarga Sensorial")
    print("=" * 65)
    print(f"\n[1/2] Iniciando Servidor Web na porta local {port}...")

    public_url = None
    if args.tunnel == "cloudflare":
        print("[2/2] Gerando link publico seguro (HTTPS) via Cloudflare Tunnel...")
        public_url = start_cloudflare_tunnel(port)
    elif args.tunnel == "ngrok":
        print("[2/2] Gerando link publico via Ngrok...")
        public_url = start_ngrok_tunnel(port, args.ngrok_token)

    print("\n" + "=" * 65)
    print("  🚀 SERVIDOR PRONTO PARA O PROFESSOR E ALUNO!")
    print("=" * 65)
    print(f"  • Acesso Local no PC:    http://127.0.0.1:{port}")
    if public_url:
        print(f"\n  👉 LINK PÚBLICO HTTPS PARA O CELULAR:")
        print(f"     \033[92m{public_url}\033[0m")
        print("\n  📲 Abra o link acima no navegador do celular (Safari/Chrome).")
        print("  (A câmera do celular será acionada diretamente pelo navegador)")
    else:
        print("\n  ⚠️ Rodando apenas local. Para usar HTTPS no celular, use --tunnel cloudflare")
    print("=" * 65 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
