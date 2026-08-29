"""
TEA Monitor - Backend Server & HTTPS Tunnel Manager
Desenvolvido para Tecnologias na Educação - Monitoramento de Estereotipias e Sobrecarga Sensorial (TEA)
"""

import os
import sys
import time
import json
import re
import argparse
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

event_history = []
MAX_EVENTS = 100

CLOUDFLARED_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe")
TUNNEL_URL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunnel_url.txt")

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
            "severity": data.get("severity", "medium"),
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

def monitor_cloudflared_output(process):
    for line in iter(process.stderr.readline, ""):
        if not line:
            break
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            with open(TUNNEL_URL_FILE, "w", encoding="utf-8") as f:
                f.write(url)
            print("\n" + "=" * 65)
            print("  >>> LINK PUBLICO HTTPS GERADO COM SUCESSO:")
            print(f"      {url}")
            print("=" * 65)
            print("  [!] Abra o link acima no Safari ou Chrome do seu celular!\n", flush=True)

def start_cloudflare_tunnel(port):
    if not os.path.exists(CLOUDFLARED_EXE):
        print("[!] cloudflared.exe nao encontrado localmente.")
        return None

    cmd = [CLOUDFLARED_EXE, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    t = threading.Thread(target=monitor_cloudflared_output, args=(process,), daemon=True)
    t.start()
    return process

def main():
    parser = argparse.ArgumentParser(description="TEA Monitor Web Server")
    parser.add_argument("--port", type=int, default=5000, help="Porta local (padrao: 5000)")
    parser.add_argument("--tunnel", type=str, choices=["cloudflare", "none"], default="cloudflare",
                        help="Gerar tunel HTTPS com Cloudflare (padrao: cloudflare)")
    args = parser.parse_args()

    port = args.port

    print("=" * 65)
    print("  [*] TEA MONITOR - VISAO COMPUTACIONAL NA EDUCACAO")
    print("  Monitoramento de Estereotipias e Sobrecarga Sensorial (TEA)")
    print("=" * 65)
    print(f"\n[1/2] Iniciando Servidor Web na porta {port}...")
    print(f"  * Acesso Local no PC: http://127.0.0.1:{port}")

    if args.tunnel == "cloudflare":
        print("\n[2/2] Iniciando tunel seguro HTTPS da Cloudflare...")
        start_cloudflare_tunnel(port)

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
