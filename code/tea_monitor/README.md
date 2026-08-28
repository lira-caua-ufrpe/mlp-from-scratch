# 🧠 TEA Monitor — Visão Computacional na Educação Inclusiva
> Sistema web em tempo real para auxílio pedagógico no monitoramento de estereotipias motoras (*stimming*) e identificação precoce de sinais de sobrecarga sensorial em alunos com Transtorno do Espectro Autista (TEA).

---

## 🎯 Contexto e Justificativa Pedagógica

No ambiente escolar inclusivo, alunos no espectro autista frequentemente experienciam **sobrecarga sensorial** (estímulos visuais, ruídos excessivos ou mudanças bruscas de rotina) e utilizam **estereotipias motoras (*stimming*)** como mecanismos autorregulatórios de conforto ou resposta a estresse.

### Como este sistema auxilia o professor:
1. **Identificação Precoce de Sobrecarga:** Detecta quando o aluno cobre os ouvidos ou segura a cabeça repetidamente, permitindo ao professor intervir reduzindo estímulos antes que ocorra uma crise de desregulação (*meltdown*).
2. **Mapeamento de Frequência de Estereotipias:** Monitora a cadência e intensidade de movimentos como *flapping* de mãos e balanço de tronco, ajudando a identificar momentos da aula com maior demanda cognitiva ou sensorial.
3. **Privacidade e Ética:** Todo o processamento de visão computacional ocorre localmente no navegador (**Edge AI**), sem gravação ou transmissão de imagens de vídeo para servidores externos.

---

## 🚀 Como Rodar em Menos de 5 Minutos

### 1️⃣ Instalar as Dependências
Abra o terminal na pasta do projeto e instale os pacotes necessários:

```bash
cd D:\workspace\code\tea_monitor
pip install -r requirements.txt
```

---

### 2️⃣ Iniciar o Servidor e Gerar o Link Público (HTTPS)

Execute o servidor com o túnel automático da Cloudflare (não requer cadastro nem token):

```bash
python app.py
```

O terminal exibirá uma saída como esta:
```text
=================================================================
  🚀 SERVIDOR PRONTO PARA O PROFESSOR E ALUNO!
=================================================================
  • Acesso Local no PC:    http://127.0.0.1:5000

  👉 LINK PÚBLICO HTTPS PARA O CELULAR:
     https://random-subdomain.trycloudflare.com

  📲 Abra o link acima no navegador do celular (Safari/Chrome).
=================================================================
```

---

### 3️⃣ Testar no Celular com o Professor
1. **Copie o link HTTPS** gerado no terminal e envie para o WhatsApp ou abra no navegador do celular.
2. Ao abrir a página, clique no botão **▶️ Iniciar Câmera do Celular** e permita o acesso à câmera quando o navegador solicitar.
3. Aponte a câmera para você ou para o aluno:
   - **Flapping:** Balance as mãos rapidamente para ver o medidor de frequência (Hz) e o alerta amarelo.
   - **Sobrecarga Sensorial:** Coloque as duas mãos nos ouvidos ou na cabeça por 1 a 2 segundos para disparar o alerta vermelho de sobrecarga.
   - **Balanço:** Realize balanço suave do tronco para frente/lados para acionar o detector de *rocking*.

---

## 🔬 Algoritmos e Métricas Biomecânicas

| Sinal Monitorado | Marcos Anatômicos (MediaPipe) | Regra de Detecção Biomecânica |
| :--- | :--- | :--- |
| **Flapping de Mãos** | Punhos (15, 16) e Ombros (11, 12) | Oscilação rápida com reversão de aceleração entre **2.0 Hz e 6.0 Hz**. |
| **Mãos nos Ouvidos** | Punhos (15, 16) e Ouvidos (7, 8) | Proximidade euclidiana normalizada menor que limiar crítico mantida por > 400ms. |
| **Mãos na Cabeça** | Punhos (15, 16) e Nariz/Testa (0) | Proximidade da face e têmporas mantida em momentos de fadiga/tensão. |
| **Balanço de Tronco** | Centro dos Ombros e Nariz | Movimento pendular horizontal/anteroposterior rítmico entre **0.8 Hz e 2.2 Hz**. |

---

## 📁 Estrutura de Arquivos

```
tea_monitor/
├── app.py                     # Servidor Flask com túnel HTTPS integrado
├── requirements.txt           # Dependências Python
├── README.md                  # Este documento
├── tests/
│   └── test_server.py         # Testes automatizados da API
└── static/
    ├── index.html             # Interface Web responsiva para celulares
    ├── css/
    │   └── style.css          # Estilização com tema escuro e HUD
    └── js/
        ├── stimming_detector.js  # Motor matemático biomecânico de TEA
        └── mediapipe_camera.js   # Pipeline MediaPipe e captura de vídeo
```
