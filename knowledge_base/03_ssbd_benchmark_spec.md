# 03. O Dataset SSBD / SSBD+ & Modelos de Visão Computacional
> Especificação do Self-Stimulatory Behavior Dataset e técnicas estado da arte em Deep Learning para classificação de estereotipias motoras.

---

## 1. O que é o SSBD (*Self-Stimulatory Behavior Dataset*)?

O **SSBD** (proposto originalmente por *Rajagopalan et al., 2013* e expandido para **SSBD+ em 2023**) é o banco de dados de referência internacional padrão-ouro na literatura científica para o estudo computacional de comportamentos autoestimulatórios no autismo.

### Principais Características do Dataset:
* **Classes Anotadas:** 
  1. *Arm Flapping* (Bater/agitar braços e mãos)
  2. *Head Banging* (Bater a cabeça em superfícies ou com as mãos)
  3. *Spinning* (Girar o próprio corpo em torno do eixo vertical)
* **Ambientes Reais:** Vídeos gravados em ambientes ecológicos não controlados (casas, salas de aula, clínicas), refletindo as condições reais de luz e movimento.

---

## 2. Arquiteturas Estado da Arte Utilizadas na Literatura

Na literatura recente de Visão Computacional para TEA (2023–2026), os modelos mais eficientes utilizam duas abordagens complementares:

### A. Redes de Convolução em Grafo Espaço-Temporal (ST-GCN)
* **Princípio:** Em vez de processar os pixels brutos (pesados e sensíveis a mudanças de iluminação), o vídeo é convertido primeiro em uma sequência temporal de grafos de esqueleto ($G = (V, E)$ com 33 nós do corpo).
* **Vantagens:** Extremamente leve, invariante a roupas/cenário e preserva 100% da privacidade da criança.
* **Acurácia na Literatura:** Atinge **~81% a 87% de F1-Score** na classificação das classes do SSBD+.

### B. Análise Cinemática por Inversão de Frequência (Edge Real-Time)
* **Princípio:** Para execução em celulares sem GPU dedicada, o cálculo da transformada de Fourier ou contagem de cruzamento por zero na aceleração angular dos punhos fornece resposta instantânea em < 10ms.
