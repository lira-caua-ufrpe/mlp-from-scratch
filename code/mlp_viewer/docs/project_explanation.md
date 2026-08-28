# MLP from Scratch - Documentação Completa para Apresentação

> **Projeto:** Implementação de Multi-Layer Perceptron (MLP) do zero usando estrutura de grafo computacional
> **Disciplina:** Tópicos Avançados em Inteligência Artificial
> **Dataset:** Heart Disease (UCI Cleveland) + Desafio Diabetes 130-US Hospitals

---

## 1. Visão Geral do Projeto

### 1.1 Objetivo
Implementar uma rede neural **Multi-Layer Perceptron (MLP)** completamente **do zero** (sem PyTorch, TensorFlow, Keras), utilizando:
- **Estrutura de grafo computacional** para representar a rede
- **Feed-forward** e **Backpropagation** manuais
- **Visualização em tempo real** do grafo e pesos durante o treinamento
- **Interface web interativa** via Streamlit

### 1.2 Diferenciais Técnicos
| Aspecto | Abordagem Comum (PyTorch/TF) | Este Projeto |
|---------|------------------------------|--------------|
| Grafo | Abstraído (autograd) | **Explícito** - nós e arestas manipulados manualmente |
| Backprop | Automático | **Implementado à mão** (chain rule) |
| Visualização | TensorBoard (pós-treino) | **Tempo real** no grafo + Streamlit interativo |
| Estrutura | Camadas sequenciais | **Grafo direcionado acíclico (DAG)** |

---

## 2. Fundamentação Teórica

### 2.1 Multi-Layer Perceptron (MLP)

Uma MLP é uma rede neural **feed-forward** composta por:
- **Camada de entrada**: Recebe features (ex: 13 features do Heart Disease)
- **Camadas ocultas**: Transformações não-lineares (ex: 64 → 32 neurônios)
- **Camada de saída**: Previsão (ex: 1 neurônio sigmoid para classificação binária)

**Fluxo de dados (Forward Pass):**
```
x → [W₁, b₁] → σ → [W₂, b₂] → σ → ... → [Wₗ, bₗ] → σ → ŷ
```
Onde:
- `W` = matriz de pesos, `b` = bias
- `σ` = função de ativação (ReLU, Sigmoid, Tanh)
- `ŷ` = predição

### 2.2 Funções de Ativação

| Função | Fórmula | Derivada | Uso |
|--------|---------|----------|-----|
| **ReLU** | `max(0, x)` | `1 if x>0 else 0` | Camadas ocultas (padrão) |
| **Sigmoid** | `1/(1+e⁻ˣ)` | `σ(x)(1-σ(x))` | Saída binária |
| **Tanh** | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | `1-tanh²(x)` | Camadas ocultas |
| **Linear** | `x` | `1` | Camada de entrada |

**Por que ReLU?** Evita *vanishing gradient*, esparsidade, computacionalmente barata.

### 2.3 Funções de Loss (Erro)

| Loss | Fórmula | Uso |
|------|---------|-----|
| **MSE** | `½(y-ŷ)²` | Regressão |
| **BCE** | `-[y log ŷ + (1-y) log(1-ŷ)]` | Classificação binária |
| **Cross-Entropy** | `-Σ yᵢ log ŷᵢ` | Multi-classe |

**BCE para Heart Disease:** Target binário (doença: 0/1) → saída sigmoid ∈ (0,1).

### 2.4 Backpropagation (Retropropagação)

**Objetivo:** Calcular `∂L/∂W` e `∂L/∂b` para atualizar pesos via **Gradient Descent**.

**Regra da Cadeia (Chain Rule):**
```
∂L/∂W = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂W
       ↑         ↑          ↑
     loss    activation   linear
```

**Algoritmo:**
1. **Forward:** Computa `ŷ` e guarda valores intermediários
2. **Loss:** Calcula erro `L(y, ŷ)`
3. **Backward:** Propaga gradientes da saída → entrada
4. **Update:** `W ← W - η·∂L/∂W`, `b ← b - η·∂L/∂b`

### 2.5 Grafo Computacional

Representa a computação como **DAG (Directed Acyclic Graph)**:
- **Nós (Nodes):** Variáveis/operações (neurônios = valores + gradientes)
- **Arestas (Edges):** Dependências/fluxo de dados (pesos = arestas)

```
Input Nodes          Hidden Nodes           Output Node
    x₁      ──w₁₁──►    h₁      ──w₁ₒ──►
    x₂      ──w₂₁──►    h₂      ──w₂ₒ──►   ŷ
    x₃      ──w₃₁──►    h₃      ──w₃ₒ──►
                    ...
```

**Vantagens:**
- Backprop = travessia reversa do grafo
- Facilita visualização/debug
- Extensível para arquiteturas complexas (skip connections, etc.)

---

## 3. Arquitetura do Código

### 3.1 Estrutura de Pastas
```
workspace/
├── src/
│   ├── mlp/
│   │   ├── graph.py      # ComputationalGraph, Node, Edge
│   │   └── mlp.py        # Classe MLP (API alto nível)
│   ├── data/
│   │   └── loader.py     # Heart Disease + Diabetes loaders
│   ├── visualization/
│   │   └── dashboard.py  # GraphVisualizer, LiveTrainingDashboard
│   └── utils/
├── scripts/
│   └── train.py          # ExperimentHarness (linha de comando)
├── app.py                # Streamlit Web App
├── docs/                 # Documentação técnica
└── tests/
```

### 3.2 Módulos Principais

#### `src/mlp/graph.py` - Coração do Sistema
```python
class Node:
    """Neurônio no grafo"""
    value: float        # Saída da ativação (forward)
    gradient: float     # ∂L/∂value (backward)
    bias: float         # Bias do neurônio
    bias_gradient: float
    activation: str     # "relu", "sigmoid", etc.

class Edge:
    """Conexão pesada entre neurônios"""
    weight: float       # Peso w
    gradient: float     # ∂L/∂weight
    source: Node        # Neurônio origem
    target: Node        # Neurônio destino

class ComputationalGraph:
    """Gerencia topologia + forward/backward"""
    def build_mlp(layer_sizes, activations)  # Constrói grafo completo
    def forward_pass(inputs)                  # Feed-forward
    def backward_pass(targets, loss_fn)       # Backpropagation
    def update_weights(learning_rate)         # SGD step
    def get_weights_snapshot()                # Para visualização
```

#### `src/mlp/mlp.py` - API de Alto Nível
```python
class MLP:
    def __init__(self, layer_sizes, activations, ...):
        self.graph = ComputationalGraph()
        self.graph.build_mlp(layer_sizes, activations)
    
    def fit(X_train, y_train, X_val, y_val, epochs, batch_size, callback):
        # Loop de treino com batches, validação, callbacks
    
    def predict(X) / predict_proba(X)
    def save_checkpoint(path) / load_checkpoint(path)
```

#### `src/data/loader.py` - Dados
```python
def load_heart_disease(download=True):
    # Baixa do UCI, limpa "?", binariza target (0/1)
    return X (n,13), y (n,)

def preprocess_data(X, y, test_size=0.2, scale=True):
    # Split estratificado + StandardScaler (média=0, std=1)
```

#### `src/visualization/dashboard.py` - Visualização
```python
class GraphVisualizer:
    """Desenha grafo com NetworkX + Matplotlib"""
    def draw(epoch, loss, show_weights=True):
        # Nós coloridos por ativação
        # Arestas: largura = |peso|, cor = sinal

class LiveTrainingDashboard:
    """Callback para mlp.fit() - atualiza gráficos em tempo real"""
    def on_epoch_end(epoch, metrics, weights_snapshot):
        # Atualiza loss/acc curves + grafo + histogramas
```

#### `app.py` - Streamlit Web App
Interface interativa com:
- Sidebar: configuração completa (arquitetura, hiperparâmetros, dataset)
- Abas: Dados → Arquitetura → Treino (AO VIVO) → Resultados → Inferência
- Callbacks customizados para atualizar `st.line_chart`, `st.pyplot` a cada época

---

## 4. Fluxo de Execução Completo

### 4.1 Via Linha de Comando (`scripts/train.py`)
```python
config = ExperimentConfig(
    layer_sizes=[13, 64, 32, 1],
    activations=["relu", "relu", "sigmoid"],
    epochs=200, batch_size=32, lr=0.01
)
harness = ExperimentHarness(config)
results = harness.run_full_experiment()
# → prepara dados → build model → train → evaluate → save → animation
```

### 4.2 Via Streamlit (`app.py`)
```
Usuário configura na sidebar
        ↓
Carrega dados (aba Dados)
        ↓
Define arquitetura (aba Arquitetura)
        ↓
Clica "Iniciar Treino" (aba Treino)
        ↓
mlp.fit() com callback Streamlit
        ↓
A cada época: atualiza gráficos + grafo no browser
        ↓
Fim: mostra métricas, matriz confusão, inferência
```

---

## 5. GitFlow - Versionamento Profissional

### 5.1 Branches
```
main (v0.1.0 tagged) ──────────────────────► Produção
       ↑
       │ merge --no-ff
       │
develop ──────────────────────────────────► Integração contínua
       ↑
       │ merge --no-ff
       │
feature/mlp-graph ───► feature/nova-ideia ──► Features isoladas
```

### 5.2 Comandos Usados
```bash
# Inicialização
git init
git config user.email/name
git add . && git commit -m "feat: initial implementation"

# Branches
git checkout -b develop
git checkout -b feature/nome-da-feature
# ... trabalho ...
git checkout develop
git merge --no-ff feature/nome-da-feature
git branch -d feature/nome-da-feature

# Release
git checkout main
git merge --no-ff develop
git tag -a v0.1.0 -m "Release v0.1.0"

# GitHub
git remote add origin https://github.com/user/repo.git
git push -u origin main develop --tags
```

### 5.3 Por que GitFlow?
- **Histórico limpo** (merge --no-ff preserva feature branches)
- **Paralelismo** (múltiplas features simultâneas)
- **Rastreabilidade** (issues → branches → commits → tags)
- **Padrão da indústria**

---

## 6. Dataset: Heart Disease (Cleveland)

### 6.1 Características
- **Fonte:** UCI Machine Learning Repository
- **Amostras:** 303 (após limpeza de NaN)
- **Features:** 13 (idade, sexo, tipo dor, pressão, colesterol, etc.)
- **Target:** Binário (0 = sem doença, 1 = com doença)
- **Split:** 80% treino / 20% teste (estratificado)

### 6.2 Features (13)
| Índice | Feature | Descrição |
|--------|---------|-----------|
| 0 | age | Idade |
| 1 | sex | Sexo (1=M, 0=F) |
| 2 | cp | Tipo dor no peito (1-4) |
| 3 | trestbps | Pressão arterial repouso |
| 4 | chol | Colesterol sérico |
| 5 | fbs | Glicemia jejum > 120 |
| 6 | restecg | ECG repouso (0-2) |
| 7 | thalach | Freq. cardíaca máx |
| 8 | exang | Angina exercício |
| 9 | oldpeak | Depressão ST |
| 10 | slope | Inclinação ST |
| 11 | ca | Vasos principais (0-3) |
| 12 | thal | Talassemia (1-3) |

### 6.3 Pré-processamento
1. **Remoção de NaN** (valores "?" no original)
2. **StandardScaler:** `z = (x - μ)/σ` → média 0, desvio 1
3. **Split estratificado:** Mantém proporção classes treino/teste

---

## 7. Desafio Extra: Diabetes 130-US Hospitals

### 7.1 Dataset
- **100.000+** internações hospitalares (1999-2008)
- **50+ features** (demografia, diagnósticos, medicamentos, etc.)
- **Target:** Readmissão < 30 dias (binário: ~11% positivos)
- **Desafio:** Class imbalance + alta dimensionalidade

### 7.2 Estratégias para Leaderboard
```python
# 1. Feature engineering
- Contagem de medicamentos
- Severidade diagnósticos (ICD-9)
- Histórico internações prévias

# 2. Class imbalance
- Class weights no loss
- Oversampling (SMOTE) / Undersampling
- Focal Loss

# 3. Arquitetura
- Mais camadas/neurônios (ex: 128, 64, 32)
- Dropout (implementar no grafo)
- Batch Normalization

# 4. Ensemble
- Treinar 5 seeds diferentes
- Média das predições (soft voting)
```

---

## 8. Conceitos-Chave para Explicar ao Professor

### 8.1 "Por que grafo computacional?"
> "Diferente de frameworks que abstraem o grafo, eu **explicitamente modelei** cada neurônio como Node e cada peso como Edge. Isso me permitiu:
> - Implementar backprop **passo a passo** entendendo a chain rule
> - Visualizar **exatamente** como gradientes fluem
> - Estender facilmente para arquiteturas não-sequenciais"

### 8.2 "Como funciona o backprop no seu código?"
> "No `ComputationalGraph.backward_pass()`:
> 1. Calculo `∂L/∂ŷ` na camada de saída (derivada do BCE/MSE)
> 2. Para cada camada reversa: multiplico pela derivada da ativação (`σ'(z)`)
> 3. Acumulo `∂L/∂W = input × δ` e `∂L/∂b = δ` nas arestas/nós
> 4. Atualizo pesos: `W -= lr × ∂L/∂W`"

### 8.3 "Por que Streamlit?"
> "O matplotlib salva frames estáticos. O Streamlit permite:
> - **Interatividade:** sliders para LR, epochs, arquitetura
> - **Feedback imediato:** gráficos atualizam a cada época
> - **Acessibilidade:** roda no browser, não precisa instalar nada extra
> - **Demonstração:** professor pode mexer nos hiperparâmetros ao vivo"

### 8.4 "O que você aprendeu implementando do zero?"
- **Derivadas na prática:** chain rule não é só fórmula, é fluxo de gradientes
- **Inicialização importa:** Xavier vs He muda convergência
- **Batch size vs LR:** trade-off ruído vs estabilidade
- **Debug visual:** ver pesos explodindo/vanishing no grafo ensina mais que logs

---

## 9. Como Rodar (Resumo para Professor)

### 9.1 Opção 1: Streamlit (Recomendado para Demo)
```bash
cd workspace
python -m streamlit run app.py
# Abre http://localhost:8501
```

### 9.2 Opção 2: Linha de Comando
```bash
cd workspace
python scripts/train.py
# Gera outputs/visualization/*.png + training_animation.gif
```

### 9.3 Opção 3: Jupyter/Notebook
```python
from src.mlp.mlp import MLP
import numpy as np
X, y = load_heart_disease()
mlp = MLP([13, 64, 32, 1], ["relu", "relu", "sigmoid"])
mlp.fit(X_train, y_train, X_val, y_val, epochs=100)
```

---

## 10. Possíveis Perguntas do Professor + Respostas

| Pergunta | Resposta Chave |
|----------|----------------|
| **Por que não usou PyTorch?** | Objetivo pedagógico: entender *como* funciona, não só *usar* |
| **Como validou se backprop está correto?** | Gradient checking numérico + XOR problem (non-linear) + convergência em dados reais |
| **Qual a complexidade?** | Forward/Backward: O(E) onde E = arestas. Memória: O(N+E) |
| **Como lida com overfitting?** | Validação split, early stopping (callback), weight decay (fácil add no update) |
| **Por que StandardScaler?** | Features em escalas diferentes (idade vs colesterol) → gradientes instáveis |
| **Como estender para CNN/RNN?** | Grafo já suporta conexões arbitrárias - só adicionar Nodes/Edges apropriados |

---

## 11. Próximos Passos / Melhorias Futuras

- [ ] **Otimizadores:** Adam, RMSprop (substituir SGD simples)
- [ ] **Regularização:** L2 weight decay, Dropout no grafo
- [ ] **Batch Norm:** Nós especiais para normalização por batch
- [ ] **Learning Rate Scheduling:** Decay, cosine annealing
- [ ] **Diabetes Leaderboard:** Feature engineering + ensemble
- [ ] **Export ONNX:** Interoperabilidade com outros frameworks
- [ ] **Deploy:** Docker + Streamlit Cloud / Hugging Face Spaces

---

## 12. Referências de Estudo

### Livros/Artigos
- **Goodfellow, Bengio, Courville** - *Deep Learning* (Cap. 6: MLP, Cap. 8: Optimization)
- **Nielsen** - *Neural Networks and Deep Learning* (Cap. 2: Backprop)
- **Colah's Blog** - "Calculus on Computational Graphs" (visualização intuitiva)

### Códigos de Referência
- **micrograd** (Karpathy) - Autograd minimal em 100 linhas
- **nnfs** (Sentdex) - Neural Networks from Scratch

### UCI Datasets
- Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- Diabetes 130-US: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

## 13. Checklist para Apresentação

- [ ] **Slide 1:** Título, nome, disciplina, dataset
- [ ] **Slide 2:** Problema (classificação binária doença cardíaca)
- [ ] **Slide 3:** Arquitetura MLP + Grafo Computacional (diagrama)
- [ ] **Slide 4:** Forward Pass (equações)
- [ ] **Slide 5:** Backpropagation (chain rule + fluxo gradientes)
- [ ] **Slide 6:** Implementação - classes Node/Edge/Graph
- [ ] **Slide 7:** Demo ao vivo no Streamlit (treino + visualização)
- [ ] **Slide 8:** Resultados Heart Disease (accuracy, matriz confusão)
- [ ] **Slide 9:** GitFlow + versionamento
- [ ] **Slide 10:** Desafio Diabetes + estratégias
- [ ] **Slide 11:** Lições aprendidas + próximas melhorias
- [ ] **Slide 12:** Referências + GitHub QR code

---

**Dica final:** Durante a demo, **mude um hiperparâmetro** (ex: LR de 0.01 → 0.1) e mostre como o loss explode. Isso prova que você **entende** a dinâmica do treinamento, não só rodou código pronto.

**Boa apresentação! 🚀**