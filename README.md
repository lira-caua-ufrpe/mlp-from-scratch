# MLP Graph Viewer - Multi-Layer Perceptron como Grafo Explícito

Implementação de MLP (Multi-Layer Perceptron) do zero representada como **grafo explícito** de neurônios (nós) e conexões (arestas), com interface gráfica interativa PyQt6 e demo web Streamlit.

---

## 🎯 Objetivos

- **Visualizar a rede como grafo**: camadas, neurônios, conexões, pesos, bias
- **Entender conexões entre camadas adjacentes**: geração automática fully-connected
- **Modificação dinâmica da arquitetura**: add/remove camadas e neurônios em tempo real
- **Código autoexplicativo**: nomes claros sem comentários extensos
- **Step-by-step forward**: acompanhar propagação neurônio a neurônio
- **Visualização de gradientes**: cores verde/magenta nas arestas após train step

---

## 📁 Estrutura do Projeto

```
workspace/
├── Core (PyQt6 + Python puro)
│   ├── neuron.py          # Neuron (id, bias, input_sum, output, delta)
│   ├── connection.py      # Connection (source, target, weight, gradient)
│   ├── layer.py           # Layer (neurons[], add/remove neuron)
│   ├── network.py         # Network (forward, backprop, treino, arquitetura dinâmica)
│   ├── graph_utils.py     # Posições, cores, espessuras para visualização
│   ├── qt_viewer.py       # PyQt6 GUI completa (grafo + controles + treino)
│   ├── loader.py          # CSV Loader (última coluna = target, normalize/standardize)
│   ├── main.py            # Exemplo mínimo não-gráfico
│   ├── weights.json       # Persistência de pesos
│   └── rsc/
│       ├── heart.csv      # Heart Disease (UCI Cleveland, 1025 samples, 13 features)
│       └── diabetes.csv   # Diabetes 130-US (101k samples, 39 features, processado)
│
├── Extras (Identidade Própria)
│   ├── app.py             # Streamlit Web App (demo web interativa)
│   ├── src/               # Versão NumPy/vetorizada para comparação
│   ├── scripts/train.py   # ExperimentHarness (configuração via dataclass)
│   ├── docs/              # Documentação completa (arquitetura, API, guia)
│   ├── tests/             # Testes unitários
│   ├── .github/workflows/ # CI/CD (GitHub Actions)
│   ├── requirements.txt   # Dependências
│   └── README.md          # Este arquivo
```

---

## 🚀 Como Executar

### 1. Interface Gráfica PyQt6 (Principal)
```bash
pip install -r requirements.txt
python qt_viewer.py
```

**Funcionalidades da GUI:**
- **Visualização**: Grafo com zoom (scroll), pan (arraste), highlight (clique)
- **Arquitetura**: Adicionar/remover camadas ocultas e neurônios
- **Entrada**: Valores manuais ou navegação pelo dataset
- **Forward**: Completo ou passo-a-passo (neurônio a neurônio)
- **Auto-play**: Animação automática do forward
- **Treino**: Train Step (1 sample), Train Época (dataset todo), Learning Rate
- **Gradientes**: Flash verde/magenta nas arestas por 1.2s após train step
- **Loss History**: Janela com últimos 200 valores de loss

### 2. Exemplo Mínimo (Terminal)
```bash
python main.py
```
Saída esperada:
```
dados carregados: features=13 linhas=1025
rede criada: layers:13, 8, 1 | connections:112 | activation: sigmoid
época 1: loss=0.8711 acc=37.29%
teste: acurácia=40/60 = 66.67%
```

### 3. Streamlit Web App
```bash
streamlit run app.py
```
Interface web com abas: Dados → Arquitetura → Treino (AO VIVO) → Resultados → Inferência

---

## 🎮 Controles da Interface PyQt6

| Área | Controles |
|------|-----------|
| **Dataset** | Carregar heart.csv / diabetes.csv / CSV personalizado |
| **Arquitetura** | Camadas ocultas (0-5), neurônios/camada (1-64), ativação (sigmoid/relu/identity) |
| **Entrada Manual** | Inputs separados por vírgula, Forward completo, Passo a passo, Auto-play |
| **Navegação** | Próximo/Anterior sample, Fast Forward (teste completo) |
| **Treino** | Learning Rate, Train Step, Train Época (1-100), Parar, Ver Loss |
| **Visualização** | Mostrar pesos, bias, nomes das features |

---

## 🧠 Conceitos Implementados

### Neuron (`neuron.py`)
```python
@dataclass
class Neuron:
    bias: float
    input_sum: float      # soma ponderada + bias
    output: float         # após ativação
    delta: float          # gradiente local (backprop)
    neuron_id: str        # UUID único
    layer_index: int      # índice da camada
    position_in_layer: int
```

### Connection (`connection.py`)
```python
@dataclass
class Connection:
    source: Neuron
    target: Neuron
    weight: float         # inicial aleatório [-1, 1]
    gradient: float       # dL/dw (calculado no backprop)
```

### Network (`network.py`)
- **Forward**: `set_input()` → `forward_step()` (neurônio a neurônio) ou `forward_all()`
- **Backprop**: MSE loss, 1 neurônio saída, chain rule manual
- **Gradientes**: `dL/dw = source.output * target.delta`, `dL/db = delta`
- **Treino**: `train_step()`, `train_epoch()`, atualização SGD
- **Arquitetura dinâmica**: `add_hidden_layer()`, `remove_hidden_layer()`, `add_neuron_to_hidden_layer()`, `remove_neuron_from_hidden_layer()`

### Graph Utils (`graph_utils.py`)
- Layout automático: camadas horizontais, neurônios verticais centralizados
- Cores: pesos (azul+/vermelho-), gradientes (verde+/magenta-)
- Espessura proporcional à magnitude

---

## 📊 Datasets

### Heart Disease (`rsc/heart.csv`)
- **Fonte**: UCI Cleveland
- **1025 amostras**, **13 features**, target binário (0/1)
- Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

### Diabetes 130-US (`rsc/diabetes.csv`)
- **Fonte**: UCI Diabetes 130-US Hospitals (1999-2008)
- **101.766 amostras**, **39 features** (após processamento)
- Target: readmitted < 30 dias (1) vs ≥30/NO (0) — **imbalance 89%/11%**
- Features: demografia, histórico médico, medicamentos, diagnósticos

---

## 🏗️ GitFlow (Versionamento)

```bash
# Branches principais
main          # Releases (tagged v0.1.0, v1.0.0...)
develop       # Integração contínua
feature/*     # Novas funcionalidades
hotfix/*      # Correções urgentes em produção

# Fluxo típico
git checkout develop
git checkout -b feature/nova-funcionalidade
# ... desenvolve ...
git checkout develop
git merge --no-ff feature/nova-funcionalidade
git branch -d feature/nova-funcionalidade

# Release
git checkout main
git merge --no-ff develop
git tag -a v1.0.0 -m "Release v1.0.0"
```

---

## 🧪 Testes

```bash
pytest tests/ -v
```

---

## 📚 Documentação

| Arquivo | Conteúdo |
|---------|----------|
| `docs/architecture.md` | Arquitetura detalhada (grafo, forward, backprop) |
| `docs/api.md` | Referência completa de classes/métodos |
| `docs/experiment_guide.md` | Guia de experimentos, hiperparâmetros, diabetes challenge |
| `docs/project_explanation.md` | Documento completo para apresentação |

---

## 🔮 Próximos Passos

- [ ] Gráfico de loss em tempo real na GUI (matplotlib/QCustomPlot)
- [ ] Múltiplos neurônios de saída + Softmax
- [ ] Persistência (salvar/carregar pesos JSON/pickle)
- [ ] Normalização integrada na Network
- [ ] Mini-batches e shuffle no treino
- [ ] Dropout / BatchNorm na arquitetura dinâmica
- [ ] Exportar imagem da topologia (PNG/SVG)

---

## 📄 Licença

Uso educacional. Adapte livremente para fins de ensino.

---

**Desenvolvido para Tópicos Avançados em IA**  
*Implementação from scratch de MLP com estrutura de grafo, feed-forward, backpropagation e visualização PyQt6/Streamlit*