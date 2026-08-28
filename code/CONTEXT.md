# Code Subtree Context
> Diretório de implementações práticas, algoritmos e modelos da disciplina.

## 📦 Estrutura de Projetos:
- `mlp_viewer/`: Implementação de Multi-Layer Perceptron como grafo explícito com PyQt6 e demo Streamlit.
- `tea_monitor/`: Sistema web de visão computacional (MediaPipe + Flask + Túnel HTTPS) para monitoramento em tempo real de estereotipias motoras (*stimming*) e sobrecarga sensorial em alunos com TEA (Disciplina de Tecnologias na Educação).

## ⚙️ Diretrizes para Agentes em `code/`:
1. Mantenha os testes automatizados atualizados (`pytest`).
2. Exporte saídas gráficas e figuras de experimentos para `papers/figures/` quando relevante para o artigo.
3. Não crie scripts avulsos na raiz de `code/`; use subpastas de projetos dedicadas.
