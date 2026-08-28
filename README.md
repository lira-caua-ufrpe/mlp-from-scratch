# Workspace OS — Tópicos Avançados em Inteligência Artificial

Ambiente integrado para pesquisa, experimentação prática de código e confecção de artigos científicos, operado por agentes de IA e versionado sob GitFlow.

Inspirado no modelo de [lsfcin/workspace](https://github.com/lsfcin/workspace).

---

## 📁 Estrutura do Repositório

```
workspace/
├── AGENTS.md            # Normas e diretrizes de trabalho para agentes de IA
├── CONTEXT.md           # Roteamento de contexto para agentes
├── INBOX.md             # Backlog de tarefas, anotações e ideias
├── README.md            # Este arquivo
├── .gitignore           # Exclusões de arquivos temporários, builds e segredos
│
├── code/                # Projetos práticos e experimentos
│   ├── CONTEXT.md
│   └── mlp_viewer/      # Implementação e visualizador de MLP (PyQt6 / Streamlit)
│
├── papers/              # Produção acadêmica (Artigo e Relatórios Técnicos)
│   ├── CONTEXT.md
│   ├── main.tex         # Documento principal em LaTeX
│   ├── references.bib   # Referências bibliográficas (BibTeX)
│   ├── sections/        # Seções modulares (.tex)
│   └── figures/         # Figuras, gráficos e diagramas
│
└── core/                # Automações, ferramentas e políticas de harness
    ├── CONTEXT.md
    └── harness/         # Guia de failover (Antigravity <-> OpenCode)
```

---

## 🤖 Harnesses de IA (Principal & Contingência)

1. **Harness Principal (Google Antigravity):**
   - Agente autônomo com suporte a planejamento, edição de múltiplos arquivos, execução de testes e interface gráfica.
2. **Harness de Backup (OpenCode):**
   - Usado quando a cota do modelo principal atingir o limite diário.
   - Compartilha exatamente os mesmos arquivos e lê as diretrizes em `AGENTS.md`.
   - Consulte o guia completo em [`core/harness/FAILOVER.md`](core/harness/FAILOVER.md).

---

## 📝 Artigo Científico e Overleaf

O artigo principal está modularizado em `papers/`.

### Como compilar localmente:
```bash
# Na pasta papers:
cd papers
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Como sincronizar com o Overleaf:
1. Assim que o professor fornecer o repositório/link do Overleaf com Git Bridge:
   ```bash
   git remote add overleaf <URL_GIT_DO_OVERLEAF>
   git fetch overleaf
   ```
2. Consulte o passo a passo completo em [`papers/CONTEXT.md`](papers/CONTEXT.md).

---

## 🌿 GitFlow

- **`main`**: Versão estável (entregas finais da disciplina).
- **`develop`**: Desenvolvimento ativo contínuo.
- **`feature/<nome>`**: Novas funcionalidades de código ou seções de artigos.
