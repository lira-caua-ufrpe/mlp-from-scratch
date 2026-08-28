# Workspace Root - Normas do Agente
> Ponto de entrada canônico do Workspace. Lido obrigatoriamente antes de qualquer tarefa por qualquer Harness (Antigravity, OpenCode, Claude Code).

<!-- norms:start -->
- **SISTEMA DE ARQUIVOS É A FONTE DA VERDADE**: Nada vive apenas na memória do chat ou em configs voláteis. Se existe e importa, é um arquivo versionado.
- **STORAGE AGNÓSTICO DE HARNESS**: O workspace é o dono do seu estado, nunca o harness. Antigravity (principal) e OpenCode (backup) operam sobre os mesmos arquivos.
- **SEGREDOS FORA DO GIT**: Senhas, tokens, chaves de API e credenciais ficam em `.env` ou `segredos.env` (sempre no `.gitignore`).
- **NÃO ASSUMA, PERGUNTE**: Em caso de ambiguidade sobre o desenho experimental, requisitos ou estrutura do artigo, entreviste o usuário.
- **EDITAR > CRIAR**: Prefira refatorar, modularizar e aprimorar arquivos existentes a criar novos arquivos dispersos.
- **DIVISÃO RÍGIDA DE DOMÍNIOS**:
  - `code/`: Todo o desenvolvimento de modelos, scripts de treino, pipelines de dados e visualizações de IA.
  - `papers/`: Toda a produção de artigos científicos, relatórios técnicos, seções `.tex`, figuras e referências `.bib`.
  - `core/`: Utilitários, scripts de automação, documentações de harness e ferramentas de suporte.
- **GIT FLOW OBRIGATÓRIO**:
  - `main`: Versão estável (entregas finais de código e artigo compilado).
  - `develop`: Branch de integração ativa.
  - `feature/*`: Novas funcionalidades, seções de texto ou experimentos específicos.
- **LIMPEZA E HIGIENE**: Arquivos de build (LaTeX `.aux`, `.log`, caches Python `__pycache__`) nunca são commitados.
<!-- norms:end -->

---

## 🗺️ Mapa de Roteamento de Contexto
Antes de modificar qualquer arquivo, consulte o `CONTEXT.md` do diretório correspondente:
- Raiz: [CONTEXT.md](CONTEXT.md)
- Códigos e Modelos: [code/CONTEXT.md](code/CONTEXT.md)
- Artigo e Pesquisa: [papers/CONTEXT.md](papers/CONTEXT.md)
- Ferramentas e Harness: [core/CONTEXT.md](core/CONTEXT.md)
