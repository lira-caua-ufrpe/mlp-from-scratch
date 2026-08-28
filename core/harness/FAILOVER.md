# Guia de Failover & Harness de Contingência

Este documento estabelece o protocolo de alternância quando o **Harness Principal (Google Antigravity)** atingir limites de taxa/cota e for necessário ativar o **Harness de Backup (OpenCode)**.

---

## 🎯 Princípio Fundamental
**O Sistema de Arquivos é o Dono do Estado.**
Nenhum estado reside exclusivamente na memória da IA. Todo o progresso, código e texto do artigo estão salvos no disco em `code/`, `papers/`, `AGENTS.md` e `INBOX.md`.

---

## 🔄 Procedimento de Failover (Antigravity -> OpenCode)

### 1. Quando o Antigravity atingir o limite:
1. Certifique-se de salvar os arquivos abertos no editor.
2. Abra o terminal na raiz do workspace (`C:\Users\conta\OneDrive\Documentos\workspace`).
3. Inicie o **OpenCode**:
   ```bash
   opencode
   ```
4. O OpenCode lerá automaticamente:
   - `AGENTS.md` (normas de comportamento e restrições)
   - `CONTEXT.md` (mapa das pastas)
   - `INBOX.md` (tarefas pendentes atuais)

### 2. Prompt Inicial sugerido para o OpenCode:
> *"Estou alternando para você como meu harness de backup. Leia `AGENTS.md` para entender as normas do workspace e consulte `INBOX.md` para ver as tarefas em andamento. Continue a partir do último commit."*

### 3. Retorno ao Antigravity:
Assim que a cota for renovada, basta continuar a conversa no Antigravity normalmente.
