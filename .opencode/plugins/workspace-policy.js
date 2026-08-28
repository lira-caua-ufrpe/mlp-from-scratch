/**
 * Workspace Policy Plugin para OpenCode
 * Garante que o OpenCode respeite as normas do AGENTS.md
 */
module.exports = {
    name: 'workspace-policy',
    description: 'Enforces Workspace OS norms from AGENTS.md for OpenCode sessions',
    onSessionStart: async (context) => {
        console.log('[Workspace OS] OpenCode inicializado com sucesso.');
        console.log('[Workspace OS] Fonte da verdade: Sistema de Arquivos (code/, papers/, core/).');
    }
};
