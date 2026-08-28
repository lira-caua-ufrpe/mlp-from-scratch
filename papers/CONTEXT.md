# Papers Subtree Context
> Diretório dedicado à produção acadêmica: relatórios técnicos e artigo científico.

## 📄 Arquivos e Estrutura:
- `main.tex`: Arquivo mestre do documento LaTeX. Inclui os pacotes e importa as seções.
- `references.bib`: Base de dados BibTeX para citações científicas.
- `sections/`: Seções do documento divididas em arquivos `.tex` modulares:
  - `01_introducao.tex`: Motivação, contextualização e objetivos.
  - `02_trabalhos_relacionados.tex`: Estado da arte e literatura correlata.
  - `03_metodologia.tex`: Formulação teórica, arquitetura e métodos.
  - `04_experimentos.tex`: Protocolo experimental, métricas e discussão de resultados.
  - `05_conclusao.tex`: Conclusões e trabalhos futuros.
- `figures/`: Imagens, esquemas arquiteturais e gráficos (preferencialmente em formato PDF ou PNG em alta resolução).

## 🔄 Conexão com o Overleaf:
Para sincronizar este diretório com o projeto Overleaf do professor via Git:
1. Adicione o remote do Overleaf:
   ```bash
   git remote add overleaf https://git.overleaf.com/<ID_DO_PROJETO>
   ```
2. Para enviar alterações feitas localmente pelo agente:
   ```bash
   git push overleaf main
   ```
3. Para puxar alterações feitas online no Overleaf:
   ```bash
   git pull overleaf main
   ```
