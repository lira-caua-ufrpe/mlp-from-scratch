# 04. Integração em Sala de Aula, Ética & LGPD
> Diretrizes pedagógicas, ergonomia para o professor e modelo tripartite de cooperação com psicólogos e equipe multidisciplinar.

---

## 1. O Desafio da Sala de Aula Real

Em uma turma com 30 alunos, o professor desempenha múltiplas funções simultâneas (conduzir o conteúdo, tirar dúvidas, manter a ordem, atender necessidades individuais).

### O que o sistema DEVE fazer:
* **Ser Invisível e Não Intrusivo:** Rodar silenciosamente em segundo plano.
* **Alertas Claros e Rápidos:** Em vez de gráficos complexos, emitir um status visual simples (Semáforo de 4 Estágios) com sugestões práticas de ação imediata.
* **Registro Automático de Evidências:** Gerar o log da aula para desonerar o professor da necessidade de preencher relatórios manuais cansativos.

### O que o sistema NUNCA deve fazer:
* ❌ Emitir alarmes sonoros estridentes que assustem o aluno autista.
* ❌ Emitir qualquer diagnóstico médico ("o aluno é autista grau X").
* ❌ Reprimir ou punir estereotipias de autorregulação.

---

## 2. O Modelo ABC para Apoio aos Psicólogos (ABA / TCC)

A psicologia comportamental aplicada ao autismo utiliza o **Paradigma ABC** para analisar funções de comportamento:

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│     A (Antecedente)       │ ──▶  │     B (Comportamento)     │ ──▶  │    C (Consequência)       │
│ O que aconteceu antes?    │      │ Qual foi a resposta?      │      │ O que foi feito em seguida?│
│ • Picos de barulho na sala│      │ • Flapping de mãos (4 Hz) │      │ • Fone abafador oferecido │
│ • Mudança de atividade    │      │ • Mãos nos ouvidos (2s)   │      │ • Pausa sensorial de 3 min│
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

Nosso sistema automatiza a captura da componente **B (Comportamento)** com carimbo de data/hora preciso e métricas quantitativas, permitindo que a escola e os psicólogos identifiquem com precisão quais foram os **Antecedentes (A)** que desencadearam cada evento.

---

## 3. Privacidade e Proteção de Dados (LGPD & Edge Computing)

* **Processamento na Borda (Edge AI):** A captura do vídeo da câmera é processada exclusivamente na memória RAM volátil do navegador do dispositivo através de WebAssembly.
* **Não Gravação de Imagens:** Nenhuma imagem ou vídeo de crianças é gravado em disco ou transmitido pela internet. Apenas métricas biomecânicas agregadas (ex: *"14:32 - Flapping detectado com frequência de 3.2 Hz por 4 segundos"*) são salvas no relatório da sessão.
