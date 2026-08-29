# 02. Mapeamento de Microexpressões e AUs Faciais (FACS) em TEA
> Utilização do Facial Action Coding System (FACS) e malha facial (MediaPipe FaceMesh) para identificação precoce de sobrecarga sensorial.

---

## 1. Fundamentos do FACS no Autismo

O sistema **FACS (Facial Action Coding System)**, desenvolvido por Paul Ekman, mapeia os movimentos anatômicos dos músculos faciais em **Action Units (AUs)** individuais.

Em crianças autistas, a expressão emocional macroscópica pode ser atípica ou amortecida, mas as **microexpressões involuntárias (duração entre 100 e 500 ms)** revelam o estado interno do sistema nervoso autônomo.

---

## 2. Action Units Críticas para Detecção de Sobrecarga e Estresse

| Action Unit (AU) | Músculo Envolvido | Manifestação Visual | Significado Clínico em TEA |
| :--- | :--- | :--- | :--- |
| **AU 04 (Brow Lowerer)** | *Corrugator supercilii* | Sobrancelhas franzidas e puxadas para baixo e centro. | Esforço cognitivo excessivo, dor, frustração ou sobrecarga sensorial. |
| **AU 01 + AU 02 (Brow Raiser)** | *Frontalis* | Elevação da parte interna ou externa das sobrancelhas. | Alarme, surpresa ou hipervigilância ambiental. |
| **AU 12 (Lip Corner Puller)** | *Zygomaticus major* | Cantos da boca puxados para trás/cima (sorriso assimétrico). | Em TEA, pode ocorrer como "sorriso atípico de tensão" (não alegria). |
| **AU 24 (Lip Pressor)** | *Orbicularis oris* | Lábios pressionados e comprimidos um contra o outro. | Tensão mandibular, retenção de frustração ou tentativa de autocontrole. |
| **AU 43 (Eyes Closed)** | *Levator palpebrae* | Fechamento prolongado dos olhos (> 500 ms sem piscar). | Fotofobia ou tentativa de bloqueio de sobrecarga visual. |

---

## 3. Mapeamento dos Marcos Faciais no MediaPipe FaceMesh (468 Pontos)

No MediaPipe FaceMesh, as AUs são calculadas geometricamente através das distâncias euclidianas entre pontos específicos:

```
                  MAPEAMENTO GEOMÉTRICO FACS (FaceMesh)
       [Sobrancelha Esq: 70, 63, 105]     [Sobrancelha Dir: 336, 296, 334]
                    \                             /
                     \                           /
                      ──▶ [Ponto Glabela: 9] ◀── (AU 04: Franzimento)
                                 │
                         [Ponta Nariz: 1]
                                 │
           [Canto Boca Esq: 61] ─── [Canto Boca Dir: 291] (AU 12 / AU 24)
```

1. **Cálculo da AU 04 (Franzir Sobrancelhas):**
   $$\text{Distância}_{\text{AU04}} = \frac{\|\mathbf{p}_{105} - \mathbf{p}_{334}\|}{\|\mathbf{p}_{234} - \mathbf{p}_{454}\|}$$
   *Uma redução na distância relativa entre as sobrancelhas normalizada pela largura do rosto indica ativação da AU 04.*

2. **Cálculo de Desvio de Olhar (Gaze Drift / Aversion):**
   * Posição da íris esquerda (468) e direita (473) em relação ao centro dos olhos (33, 133 e 362, 263).
