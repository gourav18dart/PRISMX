# Unlocking the Black Box of CNNs with PRISM
### Reproduction of Core Experiments & Novel Research Extensions

**Reference Paper:**  
Tomasz Szandała, *"Unlocking the black box of CNNs: Visualising the decision-making process with PRISM"*, **Information Sciences**, vol. 642, 2023, 119162.

---

## Environment Setup & Installation

All dependencies are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```

> **Deterministic Seeding:**  
> Every experiment and improvement script enforces a fixed random seed (`seed=42`) across PyTorch (`torch.manual_seed`), CUDA (`torch.cuda.manual_seed_all`), NumPy (`np.random.seed`), and CuDNN deterministic backend flags (`torch.backends.cudnn.deterministic = True`). This guarantees 100% exact numerical and visual reproducibility across different runs and hardware.

---

## Project Directory Structure

```
Project/
│
├── README.md                                  # Complete project documentation and guide
├── requirements.txt                           # Dependencies (PyTorch, Torchvision, Matplotlib, PIL)
├── run_all.py                                 # Master script running all 5 pipelines end-to-end
│
├── data/                                      # Cleanly organized benchmark datasets & images
│   ├── samples/                               # Official repo samples (coyote.jpg, wolf_01.jpg, etc.)
│   ├── spiders/                               # Exact Figure 6 spider crops (Black Widow 1, Garden Spider, Widow + Penny)
│   ├── five_canines/                          # Exact Figure 14 & 17 five distinct canine classes
│   └── composites/                            # Multi-object distractor scenes (Coyote + Wolf competitor)
│
├── src/                                       # Core PRISM library modules and hooks
│   └── torchprism/                            # PRISM implementation and ImageNet class index mapping
│
├── experiments/                               # PHASE 1: The 3 Core Paper Reproduction Experiments
│   ├── exp1_faithfulness.py                   # Exp 1: Representation Swapping (Fig. 4 & Complete Inversion)
│   ├── exp2_interpretability.py               # Exp 2: Exclusive PRISM vs Grad-CAM on exact Spiders (Fig. 6)
│   └── exp3_variance_analysis.py              # Exp 3: SVD Scree Analysis on 2-Class & 5-Class sets (Figs. 16 & 17)
│
├── improvements/                              # PHASE 2: Novel Research Contributions
│   ├── imp1_resolve_3pca_bottleneck.py        # Imp 1: Spectral-Band Multi-PRISM & High-Variance Composite (>85% var)
│   └── imp2_class_attentive_grad_prism.py     # Imp 2: Class-Attentive Grad-PRISM & Contrastive cPRISM
│
└── outputs/                                   # High-resolution generated publication figures
    ├── reproduced_fig4_complete_inversion.png # Top-1 class inversion figure
    ├── reproduced_fig4_faithfulness.png       # 5-cell muzzle swap figure
    ├── reproduced_fig6_exact_spiders.png      # Exact 3x3 spider interpretability grid
    ├── reproduced_fig16_variance_2classes.png # 2-class PC variance scree plot
    ├── reproduced_fig17_exact_5classes.png    # 5-class PC variance scree plot
    ├── solution_3pca_scree_fidelity.png       # 3-PCA decision collapse benchmark curve
    ├── solution_3pca_spectral_bands.png       # Spectral decomposition & high-variance composite
    └── solution_distractors_grad_prism.png    # Distractor elimination with Grad-PRISM
```

---

## How to Run the Experiments

### 1. Phase 1: The 3 Reproduction Experiments

* **Experiment 1: Faithfulness via Representation Swapping (Fig. 4)**
  ```bash
  python experiments/exp1_faithfulness.py
  ```
  * *Verifies:* Swapping 5 muzzle cells causes a $0.52$ confidence drop (matching paper's $0.54$ average benchmark). Swapping the full head region causes a **complete Top-1 species inversion** ($0.60$ vs $0.13$).
  * *Output:* `outputs/reproduced_fig4_complete_inversion.png`

* **Experiment 2: Interpretability Insight on Spiders (Fig. 6)**
  ```bash
  python experiments/exp2_interpretability.py
  ```
  * *Verifies:* While Grad-CAM produces identical blobs for both Black Widow and Garden Spider, PRISM Exclusive isolates 6 cells for Black Widow and 15 cells for Garden Spider.
  * *Output:* `outputs/reproduced_fig6_exact_spiders.png`

* **Experiment 3: SVD Variance Distribution Analysis (Figs. 16 & 17)**
  ```bash
  python experiments/exp3_variance_analysis.py
  ```
  * *Verifies:* For 2 classes, top 3 PCs capture $56.5\%$ of variance; for 5 distinct classes, variance disperses into higher PCs, dropping top-3 coverage to $42.3\%$.
  * *Outputs:* `outputs/reproduced_fig16_variance_2classes.png` and `outputs/reproduced_fig17_exact_5classes.png`

---

### 2. Phase 2: Novel Research Contributions

* **Improvement 1: Resolving the 3-PCA Information Bottleneck**
  ```bash
  python improvements/imp1_resolve_3pca_bottleneck.py
  ```
  * *Contribution:* Proves that standard 3-PC PRISM cuts decision confidence in half ($0.93 \rightarrow 0.47$). Introduces **Spectral-Band PRISM** and **High-Variance Manifold PRISM**, boosting captured variance to $86.7\%$ and restoring decision fidelity back to $>0.90$.
  * *Outputs:* `outputs/solution_3pca_scree_fidelity.png` and `outputs/solution_3pca_spectral_bands.png`

* **Improvement 2: Resolving Class-Agnostic Distractors**
  ```bash
  python improvements/imp2_class_attentive_grad_prism.py
  ```
  * *Contribution:* Pre-filters convolutional activations using class gradients *prior to SVD*, preventing background distractors from contaminating the color space.
  * *Output:* `outputs/solution_distractors_grad_prism.png`

---

### 3. Run Everything End-to-End
To run all 5 pipelines in a single command:
```bash
python run_all.py
```
