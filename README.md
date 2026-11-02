# PRISMX: Unlocking the Black Box of CNNs via Principal Image Sections Mapping

This repository contains the complete reproduction and novel research extensions of the paper:

> **Unlocking the black box of CNNs: Visualising the decision-making process with PRISM**  
> Tomasz Szandała  
> *Information Sciences*, Volume 642, 2023, 119162.  
> DOI: [10.1016/j.ins.2023.119162](https://doi.org/10.1016/j.ins.2023.119162)

---

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Analysis of Original Method Limitations](#theoretical-analysis-of-original-method-limitations)
3. [Summary of Benchmarks and Findings](#summary-of-benchmarks-and-findings)
4. [Phase 1: Published Paper Reproduction](#phase-1-published-paper-reproduction)
5. [Phase 2: Proposed PRISMX Extensions](#phase-2-proposed-prismx-extensions)
6. [Interactive Custom Testing Suite](#interactive-custom-testing-suite)
7. [Repository Architecture](#repository-architecture)
8. [Installation, Testing, and Execution](#installation-testing-and-execution)
9. [Formal Submission Documents & Repository Link](#formal-submission-documents--repository-link)
10. [Identification of Modified and New Files](#identification-of-modified-and-new-files)

---

## Overview

Explainable Artificial Intelligence (XAI) for Convolutional Neural Networks (CNNs) has predominantly utilized Class Activation Mapping (CAM, Grad-CAM). However, CAM methods generate coarse spatial heatmaps that identify regions of interest without isolating individual semantic features.

**PRISM (Principal Image Sections Mapping)** addresses this limitation by applying Singular Value Decomposition (SVD) to the final convolutional feature maps ($512 \times 7 \times 7$ in VGG-16). By projecting singular components into RGB channels and upsampling via Gradual Extrapolation (GE), PRISM generates color-coded, high-resolution feature sections corresponding to distinct anatomical and structural components.

This repository:
1. Re-implements and verifies all primary paper experiments with reproducible consistency.
2. Formulates and resolves two theoretical limitations:
   * **The 3-PCA Representation Bottleneck:** Alleviates feature variance loss and restores model prediction certainty.
   * **Target-Agnostic Distractor Contamination:** Eliminates background interference using pre-SVD dual spatial-channel gradient gating.

---

## Theoretical Analysis of Original Method Limitations

### Limitation 1: The 3-PCA Representation Bottleneck

#### Technical Cause
In VGG-16, layer `features.30` outputs 512 channels across a $7 \times 7$ spatial grid. To render outputs onto standard 3-channel RGB displays, PRISM hardcodes $K=3$:
$$PC_1 \rightarrow \text{Red}, \quad PC_2 \rightarrow \text{Green}, \quad PC_3 \rightarrow \text{Blue}$$
All remaining 509 dimensions ($PC_4$ through $PC_{512}$) are discarded.

#### Empirical Evaluation
1. **Variance Loss:**
   * On 2 canine classes (Wolf and Coyote), the top 3 components capture **56.5% variance** (**43.5% lost**).
   * On 5 canine classes (Wolf, Coyote, Samoyed, Collie, Fox), top-3 components capture **42.3% variance** (**57.7% lost**).
2. **Classifier Confidence Collapse:**
   Passing the reconstructed tensor $\hat{F}_3 = U_3 S_3 V_3^T + \mu$ through `model.classifier`:
   * **Coyote:** Softmax probability drops from **0.9255 down to 0.4740** (a 45% reduction).
   * **Timber Wolf:** Softmax probability drops from **0.9118 down to 0.6035** (a 31% reduction).

#### PRISMX Solution
* **Adaptive Variance Threshold ($\ge 85\%$):** Dynamically selects $K$ (e.g., $K=14\text{--}15$) to preserve at least 85% of total variance.
* **Feature Reconstruction vs. Visualization Projection:**
  * **SVD Feature Reconstruction ($\hat{F}_{15} \in \mathbb{R}^{B \times 512 \times 7 \times 7}$):** $\hat{F}_{15} = U_{15} S_{15} V_{15}^T + \mu$ reconstructs the 512-channel feature tensor, passed into `model.classifier` to verify prediction preservation.
  * **SVD 3-Channel Projection ($P_3 \in \mathbb{R}^{B \times 3 \times 7 \times 7}$):** A variance-weighted linear projection matrix $W \in \mathbb{R}^{K \times 3}$ ($W_{i, i \bmod 3} = S_i / S_0$) maps $K \ge 15$ dimensions into an RGB representation for visual inspection.
* **Hierarchical Spectral-Band Decomposition:** Splits the 15 components into 3 dedicated visual tiers:
  * **Tier 1 (PCs 1–3: Macro):** Coarse body silhouettes and broad anatomical mass.
  * **Tier 2 (PCs 4–6: Contours):** Structural edges, muzzle outlines, and ear perimeters.
  * **Tier 3 (PCs 7–9: Textures):** High-frequency fur striations, whiskers, and mechanical details.
* **Results:** Relative feature reconstruction error drops from **59.2% to 32.8%**, and top-1 prediction confidence returns to **0.902**.

---

### Limitation 2: Target-Agnostic Distractor Contamination

#### Technical Cause
* PRISM computes SVD in an unsupervised manner based on maximum spatial activation variance, without conditioning on the predicted class label.
* High-contrast background elements (e.g., a mushroom in a forest, a metallic coin, or pedestrians) dominate activation variance and are assigned primary color channels, creating the false visual impression of model relevance.

#### Empirical Evaluation
1. **Malamute + Mushroom (Paper Figure 12):**
   * VGG-16 predicts Malamute (37.8%), Eskimo Dog (35.2%), and Husky (24.5%) — total canine confidence of **97.5%**. Bolete Mushroom probability is **0.0000%**.
   * Standard PRISM allocates a primary color channel to the mushroom with an activation energy of **1.015** (Dog-to-Mushroom Ratio = **1.05**).
2. **Black Widow + Penny Coin (Paper Figure 6):**
   * CNN predicts Black Widow (85.7%). PRISM allocates a major color component to the metallic coin ($\text{SDR} = 1.27$).
3. **Failure of Post-Hoc Overlay:**
   * Alpha-blending Grad-CAM over PRISM (`--saliency grad_cam`) fails because SVD has already assigned components to the distractor prior to masking.

#### PRISMX Solution: Dual Spatial-Channel Gradient Gating
* **Formulation:** Applies channel gradient weighting and spatial saliency modulation prior to SVD:
  $$\tilde{A}^k(u,v) = A^k(u,v) \cdot w_{ch}^k \cdot (M^c(u,v))^\gamma$$
  * **Feature-Channel Gradient Gating ($w_{ch}^k$):** Filters channels by pooled positive gradient contribution:
    $$w_{ch}^k = \frac{\text{ReLU}(\alpha_k^c)}{\max_m \alpha_m^c + \epsilon}, \quad \alpha_k^c = \frac{1}{HW}\sum_{u,v}\frac{\partial Y^c}{\partial A_{u,v}^k}$$
  * **Spatial Attentive Modulation ($M^c(u,v)$):** Modulates spatial activations by normalized Grad-CAM saliency:
    $$M^c(u,v) = \frac{\text{ReLU}\left(\sum_k \alpha_k^c A_{u,v}^k\right)}{\max_{x,y}\sum_k \alpha_k^c A_{x,y}^k + \epsilon}$$
* **Contrastive cPRISM:** Disentangles overlapping competing classes counterfactually:
  $$w_{\text{contrast}}^k = \frac{\text{ReLU}(\alpha_k^{c_1} - \alpha_k^{c_2})}{\max_m (\alpha_m^{c_1} - \alpha_m^{c_2}) + \epsilon}$$
* **Results:** Mushroom energy drops from 1.015 to 0.641; target selectivity improves by **+44.3%**. Coin distractor energy is attenuated ($\text{SDR} > 1000$).

---

## Summary of Benchmarks and Findings

| Metric / Experiment | Paper Baseline | Our Reproduction | Proposed PRISMX | Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness Drop (5 cells)** | 0.54 drop | **0.51 drop** | N/A | Close replication within empirical margin |
| **Class Decision Inversion** | Fig. 4, 3, 5 inverted | **Inverted across all pairs** | N/A | Verified on canines, food, trains |
| **Spider Exclusive Cells (Fig. 6)** | 6, 15, 21 cells | **6, 15, 21 cells** | N/A | Exact match to published cell counts |
| **Top-3 Variance (5 Canines, Fig. 17)**| ~40% variance | **42.3% variance** | N/A | Multi-class variance drop verified |
| **Variance Retained (Information)** | 56.5% (43.5% lost) | 56.5% (43.5% lost) | **86.7% (13.3% lost)** | **+30.2% variance preserved** |
| **Decision Fidelity (Coyote)** | 0.4740 (collapsed) | 0.4740 (collapsed) | **0.9020 (restored)** | **+42.8% confidence recovery** |
| **Decision Fidelity (Wolf)** | 0.6035 (collapsed) | 0.6035 (collapsed) | **0.9017 (restored)** | **+29.8% confidence recovery** |
| **Relative Feature Error** | 59.2% error | 59.2% error | **32.8% error** | **Error reduced by 44.6%** |
| **Distractor Selectivity (Fig. 12)** | SDR = 1.05 | SDR = 1.05 | **SDR = 1.51** | **+44.3% selectivity gain** |
| **Distractor Selectivity (Coin)** | SDR = 1.27 (Coin leaks) | SDR = 1.27 | **SDR > 1000 (Suppressed)** | **High distractor attenuation** |

---

## Phase 1: Published Paper Reproduction

### 1. Faithfulness via Representation Swapping (`experiments/exp1_faithfulness.py`, `experiments/exp4_multiclass_faithfulness.py`)
* **Canines (Figure 4):** 5-cell swap replicates the confidence drop ($0.93 \rightarrow 0.42$, drop of 0.51 vs. 0.54 paper benchmark). Full head swap induces Top-1 classification inversion (Coyote becomes Wolf $0.60$ vs. $0.13$; Wolf becomes Coyote $0.60$ vs. $0.23$). Output: `outputs/reproduced_fig4_complete_inversion.png`.
* **Food & Trains (Figures 3 & 5):**
  * Cheeseburger vs. Bagel (Fig. 3): 6-cell swap flips prediction ($0.73 \rightarrow 0.08$ vs. Bagel $0.20$). Output: `outputs/reproduced_fig3_cheeseburger_bagel_swapping.png`.
  * Passenger Car vs. Locomotive (Fig. 5): Inversion occurs at 2 cells ($0.55 \rightarrow 0.35$ vs. Car $0.56$). Output: `outputs/reproduced_fig5_trains_swapping.png`.

### 2. Interpretability and Exclusive Feature Isolation (`experiments/exp2_interpretability.py`, `experiments/exp2_trains_interpretability.py`)
* **Spiders (Figure 6):** Validates active exclusive cell counts on exact spider crops: Black Widow 1 (**6 cells**), Garden Spider (**15 cells**), Black Widow with Penny Coin (**21 cells**). Output: `outputs/reproduced_fig6_exact_spiders.png`.
* **Trains (Figure 7):** PRISM Exclusive isolates passenger window strips for passenger cars and pantograph roof structures for locomotives. Output: `outputs/reproduced_trains_interpretability_cam_vs_prism.png`.

### 3. SVD Variance Distribution Analysis (`experiments/exp3_variance_analysis.py`)
* 2-Class Canine Set: Top-3 capture **56.5% variance** (43.5% discarded). Output: `outputs/reproduced_fig16_variance_2classes.png`.
* 5-Class Canine Set: Top-3 drop to **42.3% variance** (57.7% discarded). Output: `outputs/reproduced_fig17_exact_5classes.png`.

---

## Phase 2: Proposed PRISMX Extensions

### 1. High-Variance Manifold PRISM & Spectral Bands (`improvements/imp1_resolve_3pca_bottleneck.py`)
* Retains $\ge 85\%$ variance ($K=15$) and implements 3-tier Spectral Bands. Outputs: `outputs/solution_3pca_scree_fidelity.png`, `outputs/solution_3pca_spectral_bands.png`.
* Side-by-side 3-case fidelity cards (`experiments/eval_fidelity_comparison.py`). Output: `outputs/fidelity_and_variance_comparison.png`.
* Residual energy heatmap for PCs 4–15 (`experiments/eval_wolf_difference_revealed.py`). Output: `outputs/wolf_3cases_difference_revealed.png`.

### 2. Class-Attentive Grad-PRISM & cPRISM (`improvements/imp2_class_attentive_grad_prism.py`)
* Implements Dual Spatial-Channel Gradient Gating and contrastive counterfactual analysis. Output: `outputs/solution_distractors_grad_prism.png`.
* Evaluates distractor resolution on Malamute + Mushroom (`experiments/eval_distractor_malamute_mushroom.py`). Output: `outputs/malamute_mushroom_distractor_resolved.png`.

### 3. Multi-Domain Generalization (`experiments/eval_multidomain_improvements.py`)
* **Electric Locomotive:** Recovers roof pantograph structures ($47.8\% \rightarrow 87.6\%$). Output: `outputs/locomotive_imp1_spectral_fidelity.png`.
* **Cheeseburger:** Isolates fine sesame seed textures ($56.2\% \rightarrow 89.4\%$). Output: `outputs/cheeseburger_imp1_spectral_fidelity.png`.
* **Black Widow + Coin:** Attenuates metallic distractor ($\text{SDR} > 1000$). Output: `outputs/spider_coin_distractor_resolved.png`.
* **Golden Retriever + Cat:** Disentangles overlapping animal representations. Outputs: `outputs/user_dog_cat_imp1_spectral_fidelity.png`, `outputs/user_dog_cat_imp2_disentanglement.png`.
* **Street Car:** Suppresses dense pedestrian clutter. Outputs: `outputs/user_car_street_imp1_spectral_fidelity.png`, `outputs/user_car_street_imp2_clutter_suppression.png`.

---

## Interactive Custom Testing Suite

A modular testing pipeline is provided in `custom_test_suite/` to evaluate arbitrary input images:

1. Place image(s) into:
   ```text
   custom_test_suite/input_data/
   ```
2. Run the pipeline:
   ```bash
   python custom_test_suite/run_all_custom.py
   ```
3. Generated comparison figures are saved into:
   ```text
   custom_test_suite/outputs/
   ```

---

## Repository Architecture

```text
PRISMX/
├── .github/workflows/ci.yml            # Automated GitHub Actions CI workflow
├── custom_test_suite/                  # Interactive Custom Testing Framework
│   ├── input_data/                     # Directory for custom input images
│   ├── outputs/                        # Output figures from custom tests
│   └── run_all_custom.py               # Master CLI runner for custom inputs
├── data/                               # Datasets and benchmark images
│   ├── samples/                        # Canines, burgers, trains, and distractors
│   ├── spiders/                        # Exact spider crops from Paper Figure 6
│   ├── five_canines/                   # Exact 5 canine classes from Paper Figure 14
│   └── composites/                     # Multi-object test scenes
├── experiments/                        # Structured evaluation scripts
│   ├── exp1_faithfulness.py            # Faithfulness swapping on wolves (Fig. 4)
│   ├── exp2_interpretability.py        # Spider interpretability benchmark (Fig. 6)
│   ├── exp2_trains_interpretability.py # Train interpretability benchmark (Fig. 7)
│   ├── exp3_variance_analysis.py       # SVD variance analysis on 2 & 5 classes (Fig. 16/17)
│   ├── exp4_multiclass_faithfulness.py # Food & train swapping (Fig. 3 & Fig. 5)
│   ├── eval_distractor_malamute_mushroom.py # Fig. 12 distractor resolution
│   ├── eval_fidelity_comparison.py     # Side-by-side 3-case fidelity breakdown
│   ├── eval_multidomain_improvements.py# Multi-domain tests (vehicles, food, clutter)
│   └── eval_wolf_difference_revealed.py# Higher PCs residual energy heatmap
├── improvements/                       # Proposed algorithmic solutions
│   ├── imp1_resolve_3pca_bottleneck.py # High-Variance Manifold & Spectral Bands
│   └── imp2_class_attentive_grad_prism.py # Dual Spatial-Channel Grad-PRISM & cPRISM
├── outputs/                            # Publication-quality generated figures
│   └── experiment_metadata.json        # Machine-readable experiment provenance log
├── src/                                # Core Engine
│   └── torchprism/                     # Modular PyTorch PRISM package
│       ├── classes.json                # ImageNet 1000 class index mapping
│       ├── metadata.py                 # Dynamic class resolution & provenance logger
│       └── PRISM.py                    # Base PRISM implementation
├── tests/                              # Automated Unit Test Suite
│   ├── test_torchprism.py              # SVD conservation & metadata resolution tests
│   ├── test_improvements.py            # Spectral reconstruction & dual gating tests
│   └── run_tests.py                    # Master test runner
├── AI_USAGE.md                         # Formal academic AI usage disclosure
├── EXPERIMENTAL_RESULTS_AND_COMPARISON.md # Complete experimental results & comparison report
├── RESEARCH_GAP.md                     # 1-page theoretical research gap report
├── README.md                           # Master project documentation
├── requirements.txt                    # Project environment dependencies
├── run_all.py                          # Master execution pipeline
└── .gitignore                          # Standard repository ignore file
```

---

## Installation, Testing, and Execution

### 1. Prerequisites and Setup
```bash
git clone https://github.com/gourav18dart/PRISMX.git
cd PRISMX

# Install dependencies:
pip install -r requirements.txt
```

### 2. Run Automated Unit Test Suite
```bash
python tests/run_tests.py
```
Executes 8 automated unit tests covering hook registration, SVD variance conservation, rank-$K$ reconstruction convergence, dual spatial-channel gating bounds, and dynamic class ID resolution.

### 3. Run All Benchmarks in Sequence
```bash
python run_all.py
```
Executes the automated unit tests, all reproduction experiments, and novel improvement benchmarks in sequence, generating all figures in `outputs/` and recording execution provenance to `outputs/experiment_metadata.json`.

### 4. Run Custom Tests on Your Own Images
```bash
python custom_test_suite/run_all_custom.py
```

---

## Formal Submission Documents & Repository Link

* **Official Repository Link:** [https://github.com/gourav18dart/PRISMX.git](https://github.com/gourav18dart/PRISMX.git)
* **Part A (Research Gap):** [RESEARCH_GAP.md](RESEARCH_GAP.md) — 1-page paper-style theoretical research gap analysis.
* **Part B (AI Usage Disclosure):** [AI_USAGE.md](AI_USAGE.md) — Academic disclosure covering tools, standardized prompts, and human verification.
* **Part C (Experimental Results & Comparison):** [EXPERIMENTAL_RESULTS_AND_COMPARISON.md](EXPERIMENTAL_RESULTS_AND_COMPARISON.md) — Complete report containing original paper page citations, reproduction benchmarks, supporting academic literature, before-and-after tables, and multi-domain evaluations.

---

## Identification of Modified and New Files

| File / Component | Category | Purpose and Exact Code Modifications |
| :--- | :---: | :--- |
| **`improvements/imp1_resolve_3pca_bottleneck.py`** | **NEW** | Implements **Improvement 1**: rank-$K$ SVD feature reconstruction ($\hat{F}_K = U_K S_K V_K^T + \mu$), quantitative sweep across $K=1..30$, 3-tier Spectral Bands (Macro, Contours, Textures), and variance-weighted 3-channel projection matrix $W \in \mathbb{R}^{K \times 3}$. |
| **`improvements/imp2_class_attentive_grad_prism.py`** | **NEW** | Implements **Improvement 2**: Pre-SVD **Dual Spatial-Channel Gradient Gating** ($\tilde{A}^k = A^k \cdot w_{ch}^k \cdot (M^c)^\gamma$), and counterfactual Contrastive cPRISM ($w_{\text{diff}} = \text{ReLU}(w_1 - w_2)$). |
| **`src/torchprism/metadata.py`** | **NEW** | Provides dynamic ImageNet class ID resolution (`get_class_id`, `get_class_name`), eliminating fragile hardcoded numbers, and automated experiment provenance logging (`record_experiment_metadata`). |
| **`src/torchprism/classes.json`** | **NEW** | Complete 1000-class ImageNet index dictionary. |
| **`src/torchprism/PRISM.py`** | **MODIFIED** | Refactored Tomasz Szandała's original PRISM code into a clean, reusable PyTorch module with gradient tracking hooks and numerical stabilizers. |
| **`experiments/exp1_faithfulness.py`** | **NEW** | Reproduction of Paper Section 3.1 & Figure 4: 5-cell canine muzzle swap and full head swap showing Top-1 classification inversion. |
| **`experiments/exp2_interpretability.py`** | **NEW** | Reproduction of Paper Section 3.2 & Figure 6: Spider trio exclusive feature isolation (6, 15, and 21 cells). |
| **`experiments/exp2_trains_interpretability.py`** | **NEW** | Reproduction of Paper Section 3.2 & Figure 7: Passenger Car vs. Locomotive interpretability. |
| **`experiments/exp3_variance_analysis.py`** | **NEW** | Reproduction of Paper Section 4.1 & Figures 16/17: Scree variance distribution analysis on 2-class and 5-class sets. |
| **`experiments/exp4_multiclass_faithfulness.py`** | **NEW** | Reproduction of Paper Figures 3 & 5: Faithfulness swapping on food items and trains. |
| **`experiments/eval_distractor_malamute_mushroom.py`** | **NEW** | Rigorous evaluation of Paper Figure 12 Malamute + Mushroom distractor resolution using Dual Spatial-Channel Gating. |
| **`experiments/eval_fidelity_comparison.py`** | **NEW** | Side-by-side 3-case fidelity breakdown scorecard (512 dims vs 3 PCs vs 15 PCs). |
| **`experiments/eval_wolf_difference_revealed.py`** | **NEW** | Higher PCs residual energy heatmap revealing the hidden 43.5% variance. |
| **`experiments/eval_multidomain_improvements.py`** | **NEW** | Multi-domain evaluation across vehicles, food items, insects, multi-animal scenes, and dense street clutter. |
| **`tests/test_torchprism.py`** | **NEW** | Automated unit tests for SVD variance conservation, singular value monotonicity, and metadata resolution. |
| **`tests/test_improvements.py`** | **NEW** | Automated unit tests for rank-$K$ reconstruction convergence and Dual Spatial-Channel Gating bounds. |
| **`tests/run_tests.py`** | **NEW** | Master automated test runner (8/8 passing). |
| **`.github/workflows/ci.yml`** | **NEW** | Automated GitHub Actions CI workflow executing unit tests on Python 3.10, 3.11, and 3.12. |
| **`run_all.py`** | **NEW** | Master end-to-end pipeline orchestrating automated unit tests and all 11 experimental benchmarks in sequence. |
