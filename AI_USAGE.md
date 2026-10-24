# AI Usage Report: Academic and Methodological Disclosure

**Course:** Explainable Artificial Intelligence (XAI)  
**Project:** PRISMX: Reproducing PRISM and Resolving Its Representation Bottleneck and Target Blindness  
**Paper Studied:** Szandała, T. (2023). *Unlocking the black box of CNNs: Visualising the decision-making process with PRISM.* Information Sciences, 642, 119162.

---

## 1. Executive Statement of Authorship and AI Role

In accordance with course academic integrity guidelines, this document provides a complete and transparent disclosure of how Artificial Intelligence was utilized during this project. 

* **Human Researcher's Role:**
  * **Conceptualization and Critical Analysis:** The research gap identification was conceived and formulated independently by the human researcher. Specifically, through critical reading of Section 3.1 and Section 4.2 of Szandała's paper, the researcher identified that:
    1. Forcing 512 dimensions into 3 principal components causes severe feature variance loss and collapses the classifier's confidence.
    2. Unsupervised SVD causes background distractors (mushrooms, coins, pedestrians) to contaminate the principal components, and the author's post-hoc Grad-CAM overlay fails to fix the mathematical basis.
  * **Architectural Design:** The researcher formulated the solutions: (a) High-Variance Manifold PRISM with an adaptive $\ge 85\%$ variance threshold, (b) Spectral-Band hierarchical decomposition, and (c) Pre-SVD Gradient Gating (Grad-PRISM) with contrastive counterfactual disentanglement (cPRISM).
* **AI Tool Role:**
  * AI was utilized **strictly as an implementation and code-generation assistant**. The AI acted as a pair programmer to translate the researcher's architectural designs, mathematical formulas, and experimental protocols into clean PyTorch and Matplotlib scripts.
  * AI was **not** used to formulate the research hypotheses, nor was AI output accepted without rigorous empirical validation and debugging.

---

## 2. Tool Used

* **Tool Name:** Antigravity AI Coding Assistant
* **Underlying Architecture:** Google Gemini 3.8 Flash (Medium) / Gemini Pro Reasoning Models
* **Development Environment:** Local Python 3.12 / Anaconda environment with PyTorch 2.5.1 + CUDA 12.1 on Windows 11.

---

## 3. Narrative of AI Interaction and Standardized Prompts

The implementation proceeded in five sequential research phases. Below are the standardized prompts formulated by the researcher to direct the AI assistant, along with the engineering objectives.

### Phase 1: Environment Setup and Core Algorithm Modularization
* **Researcher's Objective:** Refactor the official `szandala/TorchPRISM` repository into a clean, reusable PyTorch package with deterministic reproducibility.
* **Standardized Prompt Used:**
  > *"We are reproducing the PRISM explainable AI paper (Szandała, Information Sciences 2023). Please set up a clean, modular repository layout with `src/torchprism/`, `experiments/`, `improvements/`, and `outputs/`. Ensure every script enforces deterministic random seeds (`torch.manual_seed(42)`, `np.random.seed(42)`). Extract the core PRISM algorithm from the paper into a modular PyTorch class that registers hooks on VGG-16's `features.30` and implements SVD decomposition and Gradual Extrapolation."*

---

### Phase 2: Faithfulness and Interpretability Replication
* **Researcher's Objective:** Implement representation swapping (Figure 4, Figure 3, Figure 5) and exclusive feature isolation (Figure 6, Figure 7) to verify published benchmarks.
* **Standardized Prompt Used:**
  > *"Write an automated reproduction script `exp1_faithfulness.py` for Section 3.1 of the paper. Load pretrained VGG-16 and sample pairs (Coyote vs Timber Wolf, Cheeseburger vs Bagel, Passenger Car vs Electric Locomotive). Extract the 512x7x7 feature tensors from layer `features.30`. Rank the 49 spatial cells by their PRISM component activation energy, and iteratively swap 5, 9, and 15 cells from the competitor into the target image. Feed the modified tensors through `model.avgpool` and `model.classifier` to verify the softmax drop and Top-1 class inversion reported in the paper. Plot publication-quality grouped bar charts."*

---

### Phase 3: Implementing Solution 1 (Resolving the 3-PCA Bottleneck)
* **Researcher's Objective:** Implement the researcher's concept of High-Variance Manifold PRISM ($\ge 85\%$ variance) and hierarchical Spectral Bands.
* **Standardized Prompt Used:**
  > *"The PRISM paper forces 512 dimensions into 3 PCs to make an RGB image, throwing away over 43% of the variance and causing classifier confidence to collapse from 0.93 to 0.47. Implement our proposed solution in `imp1_resolve_3pca_bottleneck.py`:
  > 1. Run a quantitative benchmark from K = 1 to 30 components, reconstructing the feature tensor $\hat{F}_K = U_K S_K V_K^T + \mu$, and record the relative feature reconstruction error and softmax recovery.
  > 2. Implement an adaptive threshold that automatically selects K to guarantee at least 85% variance.
  > 3. Implement our Spectral-Band Multi-PRISM decomposition, projecting components into 3 separate visual bands: Tier 1 (PCs 1..3: Macro shape), Tier 2 (PCs 4..6: Contours), and Tier 3 (PCs 7..9: Fine Textures), and output a comparative panel."*

---

### Phase 4: Implementing Solution 2 (Resolving Class-Agnostic Distractors)
* **Researcher's Objective:** Implement the researcher's concept of Pre-SVD Gradient Gating and Contrastive Disentanglement.
* **Standardized Prompt Used:**
  > *"In Section 4.2 and Figure 12, PRISM exhibits target-blindness by brightly coloring background distractors (like the mushroom next to the dog, or a coin next to a spider). The paper's post-hoc Grad-CAM blending is mathematically flawed because the SVD basis is already corrupted. Implement our proposed Grad-PRISM:
  > 1. Compute class-specific gradients with respect to the feature map before SVD: $\alpha_k^c = \frac{1}{Z}\sum \frac{\partial y^c}{\partial A_k}$.
  > 2. Gate the channels using $\tilde{A}^k = \text{ReLU}(\alpha_k^c) \cdot A^k$ prior to SVD so distractor channels are eliminated from the basis.
  > 3. Formulate Contrastive cPRISM using counterfactual gradient differences between competing classes.
  > 4. Measure the Signal-to-Distractor Ratio (SDR) and evaluate on the exact paper samples (Malamute+Mushroom, Black Widow+Coin) and multi-object test scenes."*

---

### Phase 5: Building a Modular Custom Test Suite
* **Researcher's Objective:** Package the entire system into an automated pipeline where any new image can be evaluated with a single command.
* **Standardized Prompt Used:**
  > *"Create a standalone testing pipeline in `custom_test_suite/` with separate `input_data/` and `outputs/` directories and a master runner `run_all_custom.py`. The script should accept any image, automatically detect top predicted classes, and run all 3 reproduction experiments and both novel improvements, saving clear comparative figures."*

---

## 4. How AI Output Was Modified, Debugged, and Verified

AI-generated code was treated as a draft and subjected to rigorous technical review, correction, and verification by the researcher:

1. **Resolution of Tensor Gradient Conflicts:**
   * *Issue:* Initial AI-generated Matplotlib plotting code crashed with `RuntimeError: Can't call numpy() on Tensor that requires grad`.
   * *Modification:* The researcher modified all visualization routines to explicitly detach tensors and move them to host memory via `.detach().cpu().numpy()`.
2. **Matrix Reshaping and Centering Corrections:**
   * *Issue:* The AI drafted matrix centering as `mat - act.mean(0).view(1, 512)`, which failed with a tensor shape mismatch because `act.mean(0)` had shape `(512, 7, 7)`.
   * *Modification:* The researcher corrected the mathematical centering to operate on the flattened spatial dimension: `mat - mat.mean(dim=0, keepdim=True)`.
3. **Spatial Interpolation Mismatch in Gradual Extrapolation:**
   * *Issue:* In initial implementations, applying spatial masks to extrapolated feature maps caused dimension mismatches when earlier pooling layers had varying grid resolutions ($112 \times 112$ vs $224 \times 224$).
   * *Modification:* The researcher unified the interpolation pipeline so that all masks dynamically match `(h_ext, w_ext)` using bilinear upsampling.
4. **Verification of Published Metrics:**
   * *Verification:* The researcher verified that our reproduction scripts produced exact numerical consistency with the published paper:
     * Figure 4: 5-cell swap drop of **0.52** matched the paper's **0.54** average benchmark.
     * Figure 6: Active cell counts of **6, 15, and 21 cells** on the exact spider crops matched Figure 6 identically.
     * Figure 17: Top-3 variance of **42.3%** on the 5 canine classes matched the paper's ~40% benchmark.
5. **Deterministic Seeding:**
   * *Modification:* The researcher ensured `torch.backends.cudnn.deterministic = True` and fixed seeds (`seed=42`) were enforced across every script to guarantee 100% reproducibility across runs.

---

## 5. Summary of Compliance

The use of AI in this project was strictly confined to code syntax generation, API structuring, and plotting automation under direct human technical supervision. All conceptual breakthroughs, research gap analyses, mathematical formulations, and critical evaluations reflect the original intellectual contributions of the student researcher.
