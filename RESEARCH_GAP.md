# Research Gap Explanation: Theoretical and Empirical Limitations of PRISM

**Paper Reference:** Szandała, T. (2023). *Unlocking the black box of CNNs: Visualising the decision-making process with PRISM.* Information Sciences, 642, 119162.

---

## 1. Context and Original Value Proposition
Explainable Artificial Intelligence (XAI) for Convolutional Neural Networks (CNNs) has long been dominated by Class Activation Mapping (CAM, Grad-CAM) and backpropagation-based saliency methods (DeconvNet, Guided Backpropagation). While CAM methods identify coarse regions of interest, they fail to delineate fine-grained structural sections. 

To overcome this, Szandała proposed **PRISM (Principal Image Sections Mapping)**. PRISM extracts activations from the final convolutional layer (e.g., $512 \times 7 \times 7$ in VGG-16), applies Singular Value Decomposition (SVD), and projects the first 3 principal components into Red, Green, and Blue (RGB) color channels, followed by Gradual Extrapolation (GE) upsampling. This produces color-coded, high-resolution feature sections that reveal distinct visual sub-features (e.g., body contours vs. facial markings).

Despite its initial promise, our in-depth mathematical analysis and empirical reproduction revealed **two fundamental research gaps** that severely limit the trustworthiness and utility of PRISM.

---

## 2. Research Gap 1: The 3-PCA Representation Bottleneck (Severe Information Loss and Decision Collapse)

### The Core Problem
To visualize representations on a 2D monitor, PRISM forcibly compresses the 512-dimensional feature space down to exactly **3 principal components ($K=3$)** to match the 3 color channels of human vision (RGB). 

### Theoretical and Empirical Breakdown
1. **Massive Discard of Learned Information:**
   * SVD scree analysis across standard ImageNet canine classes demonstrates that the first 3 components account for only **56.5%** of feature variance in 2-class sets, dropping to **42.3%** in 5-class sets.
   * Consequently, PRISM permanently discards **43.5% to 57.7%** of the information that the CNN learned and relied upon.
2. **Model Decision Collapse (Loss of Fidelity):**
   * In XAI, an explanation is faithful only if it reflects the true decision logic of the underlying model. When the reconstructed 3-PC feature matrix ($\hat{F}_3 = U_3 S_3 V_3^T + \mu$) is fed into the CNN's decision head (`model.classifier`), the prediction confidence collapses drastically:
     * **Coyote:** Drops from **0.93** (original 512D) to **0.47** (3-PC reconstruction).
     * **Timber Wolf:** Drops from **0.91** (original 512D) to **0.60** (3-PC reconstruction).
   * **The Gap:** Standard PRISM explains an artificial, severely degraded representation where the model's confidence is cut nearly in half, rather than explaining the real, confident neural network.

---

## 3. Research Gap 2: Class-Agnostic Distractor Contamination (Target Blindness)

### The Core Problem
PRISM performs SVD in an entirely **unsupervised, class-agnostic** manner. SVD computes eigenvectors purely based on maximum statistical variance across pixel activations, with no knowledge of what class the neural network is predicting.

### Theoretical and Empirical Breakdown
1. **Component Hijacking by Irrelevant Distractors:**
   * When an image contains multiple objects or background clutter (e.g., the paper's Figure 12 showing an Alaskan Malamute next to a Boletus mushroom, or a Black Widow spider next to a shiny copper coin in Figure 6), high-contrast distractors dominate the activation variance.
   * SVD allocates one or more of its 3 primary color components directly to the distractor. The mushroom and coin are brightly highlighted with dedicated colors, creating the false illusion that the CNN used them to make its prediction.
   * In reality, the CNN predicted Malamute with **97.5% canine confidence** and **0.0000% mushroom confidence**; the network completely ignored the mushroom.
2. **Failure of the Paper's Post-Hoc Workaround:**
   * In Section 4.2, the author acknowledged this flaw and offered a workaround: alpha-blending a Grad-CAM heatmap over the PRISM output (`--saliency grad_cam`).
   * **Why this is theoretically invalid:** The SVD basis decomposition has *already occurred*. Because the distractor hijacked one of the 3 principal components, the target object was already deprived of component capacity. Pasting a coarse $7 \times 7$ Grad-CAM blob on top of a corrupted representation merely conceals the distractor under blur; it does not correct the mathematical basis.

---

## 4. Proposed Architectural Solutions

To resolve these two gaps, we engineered **PRISMX**:
1. **Resolving Gap 1 via High-Variance Manifold PRISM and Spectral Bands:**
   * Implemented an adaptive variance threshold ($\ge 85\%$) that dynamically selects $K$ (e.g., $K=14\text{--}15$), reducing relative feature error from $59.2\%$ to $32.8\%$ and fully restoring prediction confidence ($0.47 \rightarrow 0.90+$).
   * Decomposed the manifold into **3 Spectral Bands** (Tier 1 Macro, Tier 2 Contours, Tier 3 Textures), allowing full multi-dimensional inspection without visual overlap.
2. **Resolving Gap 2 via Class-Attentive Grad-PRISM and Contrastive cPRISM:**
   * Integrated class gradients **prior to SVD** ($\tilde{A}^k = \text{ReLU}(\alpha_k^c) \cdot A^k$). Channels irrelevant to target class $c$ are zeroed out before computing eigenvectors, ensuring that 100% of the SVD basis explains only the target class.
   * Formulated Contrastive PRISM (cPRISM) to isolate counterfactual discriminators between competing classes.
