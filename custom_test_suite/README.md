# PRISMX Custom Test Suite

A modular testing pipeline designed to run all **3 reproduction experiments** and **2 novel improvements** on any arbitrary input images.

---

## Directory Structure

```text
custom_test_suite/
├── input_data/         <--- Drop any new images here (.jpg, .jpeg, .png)
├── outputs/            <--- All generated results and comparison figures are saved here
├── run_all_custom.py   <--- Master runner script
└── README.md           <--- Instructions
```

---

## How to Test on New Images

### Step 1: Add Your Images
Simply drop any images you want to evaluate into:
```text
custom_test_suite/input_data/
```
*(The suite automatically discovers all `.jpg`, `.jpeg`, and `.png` files in this directory).*

### Step 2: Run the Pipeline
Open your terminal and run:

```bash
python custom_test_suite/run_all_custom.py
```

You can also specify custom input and output directories if desired:
```bash
python custom_test_suite/run_all_custom.py --input_dir path/to/my_images --output_dir path/to/my_outputs
```

---

## What the Pipeline Executes & Output Figures

| Test | Script Function | Generated Output Figure | What It Evaluates |
| :--- | :--- | :--- | :--- |
| **Experiment 1** | `run_exp1()` | `custom_exp1_faithfulness_swapping.png` | **Faithfulness:** Swaps discriminative feature cells between Image 1 and Image 2 to measure prediction inversion and softmax drops. |
| **Experiment 2** | `run_exp2()` | `custom_exp2_interpretability_cam_vs_prism.png` | **Interpretability:** Compares coarse Grad-CAM heatmaps against high-resolution PRISM feature sections across the input images. |
| **Experiment 3** | `run_exp3()` | `custom_exp3_svd_variance_scree.png` | **Variance Distribution:** Generates the SVD Scree plot, measuring how much information is retained across 1 to 30 principal components. |
| **Improvement 1** | `run_imp1()` | `custom_imp1_3pca_fidelity_and_spectral_bands.png` | **3-PCA Bottleneck Solution:** Evaluates decision fidelity recovery and extracts 3 Spectral Bands (Macro, Contours, Textures) + High-Variance Composite. |
| **Improvement 2** | `run_imp2()` | `custom_imp2_class_attentive_grad_prism.png` | **Distractor Resolution:** Compares Standard PRISM against Class-Attentive Grad-PRISM, proving distractor suppression via gradient gating. |
