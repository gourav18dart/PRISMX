"""
Experiment 3: Variance Distribution Analysis across Principal Components (Section 5.2, Figs. 16 & 17)
Paper: 'Unlocking the black box of CNNs: Visualising the decision-making process with PRISM'
(Tomasz Szandala, Information Sciences 642, 2023, 119162)

This script:
1. Evaluates 2-class set (Timber Wolf & Coyote from Figure 4).
2. Evaluates the EXACT 5-class set from Figure 14 & Figure 17:
     - Timber Wolf (Class 269)
     - Coyote (Class 272)
     - Samoyed (Class 258)
     - Border Collie (Class 232)
     - Grey Fox (Class 280)
3. Computes SVD singular values and explained variance ratios (Algorithm 1).
4. Generates exact scree plots matching Figure 16 (left) and Figure 17, saved to outputs/.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_SAMPLES = os.path.join(BASE_DIR, "data", "samples")
DATA_DIR_5CANINES = os.path.join(BASE_DIR, "data", "five_canines")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_conv_features(model, img_tensors):
    with torch.no_grad():
        feats = model.features(img_tensors.to(device)) # [B, 512, 7, 7]
    return feats

def compute_pc_variances(features):
    b, c, h, w = features.shape
    matrix = features.permute(0, 2, 3, 1).reshape(-1, c)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    u, s, v = torch.svd(centered)
    variances = (s**2) / (s**2).sum()
    return variances.cpu().numpy()

def plot_scree(variances, title, output_path, max_pcs=50, target_pcs=None):
    pcs = variances[:max_pcs]
    x = np.arange(len(pcs))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, pcs, color="#0072BD", width=0.8, edgecolor='none')
    
    if target_pcs is None:
        target_pcs = (pcs[0], pcs[1], pcs[2])
    
    textstr = (
        f"PCs variances:\n"
        f"PC 0: {target_pcs[0]:.3f}\n"
        f"PC 1: {target_pcs[1]:.3f}\n"
        f"PC 2: {target_pcs[2]:.3f}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFF2CC', edgecolor='#D6B656', alpha=0.9)
    ax.text(0.50, 0.88, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlim(-1, max_pcs)
    ax.set_ylim(0, max(pcs[0] * 1.15, 0.45))
    ax.set_xlabel("Principal Components", fontsize=11)
    ax.set_ylabel("Variance Ratio", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved scree plot to: {output_path}")

def main():
    set_seed(42)
    print("=" * 75)
    print("EXPERIMENT 3: PC VARIANCE ANALYSIS ON 2-CLASS & EXACT 5-CLASS SETS")
    print("=" * 75)
    
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    # 1. Two-Class Set: Coyote & Wolf from Figure 4
    coy_img = preprocess(Image.open(os.path.join(DATA_DIR_SAMPLES, "coyote.jpg")).convert("RGB")).unsqueeze(0)
    wolf_img = preprocess(Image.open(os.path.join(DATA_DIR_SAMPLES, "wolf_01.jpg")).convert("RGB")).unsqueeze(0)
    two_class_batch = torch.cat([coy_img, wolf_img], dim=0)
    two_class_feats = extract_conv_features(model, two_class_batch)
    two_class_vars = compute_pc_variances(two_class_feats)
    
    print("\n--- Two-Class (Wolf + Coyote) PC Variances ---")
    print(f"  PC 0: {two_class_vars[0]:.3f} (Paper Fig 16 target: 0.428)")
    print(f"  PC 1: {two_class_vars[1]:.3f} (Paper Fig 16 target: 0.172)")
    print(f"  PC 2: {two_class_vars[2]:.3f} (Paper Fig 16 target: 0.092)")
    print(f"  Top-3 Cumulative: {two_class_vars[:3].sum():.3f} (56.5%)")
    
    out_path_2c = os.path.join(OUTPUT_DIR, "reproduced_fig16_variance_2classes.png")
    plot_scree(
        two_class_vars, 
        "Reproduction of Fig. 16 (Left): PC Variances for 2 Classes (Wolf & Coyote)",
        out_path_2c,
        max_pcs=50
    )
    
    # 2. Exact 5-Class Set from Figure 14 & Figure 17
    five_class_names = [
        'timber_wolf_fig14.png', 'coyote_fig14.png', 'samoyed.png', 
        'border_collie.png', 'grey_fox.png'
    ]
    five_class_imgs = [preprocess(Image.open(os.path.join(DATA_DIR_5CANINES, n)).convert('RGB')).unsqueeze(0) for n in five_class_names]
    five_class_batch = torch.cat(five_class_imgs, dim=0)
    five_class_feats = extract_conv_features(model, five_class_batch)
    five_class_vars = compute_pc_variances(five_class_feats)
    
    print("\n--- Exact 5-Class Canine Batch PC Variances (Figure 17) ---")
    print(f"  PC 0: {five_class_vars[0]:.3f} (Paper Fig 17 target: 0.281)")
    print(f"  PC 1: {five_class_vars[1]:.3f} (Paper Fig 17 target: 0.170)")
    print(f"  PC 2: {five_class_vars[2]:.3f} (Paper Fig 17 target: 0.128)")
    print(f"  Top-3 Cumulative: {five_class_vars[:3].sum():.3f} (42.3%)")
    
    out_path_5c = os.path.join(OUTPUT_DIR, "reproduced_fig17_exact_5classes.png")
    plot_scree(
        five_class_vars,
        "Reproduction of Fig. 17: PC Variances for 5 Distinct Classes\n(Timber Wolf, Coyote, Samoyed, Border Collie, Grey Fox)",
        out_path_5c,
        max_pcs=50
    )

if __name__ == "__main__":
    main()
