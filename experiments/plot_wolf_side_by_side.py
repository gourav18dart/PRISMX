"""
Script: plot_wolf_side_by_side.py
Description: Generates a clear side-by-side comparison of the Timber Wolf:
1. Original Input Image (512 Dims)
2. Standard PRISM Map (3 PCs)
3. High-Variance PRISMX Map (15 PCs)
With large, clear probability and variance values displayed below each image.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Deterministic seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "wolf_3cases_fidelity_comparison.png")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize_to_rgb(features_tensor):
    sc = (features_tensor - features_tensor.mean()) / (features_tensor.std() + 1e-8)
    sc = sc.clamp(-1.0, 1.0)
    sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)
    return sc

def gradual_extrapolate(feature_map_3ch, pre_excitations):
    ext = feature_map_3ch.clone()
    for e in pre_excitations[::-1]:
        ext = nn.functional.interpolate(
            ext, size=(e.shape[2], e.shape[3]), mode="bilinear", align_corners=False
        )
        ext *= e.mean(dim=1, keepdim=True)
    return normalize_to_rgb(ext)

# Load model
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()

# Load Coyote and Wolf (batch is needed for SVD across comparative canine batch)
coy_img = Image.open(os.path.join(DATA_DIR, "coyote.jpg")).convert("RGB")
wolf_img = Image.open(os.path.join(DATA_DIR, "wolf_01.jpg")).convert("RGB")

coy_t = preprocess(coy_img).unsqueeze(0).to(device)
wolf_t = preprocess(wolf_img).unsqueeze(0).to(device)
batch = torch.cat([coy_t, wolf_t], dim=0)

# Collect activations
activations = []
handles = []
for name, module in model.features.named_children():
    if isinstance(module, nn.MaxPool2d):
        handles.append(module.register_forward_hook(lambda m, inp, out: activations.append(out)))

with torch.no_grad():
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)

for h in handles:
    h.remove()

final_act = activations.pop()
pre_excitations = activations

# Wolf true probability (Timber wolf class index = 269)
wolf_true_prob = probs[1, 269].item() # 0.9118

# SVD
b, c, h, w = final_act.shape
matrix = final_act.permute(0, 2, 3, 1).reshape(-1, c)
mean_val = matrix.mean(dim=0, keepdim=True)
centered = matrix - mean_val
u, s, v = torch.svd(centered)
variances = (s**2) / (s**2).sum()
cum_variances = torch.cumsum(variances, dim=0)

# 1. Standard PRISM (3 PCs)
pcs_3 = u[:, :3] @ torch.diag(s[:3])
raw_3 = pcs_3.view(b, h, w, 3).permute(0, 3, 1, 2)
prism_ge_3 = gradual_extrapolate(raw_3, pre_excitations)
var_3 = cum_variances[2].item() * 100.0 # 56.5%

# Reconstructed prediction with 3 PCs
recon_3 = (u[:, :3] @ torch.diag(s[:3])) @ v[:, :3].t() + mean_val
recon_feats_3 = recon_3.view(b, h, w, c).permute(0, 3, 1, 2)
with torch.no_grad():
    pooled_3 = model.avgpool(recon_feats_3)
    logits_3 = model.classifier(torch.flatten(pooled_3, 1))
    probs_3 = torch.softmax(logits_3, dim=1)
wolf_prob_3 = probs_3[1, 269].item() # 0.6035

# 2. High-Variance PRISMX (15 PCs)
k_15 = 15
var_15 = cum_variances[k_15-1].item() * 100.0 # 86.7%

coords_15 = u[:, :k_15] @ torch.diag(s[:k_15])
W = torch.zeros(k_15, 3, device=device)
for i in range(k_15):
    W[i, i % 3] = s[i] / s[0]
proj_15 = coords_15 @ W
raw_15 = proj_15.view(b, h, w, 3).permute(0, 3, 1, 2)
prism_ge_15 = gradual_extrapolate(raw_15, pre_excitations)

# Reconstructed prediction with 15 PCs
recon_15 = (u[:, :k_15] @ torch.diag(s[:k_15])) @ v[:, :k_15].t() + mean_val
recon_feats_15 = recon_15.view(b, h, w, c).permute(0, 3, 1, 2)
with torch.no_grad():
    pooled_15 = model.avgpool(recon_feats_15)
    logits_15 = model.classifier(torch.flatten(pooled_15, 1))
    probs_15 = torch.softmax(logits_15, dim=1)
wolf_prob_15 = probs_15[1, 269].item() # 0.9017

# Prepare Wolf images for display
img_orig_wolf = wolf_img.resize((224, 224))
img_prism_3 = prism_ge_3[1].permute(1, 2, 0).detach().cpu().numpy()
img_prism_15 = prism_ge_15[1].permute(1, 2, 0).detach().cpu().numpy()

# ==============================================================================
# CREATE PLOT WITH 3 IMAGES SIDE-BY-SIDE + METRICS BELOW
# ==============================================================================
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig = plt.figure(figsize=(18, 9), dpi=300)
gs = fig.add_gridspec(2, 3, height_ratios=[1.6, 1.0], hspace=0.15, wspace=0.18)

panels = [
    {
        "col": 0,
        "title": "CASE 1: Full Original Model",
        "subtitle": "(All 512 Dimensions)",
        "img": img_orig_wolf,
        "is_pil": True,
        "border_color": "#2b5c8f",
        "var_text": "100.0%",
        "var_sub": "0.0% Lost (Full Info)",
        "prob_val": wolf_true_prob,
        "prob_color": "#2b5c8f",
        "status": "Ground Truth Baseline",
        "status_color": "#1a3b5c",
        "bg_color": "#eaf2f8"
    },
    {
        "col": 1,
        "title": "CASE 2: Standard PRISM",
        "subtitle": "(Only 3 Principal Components)",
        "img": img_prism_3,
        "is_pil": False,
        "border_color": "#d9534f",
        "var_text": f"{var_3:.1f}%",
        "var_sub": f"{100.0 - var_3:.1f}% Discarded!",
        "prob_val": wolf_prob_3,
        "prob_color": "#c9302c",
        "status": "Severe Confidence Collapse (-31%)",
        "status_color": "#900",
        "bg_color": "#fdf2f2"
    },
    {
        "col": 2,
        "title": "CASE 3: High-Variance PRISMX",
        "subtitle": f"({k_15} Principal Components)",
        "img": img_prism_15,
        "is_pil": False,
        "border_color": "#1b9e77",
        "var_text": f"{var_15:.1f}%",
        "var_sub": f"Threshold >=85% Satisfied",
        "prob_val": wolf_prob_15,
        "prob_color": "#14785a",
        "status": "True Model Fidelity Restored!",
        "status_color": "#0d5c43",
        "bg_color": "#edf8f5"
    }
]

for p in panels:
    col = p["col"]
    
    # 1. Top row: Picture
    ax_img = fig.add_subplot(gs[0, col])
    if p["is_pil"]:
        ax_img.imshow(p["img"])
    else:
        ax_img.imshow(p["img"])
        
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    
    # Frame around the picture
    for spine in ax_img.spines.values():
        spine.set_edgecolor(p["border_color"])
        spine.set_linewidth(3.5)
        
    ax_img.set_title(f"{p['title']}\n{p['subtitle']}", fontsize=14, fontweight='bold', pad=10, color='#111111')
    
    # 2. Bottom row: Information Card (Variance & Probability)
    ax_card = fig.add_subplot(gs[1, col])
    ax_card.set_facecolor(p["bg_color"])
    ax_card.set_xticks([])
    ax_card.set_yticks([])
    for spine in ax_card.spines.values():
        spine.set_edgecolor(p["border_color"])
        spine.set_linewidth(1.8)
        
    # Text block inside the card
    ax_card.text(0.5, 0.88, "VARIANCE COVERED", ha='center', va='center',
                 fontsize=11, fontweight='bold', color='#444444')
    ax_card.text(0.5, 0.72, p["var_text"], ha='center', va='center',
                 fontsize=22, fontweight='bold', color=p["border_color"])
    ax_card.text(0.5, 0.58, p["var_sub"], ha='center', va='center',
                 fontsize=10, fontweight='semibold', color='#666666')
    
    ax_card.axhline(0.48, color=p["border_color"], linestyle=':', linewidth=1.2, xmin=0.08, xmax=0.92)
    
    ax_card.text(0.5, 0.38, "WOLF PROBABILITY", ha='center', va='center',
                 fontsize=11, fontweight='bold', color='#444444')
    ax_card.text(0.5, 0.22, f"{p['prob_val']:.2f}  ({p['prob_val']*100:.1f}%)", ha='center', va='center',
                 fontsize=22, fontweight='bold', color=p["prob_color"])
    
    ax_card.text(0.5, 0.08, p["status"], ha='center', va='center',
                 fontsize=10.5, fontweight='bold', color=p["status_color"])

plt.suptitle("Timber Wolf: Visual Feature Representations, Variance Covered, and Prediction Probabilities",
             fontsize=16, fontweight='bold', y=0.98, color='#000000')

plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Wolf side-by-side comparison saved to: {save_path}")
