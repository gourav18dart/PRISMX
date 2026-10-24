"""
Script: test_bidirectional_swapping.py
Description: Runs bidirectional swapping for both Figure 3 (Burgers) and Figure 5 (Trains),
using PRISM-guided discriminative cells (highest singular component energy on object).
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CUSTOM_OUT_DIR = os.path.join(BASE_DIR, "custom_test_suite", "outputs")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()

def evaluate_pair(name_a, name_b, file_a, file_b, class_a, class_b, out_filename, title):
    img_a = Image.open(os.path.join(DATA_DIR, file_a)).convert("RGB")
    img_b = Image.open(os.path.join(DATA_DIR, file_b)).convert("RGB")
    
    t_a = preprocess(img_a).unsqueeze(0).to(device)
    t_b = preprocess(img_b).unsqueeze(0).to(device)
    batch = torch.cat([t_a, t_b], dim=0)
    
    activations = []
    handle = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    handle.remove()
    
    feat_a = activations[0][0].clone() # (512, 7, 7)
    feat_b = activations[0][1].clone()
    
    # Run SVD across batch to get PRISM component energy per cell
    mat = activations[0].permute(0, 2, 3, 1).reshape(-1, 512)
    u, s, v = torch.svd(mat - mat.mean(0))
    # Top 3 PCs energy per spatial position
    pcs = (u[:, :3] @ torch.diag(s[:3])).norm(dim=1).view(2, 7, 7)
    prism_energy_a = pcs[0]
    prism_energy_b = pcs[1]
    
    idx_a = torch.argsort(prism_energy_a.flatten(), descending=True)
    idx_b = torch.argsort(prism_energy_b.flatten(), descending=True)
    
    swap_counts = [0, 2, 4, 6, 8, 12, 16]
    
    # Direction 1: Swap B into A
    res_a_t = []
    res_a_c = []
    for k in swap_counts:
        sw = feat_a.clone()
        for i in idx_a[:k]:
            r, c = i.item() // 7, i.item() % 7
            sw[:, r, c] = feat_b[:, r, c]
        with torch.no_grad():
            p = model.classifier(torch.flatten(model.avgpool(sw.unsqueeze(0)), 1)).softmax(1)
        res_a_t.append(p[0, class_a].item())
        res_a_c.append(p[0, class_b].item())
        
    # Direction 2: Swap A into B
    res_b_t = []
    res_b_c = []
    for k in swap_counts:
        sw = feat_b.clone()
        for i in idx_b[:k]:
            r, c = i.item() // 7, i.item() % 7
            sw[:, r, c] = feat_a[:, r, c]
        with torch.no_grad():
            p = model.classifier(torch.flatten(model.avgpool(sw.unsqueeze(0)), 1)).softmax(1)
        res_b_t.append(p[0, class_b].item())
        res_b_c.append(p[0, class_a].item())
        
    print(f"\n=================================================================")
    print(f"{title}")
    print(f"=================================================================")
    print(f"Direction 1: Swapping {name_b} into {name_a}")
    for k, pt, pc in zip(swap_counts, res_a_t, res_a_c):
        print(f"  {k:2d} cells: {name_a}={pt:.4f} | {name_b}={pc:.4f} {'[INVERTED]' if pc > pt else ''}")
        
    print(f"\nDirection 2: Swapping {name_a} into {name_b}")
    for k, pt, pc in zip(swap_counts, res_b_t, res_b_c):
        print(f"  {k:2d} cells: {name_b}={pt:.4f} | {name_a}={pc:.4f} {'[INVERTED]' if pc > pt else ''}")
        
    # Plotting 2-row figure: Row 1 = Dir 1, Row 2 = Dir 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300, gridspec_kw={'width_ratios': [1, 2.2]})
    
    # Row 1: Image A -> B
    axes[0, 0].imshow(img_a.resize((200, 200)))
    axes[0, 0].set_title(f"Target: {name_a}\nBaseline Prob: {res_a_t[0]:.2f}", fontsize=11, fontweight='bold')
    axes[0, 0].axis("off")
    
    x = np.arange(len(swap_counts))
    w = 0.35
    axes[0, 1].bar(x - w/2, res_a_t, w, label=f"Original Class ({name_a})", color='#2b5c8f', edgecolor='black')
    axes[0, 1].bar(x + w/2, res_a_c, w, label=f"Donor Class ({name_b})", color='#d9534f', edgecolor='black')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"{k} cells" for k in swap_counts], fontsize=10, fontweight='bold')
    axes[0, 1].set_ylabel("Probability", fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f"Swapping {name_b} cells INTO {name_a}", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1.15)
    axes[0, 1].legend(loc='upper right', fontsize=9.5)
    axes[0, 1].grid(axis='y', linestyle=':', alpha=0.7)
    
    for idx_step, (pt, pc) in enumerate(zip(res_a_t, res_a_c)):
        if pc > pt:
            axes[0, 1].annotate('Prediction\nInverted!', xy=(idx_step + w/2, pc), xytext=(idx_step + w/2, pc + 0.16),
                                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                                ha='center', fontsize=9, fontweight='bold', color='red',
                                bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="red", lw=1))
            break

    # Row 2: Image B -> A
    axes[1, 0].imshow(img_b.resize((200, 200)))
    axes[1, 0].set_title(f"Target: {name_b}\nBaseline Prob: {res_b_t[0]:.2f}", fontsize=11, fontweight='bold')
    axes[1, 0].axis("off")
    
    axes[1, 1].bar(x - w/2, res_b_t, w, label=f"Original Class ({name_b})", color='#2b5c8f', edgecolor='black')
    axes[1, 1].bar(x + w/2, res_b_c, w, label=f"Donor Class ({name_a})", color='#d9534f', edgecolor='black')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{k} cells" for k in swap_counts], fontsize=10, fontweight='bold')
    axes[1, 1].set_ylabel("Probability", fontsize=11, fontweight='bold')
    axes[1, 1].set_title(f"Swapping {name_a} cells INTO {name_b}", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].legend(loc='upper right', fontsize=9.5)
    axes[1, 1].grid(axis='y', linestyle=':', alpha=0.7)
    
    for idx_step, (pt, pc) in enumerate(zip(res_b_t, res_b_c)):
        if pc > pt:
            axes[1, 1].annotate('Prediction\nInverted!', xy=(idx_step + w/2, pc), xytext=(idx_step + w/2, pc + 0.16),
                                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                                ha='center', fontsize=9, fontweight='bold', color='red',
                                bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="red", lw=1))
            break
            
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    sp1 = os.path.join(OUTPUT_DIR, out_filename)
    sp2 = os.path.join(CUSTOM_OUT_DIR, out_filename)
    plt.savefig(sp1, bbox_inches='tight')
    plt.savefig(sp2, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sp1}")

# Run both paper figures
evaluate_pair(
    "Cheeseburger", "Bagel",
    "cheeseburger.png", "bagel.png",
    class_a=933, class_b=931,
    out_filename="reproduced_fig3_cheeseburger_bagel_swapping.png",
    title="Paper Fig. 3 Reproduction: Faithfulness via Representation Swapping (Cheeseburger <-> Bagel)"
)

evaluate_pair(
    "Passenger Car", "Electric Locomotive",
    "passenger_car.png", "electric_locomotive.png",
    class_a=705, class_b=547,
    out_filename="reproduced_fig5_trains_swapping.png",
    title="Paper Fig. 5 Reproduction: Faithfulness via Representation Swapping (Passenger Car <-> Electric Locomotive)"
)
