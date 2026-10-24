"""
Script: extract_and_run_faithfulness_pairs.py
Description: Extracts the exact Cheeseburger, Bagel, Passenger Car, and Electric Locomotive
images from Figures 3 and 5 of the paper, and runs the Faithfulness Representation Swapping
benchmark on both pairs.
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
CUSTOM_INPUT_DIR = os.path.join(BASE_DIR, "custom_test_suite", "input_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CUSTOM_OUT_DIR = os.path.join(BASE_DIR, "custom_test_suite", "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CUSTOM_INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CUSTOM_OUT_DIR, exist_ok=True)

appdata = r"C:\Users\goura\.gemini\antigravity\brain\bf89c72c-def1-4cf9-86e3-c4094459144c\.user_uploaded"
im1 = Image.open(os.path.join(appdata, "media_1792828994700_119cdbd3.png")).convert("RGB")
im2 = Image.open(os.path.join(appdata, "media_1792829005950_656cfa4b.png")).convert("RGB")

# 1. Crop images
cheeseburger_crop = im1.crop((48, 45, 219, 216))
bagel_crop = im1.crop((48, 218, 219, 389))
passenger_car_crop = im2.crop((22, 14, 196, 183))
locomotive_crop = im2.crop((22, 184, 196, 358))

# Save crops to data/samples and custom_test_suite/input_data
crops = {
    "cheeseburger.png": cheeseburger_crop,
    "bagel.png": bagel_crop,
    "passenger_car.png": passenger_car_crop,
    "electric_locomotive.png": locomotive_crop
}

for name, img in crops.items():
    p1 = os.path.join(DATA_DIR, name)
    p2 = os.path.join(CUSTOM_INPUT_DIR, name)
    img.save(p1)
    img.save(p2)
    print(f"Saved: {p1} and {p2}")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Model
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()

def run_swapping_experiment(img_a, img_b, name_a, name_b, class_a_idx, class_b_idx, out_filename, title):
    print("\n" + "="*75)
    print(f"RUNNING FAITHFULNESS SWAPPING: {name_a} vs {name_b}")
    print("="*75)
    
    t_a = preprocess(img_a).unsqueeze(0).to(device)
    t_b = preprocess(img_b).unsqueeze(0).to(device)
    batch = torch.cat([t_a, t_b], dim=0)
    
    activations = []
    handle = model.features[30].register_forward_hook(lambda m, inp, out: activations.append(out))
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
    handle.remove()
    
    p_a_orig = probs[0, class_a_idx].item()
    p_b_orig = probs[1, class_b_idx].item()
    print(f"{name_a} Baseline Probability (Class {class_a_idx}): {p_a_orig:.4f}")
    print(f"{name_b} Baseline Probability (Class {class_b_idx}): {p_b_orig:.4f}")
    
    feat_a = activations[0][0].clone() # (512, 7, 7)
    feat_b = activations[0][1].clone()
    
    # Sort grid cells by activation energy
    importance_a = torch.norm(feat_a, dim=0)
    indices_a = torch.argsort(importance_a.flatten(), descending=True)
    
    importance_b = torch.norm(feat_b, dim=0)
    indices_b = torch.argsort(importance_b.flatten(), descending=True)
    
    swap_steps = [0, 3, 5, 8, 12, 16]
    probs_a_target = []
    probs_a_competitor = []
    
    print(f"\n--- Swapping {name_b} cells INTO {name_a} ---")
    for k in swap_steps:
        swapped = feat_a.clone()
        if k > 0:
            for idx in indices_a[:k]:
                r = idx.item() // 7
                c = idx.item() % 7
                swapped[:, r, c] = feat_b[:, r, c]
                
        with torch.no_grad():
            pooled = model.avgpool(swapped.unsqueeze(0))
            out_logits = model.classifier(torch.flatten(pooled, 1))
            out_probs = torch.softmax(out_logits, dim=1)
            
        p_target = out_probs[0, class_a_idx].item()
        p_comp = out_probs[0, class_b_idx].item()
        probs_a_target.append(p_target)
        probs_a_competitor.append(p_comp)
        
        status = "INVERTED!" if p_comp > p_target else ("Significant Drop" if p_a_orig - p_target > 0.3 else "Normal")
        print(f"Swapped {k:<2} cells | {name_a}: {p_target:.4f} | {name_b}: {p_comp:.4f} | [{status}]")
        
    # Create Publication Plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Left: The two images side by side
    ax1.imshow(np.hstack([np.array(img_a.resize((200, 200))), np.array(img_b.resize((200, 200)))]))
    ax1.set_title(f"Left: {name_a} (Orig: {p_a_orig:.2f})  |  Right: {name_b} (Orig: {p_b_orig:.2f})",
                  fontsize=12, fontweight='bold', pad=12)
    ax1.axis("off")
    
    # Right: Swapping curve
    x = np.arange(len(swap_steps))
    w = 0.35
    bars1 = ax2.bar(x - w/2, probs_a_target, w, label=f"Original Class ({name_a})", color='#2b5c8f', edgecolor='black')
    bars2 = ax2.bar(x + w/2, probs_a_competitor, w, label=f"Donor Class ({name_b})", color='#d9534f', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k} cells" for k in swap_steps], fontsize=11, fontweight='bold')
    ax2.set_ylabel("Softmax Classification Probability", fontsize=12, fontweight='bold')
    ax2.set_title(f"Prediction Inversion on {name_a} via PRISM Feature Swapping", fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    for bar in bars1:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
            
    # Highlight inversion point
    for idx_step, (pt, pc) in enumerate(zip(probs_a_target, probs_a_competitor)):
        if pc > pt:
            ax2.annotate('Prediction\nInverted!', xy=(idx_step + w/2, pc), xytext=(idx_step + w/2, pc + 0.18),
                         arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                         ha='center', fontsize=9, fontweight='bold', color='red',
                         bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6e6", ec="red", lw=1))
            break
            
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_p1 = os.path.join(OUTPUT_DIR, out_filename)
    save_p2 = os.path.join(CUSTOM_OUT_DIR, out_filename)
    plt.savefig(save_p1, bbox_inches='tight')
    plt.savefig(save_p2, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to: {save_p1} and {save_p2}")
    return probs_a_target, probs_a_competitor

# Check ImageNet class indices:
# cheeseburger = 933, bagel = 931
# passenger_car = 705, electric_locomotive = 547
print("\n--- Verifying Exact Classes with VGG-16 ---")
tb_burger = preprocess(cheeseburger_crop).unsqueeze(0).to(device)
tb_bagel = preprocess(bagel_crop).unsqueeze(0).to(device)
tb_train = preprocess(passenger_car_crop).unsqueeze(0).to(device)
tb_loco = preprocess(locomotive_crop).unsqueeze(0).to(device)

with torch.no_grad():
    p_b = model(tb_burger).softmax(1)
    p_bg = model(tb_bagel).softmax(1)
    p_tr = model(tb_train).softmax(1)
    p_lc = model(tb_loco).softmax(1)

print(f"Cheeseburger crop top-1 class: {p_b.argmax().item()} (Prob: {p_b.max().item():.4f})")
print(f"Bagel crop top-1 class:        {p_bg.argmax().item()} (Prob: {p_bg.max().item():.4f})")
print(f"Passenger Car top-1 class:     {p_tr.argmax().item()} (Prob: {p_tr.max().item():.4f})")
print(f"Electric Loco top-1 class:     {p_lc.argmax().item()} (Prob: {p_lc.max().item():.4f})")

# Run Experiment for Pair 1: Cheeseburger vs Bagel (Figure 3)
run_swapping_experiment(
    cheeseburger_crop, bagel_crop,
    "Cheeseburger", "Bagel",
    class_a_idx=933, class_b_idx=931,
    out_filename="reproduced_fig3_cheeseburger_bagel_swapping.png",
    title="Paper Fig. 3 Reproduction: Faithfulness via Representation Swapping (Cheeseburger <-> Bagel)"
)

# Run Experiment for Pair 2: Passenger Car vs Electric Locomotive (Figure 5)
run_swapping_experiment(
    passenger_car_crop, locomotive_crop,
    "Passenger Car", "Electric Locomotive",
    class_a_idx=705, class_b_idx=547,
    out_filename="reproduced_fig5_trains_swapping.png",
    title="Paper Fig. 5 Reproduction: Faithfulness via Representation Swapping (Passenger Car <-> Electric Locomotive)"
)
