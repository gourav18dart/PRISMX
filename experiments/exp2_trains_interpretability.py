"""
Script: exp2_trains_interpretability.py
Description: Runs Experiment 2 (Interpretability: PRISM vs Grad-CAM)
on the exact Passenger Car and Electric Locomotive images from the paper.
Recreates the 2x3 comparison showing:
1. Input Image
2. 7x7 Chequered Grad-CAM
3. 7x7 PRISM Exclusive Features (Quantized Set Difference)
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

# Deterministic seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CUSTOM_OUT_DIR = os.path.join(BASE_DIR, "custom_test_suite", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CUSTOM_OUT_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_train_batch():
    car_path = os.path.join(DATA_DIR, "passenger_car.png")
    loco_path = os.path.join(DATA_DIR, "electric_locomotive.png")
    
    img1 = Image.open(car_path).convert("RGB")
    img2 = Image.open(loco_path).convert("RGB")
    
    t1 = preprocess(img1).unsqueeze(0)
    t2 = preprocess(img2).unsqueeze(0)
    batch = torch.cat([t1, t2], dim=0).to(device)
    return [img1, img2], batch

def compute_gradcam_7x7(model, batch, target_classes):
    model.eval()
    conv_outputs = []
    conv_gradients = []
    
    def fwd(m, inp, out): conv_outputs.append(out)
    def bwd(m, gin, gout): conv_gradients.append(gout[0])
        
    hf = model.features[30].register_forward_hook(fwd)
    hb = model.features[30].register_full_backward_hook(bwd)
    
    cams = []
    for i in range(len(batch)):
        model.zero_grad()
        conv_outputs.clear(); conv_gradients.clear()
        logits = model(batch[i:i+1])
        logits[0, target_classes[i]].backward()
        
        act = conv_outputs[0][0]
        grad = conv_gradients[0][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam.detach().cpu().numpy())
        
    hf.remove(); hb.remove()
    return cams

def compute_exclusive_prism_7x7(model, batch, quant_step=0.5):
    with torch.no_grad():
        feats = model.features(batch) # [2, 512, 7, 7]
        
    b, c, h, w = feats.shape
    matrix = feats.permute(0, 2, 3, 1).reshape(-1, c)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    u, s, v = torch.svd(centered)
    
    raw_pcs = (u[:, :3] @ torch.diag(s[:3])).view(b, h, w, 3).permute(0, 3, 1, 2)
    
    scaled = (raw_pcs - raw_pcs.mean()) / (raw_pcs.std() + 1e-8)
    scaled = scaled.clamp(-1.0, 1.0)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
    
    # 0.5 quantization matching PRISM._quantize
    quant_maps = quant_step * torch.round(scaled / quant_step)
    quant_maps = quant_maps.permute(0, 2, 3, 1) # [2, 7, 7, 3]
    
    image_colors = []
    for img in quant_maps:
        c_set = set()
        for row in img:
            for p in row:
                c_set.add(tuple(p.cpu().numpy().round(2).tolist()))
        image_colors.append(c_set)
        
    common_colors = set.intersection(*image_colors)
    all_colors = set(chain.from_iterable(image_colors))
    exclusive_colors = all_colors - common_colors
    
    exclusive_maps = quant_maps.clone()
    for i, img in enumerate(exclusive_maps):
        for r in range(h):
            for col in range(w):
                color_tuple = tuple(img[r, col].cpu().numpy().round(2).tolist())
                if color_tuple not in exclusive_colors:
                    exclusive_maps[i, r, col] = 0.0
                    
    return exclusive_maps.cpu().numpy()

def main():
    print("=" * 75)
    print("EXPERIMENT 2: INTERPRETABILITY INSIGHT ON TRAIN SAMPLES")
    print("=" * 75)
    
    raw_images, batch = load_train_batch()
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    target_classes = [705, 547] # Passenger Car (705), Electric Locomotive (547)
    
    print("\n1. Computing 7x7 Chequered Grad-CAM...")
    cams = compute_gradcam_7x7(model, batch, target_classes)
    
    print("2. Computing 7x7 PRISM Exclusive Features (Set Difference)...")
    exclusive_prism = compute_exclusive_prism_7x7(model, batch)
    
    names = ["Passenger Car", "Electric Locomotive"]
    for i, name in enumerate(names):
        active_cells = (exclusive_prism[i].sum(axis=-1) > 0).sum()
        print(f"  {name}: {active_cells} active exclusive cells identified.")
        
    # Plotting exact 2x3 layout matching the paper's figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    col_headers = [
        "Input Image",
        "Grad-CAM (7x7 Saliency)",
        "PRISM Exclusive Features (7x7)"
    ]
    
    for c in range(3):
        axes[0, c].set_title(col_headers[c], fontsize=13, fontweight='bold', pad=12)
        
    for r in range(2):
        # Col 0: Input image
        axes[r, 0].imshow(raw_images[r].resize((224, 224)))
        axes[r, 0].set_ylabel(names[r], fontsize=13, fontweight='bold', labelpad=10)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values(): spine.set_linewidth(1.5)
        
        # Col 1: 7x7 Chequered Grad-CAM
        axes[r, 1].imshow(cams[r], cmap='jet', interpolation='nearest')
        axes[r, 1].set_xticks([])
        axes[r, 1].set_yticks([])
        for spine in axes[r, 1].spines.values(): spine.set_linewidth(1.5)
        
        # Col 2: 7x7 PRISM Exclusive
        axes[r, 2].imshow(exclusive_prism[r], interpolation='nearest')
        axes[r, 2].set_xticks([])
        axes[r, 2].set_yticks([])
        for spine in axes[r, 2].spines.values(): spine.set_linewidth(1.5)
        
    plt.suptitle("Paper Reproduction: Interpretability Insight via PRISM Exclusive Features vs. Grad-CAM",
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_p1 = os.path.join(OUTPUT_DIR, "reproduced_trains_interpretability_cam_vs_prism.png")
    save_p2 = os.path.join(CUSTOM_OUT_DIR, "custom_exp2_trains_cam_vs_prism.png")
    plt.savefig(save_p1, bbox_inches='tight')
    plt.savefig(save_p2, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Saved figures to:")
    print(f"  1. {save_p1}")
    print(f"  2. {save_p2}")

if __name__ == "__main__":
    main()
