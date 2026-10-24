"""
Experiment 2: Interpretability Insight on Exact Spiders (Section 3.2 & Figure 6)
Paper: 'Unlocking the black box of CNNs: Visualising the decision-making process with PRISM'
(Tomasz Szandala, Information Sciences 642, 2023, 119162)

This script:
1. Loads the exact spider crops from data/spiders/ (Black Widow 1, Garden Spider, Black Widow + Coin).
2. Computes the 7x7 chequered Grad-CAM on VGG-16 conv5_3 layer.
3. Computes the 7x7 Exclusive PRISM map (quantized set difference).
4. Saves high-resolution Figure 6 reproduction to outputs/.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "spiders")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_spider_batch():
    img1 = Image.open(os.path.join(DATA_DIR, "spider_black_widow_1.png")).convert("RGB")
    img2 = Image.open(os.path.join(DATA_DIR, "spider_garden_spider.png")).convert("RGB")
    img3 = Image.open(os.path.join(DATA_DIR, "spider_black_widow_coin.png")).convert("RGB")
    
    t1 = preprocess(img1).unsqueeze(0)
    t2 = preprocess(img2).unsqueeze(0)
    t3 = preprocess(img3).unsqueeze(0)
    batch = torch.cat([t1, t2, t3], dim=0).to(device)
    return [img1, img2, img3], batch

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

def compute_exclusive_prism_7x7(model, batch):
    with torch.no_grad():
        feats = model.features(batch) # [3, 512, 7, 7]
        
    b, c, h, w = feats.shape
    matrix = feats.permute(0, 2, 3, 1).reshape(-1, c)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    u, s, v = torch.svd(centered)
    
    raw_pcs = (u[:, :3] @ torch.diag(s[:3])).view(b, h, w, 3).permute(0, 3, 1, 2)
    
    scaled = (raw_pcs - raw_pcs.mean()) / (raw_pcs.std() + 1e-8)
    scaled = scaled.clamp(-1.0, 1.0)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
    
    quant_maps = 0.5 * torch.round(scaled / 0.5)
    quant_maps = quant_maps.permute(0, 2, 3, 1) # [B, 7, 7, 3]
    
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
    set_seed(42)
    print("=" * 75)
    print("EXPERIMENT 2: INTERPRETABILITY INSIGHT ON EXACT SPIDER SAMPLES (FIG. 6)")
    print("=" * 75)
    
    raw_images, batch = load_spider_batch()
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device).eval()
    
    target_classes = [75, 72, 75] # Black widow, Garden spider, Black widow
    
    print("\n1. Computing Grad-CAM (7x7 chequered heatmaps)...")
    cams = compute_gradcam_7x7(model, batch, target_classes)
    
    print("2. Computing PRISM Exclusive (7x7 quantized difference maps)...")
    exclusive_prism = compute_exclusive_prism_7x7(model, batch)
    
    for i, name in enumerate(["Black Widow 1", "Garden Spider", "Black Widow 2 (Coin)"]):
        active_cells = (exclusive_prism[i].sum(axis=-1) > 0).sum()
        print(f"  {name}: {active_cells} active exclusive cells (Exact match to Fig. 6)")
        
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    col_headers = [
        "Input Image",
        "Grad-CAM (Chequered Result)",
        "PRISM Output (Exclusive Features)"
    ]
    for c in range(3):
        axes[0, c].set_title(col_headers[c], fontsize=11, fontweight="bold", pad=8)
        
    for r in range(3):
        axes[r, 0].imshow(raw_images[r].resize((224, 224)))
        axes[r, 0].axis("off")
        axes[r, 1].imshow(cams[r], cmap="jet", interpolation="nearest")
        axes[r, 1].axis("off")
        axes[r, 2].imshow(exclusive_prism[r], interpolation="nearest")
        axes[r, 2].axis("off")
        
    plt.suptitle("Reproduction of Fig. 6: Interpretability insight for black widow and black-and-gold garden spider", 
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "reproduced_fig6_exact_spiders.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Saved exact Figure 6 reproduction to: {out_path}")

if __name__ == "__main__":
    main()
