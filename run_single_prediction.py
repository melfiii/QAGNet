#!/usr/bin/env python
"""
QAGNet Single Image Prediction Visualization
Loads an image, runs inference, and visualizes predictions
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.structures import Instances
from mask2former.maskformer_model import MaskFormer
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper

# Configure logger
setup_logger(name="fvcore")

print("=" * 80)
print("QAGNet Single Image Prediction Visualization")
print("=" * 80)

# Load config
print("\n[1/5] Loading configuration...")
cfg = get_cfg()
cfg.merge_from_file('./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
cfg.MODEL.DEVICE = 'cpu'
cfg.EVALUATION.DATAPATH = './datasets/'
cfg.EVALUATION.DATASET = 'sifr'
cfg.EVALUATION.MODEL_DIR = './PreTrained_Models/'
print("✓ Config loaded")

# Load model
print("\n[2/5] Loading model...")
model = MaskFormer(cfg)
model.eval()
checkpointer = DetectionCheckpointer(model)
checkpointer.load(os.path.join(cfg.EVALUATION.MODEL_DIR, 'SIFR_ResNet50.pth'))
print("✓ Model loaded successfully")
print(f"   Architecture: {model.__class__.__name__}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load test image (Image ID 63)
print("\n[3/5] Loading test image...")
dataset_path = './datasets/SIFR/'
image_id = 63
img_path = os.path.join(dataset_path, f'{image_id}.jpg')

if not os.path.exists(img_path):
    print(f"ERROR: Image not found at {img_path}")
    print(f"Available images in {dataset_path}:")
    imgs = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')][:10]
    print(f"   {imgs}")
    sys.exit(1)

original_image = cv2.imread(img_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
print(f"✓ Image loaded: {image_id}")
print(f"   Shape: {original_image_rgb.shape}")

# Prepare input
print("\n[4/5] Preparing input...")
mapper = SIFRdataDatasetMapper(cfg, is_train=False, augmentations=None)
dataset_dict = {
    'file_name': img_path,
    'image_id': image_id,
    'height': original_image_rgb.shape[0],
    'width': original_image_rgb.shape[1],
}
inputs = mapper(dataset_dict)
inputs = [inputs]
print("✓ Input prepared")

# Run inference
print("\n[5/5] Running inference...")
with torch.no_grad():
    predictions = model(inputs)

instances = predictions[-1]["instances"].to("cpu")
print(f"✓ Inference complete")
print(f"   Total instances detected: {len(instances)}")

# Create predictions table
threshold = cfg.EVALUATION.RESULT_THRESHOLD
print(f"\n{'='*80}")
print(f"PREDICTIONS TABLE (threshold={threshold})")
print(f"{'='*80}\n")

data = []
for idx in range(len(instances)):
    score = instances[idx].scores.item()
    rank = instances[idx].pred_rank.item()
    mask_shape = instances[idx].pred_masks.shape
    selected = '✓' if score > threshold else '✗'
    
    data.append({
        'Obj': idx,
        'Confidence': f'{score:.4f}',
        'Ranking': f'{rank:.4f}',
        'Maske (H×W)': f'{mask_shape[1]}×{mask_shape[2]}',
        'Sel': selected
    })

df = pd.DataFrame(data)
print(df.to_string(index=False))
print(f"\n✓ Total: {len(instances)} instances")
print(f"✓ Selected (Score > {threshold}): {sum(1 for x in data if x['Sel'] == '✓')} instances")

# Visualize
print(f"\n{'='*80}")
print("VISUALIZATION")
print(f"{'='*80}\n")

selected_indices = [idx for idx in range(len(instances)) 
                   if instances[idx].scores.item() > threshold]

if len(selected_indices) == 0:
    print("⚠ No instances with Score > threshold found")
else:
    print(f"Creating visualizations for {len(selected_indices)} selected instances...\n")
    
    # Comprehensive view with top-3
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'QAGNet Predictions - Image ID {image_id}', fontsize=16, fontweight='bold')

    # Show original image
    axes[0, 0].imshow(original_image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Show top-3 high-confidence predictions
    top_3 = sorted(enumerate(instances), 
                   key=lambda x: x[1].scores.item(), 
                   reverse=True)[:3]

    for i, (obj_idx, instance) in enumerate(top_3):
        score = instance.scores.item()
        rank = instance.pred_rank.item()
        mask = instance.pred_masks[0].cpu().numpy()
        
        # First row: masks
        im = axes[0, i+1].imshow(mask, cmap='hot')
        axes[0, i+1].set_title(f'Obj {obj_idx}: Score={score:.3f}', fontsize=11, fontweight='bold')
        axes[0, i+1].axis('off')
        plt.colorbar(im, ax=axes[0, i+1], fraction=0.046, pad=0.04)
        
        # Second row: mask overlay on image
        masked_image = original_image_rgb.copy().astype(float)
        masked_image = masked_image * 0.5 + mask[:, :, np.newaxis] * 127 * 0.5
        
        axes[1, i+1].imshow(masked_image.astype(np.uint8))
        axes[1, i+1].set_title(f'Rank={rank:.3f}', fontsize=11, fontweight='bold')
        axes[1, i+1].axis('off')

    axes[1, 0].axis('off')

    plt.tight_layout()
    
    # Save figure
    output_path = f'./prediction_viz_{image_id}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    plt.show()

print(f"\n{'='*80}")
print("✓ Complete!")
print(f"{'='*80}")
