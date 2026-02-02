#!/usr/bin/env python3
"""
Debug-Skript zum Inspizieren der Model-Predictions
Zeigt Masken, Klassen, Ranking-Scores für erkannte Objekte
"""

import torch
import numpy as np
from detectron2.config import get_cfg
from mask2former.maskformer_model import MaskFormer
from detectron2.checkpoint import DetectionCheckpointer
from mask2former.data.dataset_mappers.sifr_dataset_mapper import SIFRdataDatasetMapper
from mask2former.data.datasets.register_sifr import get_sifr_dicts
from detectron2.data import DatasetFromList, MapDataset
from torch.utils.data import DataLoader
import cv2

# Setup
cfg = get_cfg()
cfg.merge_from_file("./configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml")
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_list(["MODEL.SEM_SEG_HEAD.NUM_CLASSES", 1])

# Model laden
model = MaskFormer(cfg)
model.eval()
model.to(cfg.MODEL.DEVICE)
checkpointer = DetectionCheckpointer(model)
checkpointer.load("./PreTrained_Models/SIFR_ResNet50.pth")

# Erstes Test-Bild laden
sifr_dataset = get_sifr_dicts(root="./datasets/SIFR/", mode="test")
sifr_dataset_list = DatasetFromList(sifr_dataset[:1], copy=False)  # Nur 1 Bild
sifr_dataset_mapped = MapDataset(sifr_dataset_list, SIFRdataDatasetMapper(cfg, False))

dataloader = DataLoader(sifr_dataset_mapped, batch_size=1, shuffle=False, num_workers=0)

# Inference
with torch.no_grad():
    for inputs in dataloader:
        predictions = model(inputs)
        
        print("=" * 80)
        print("PREDICTIONS STRUKTUR")
        print("=" * 80)
        
        # predictions ist eine Liste mit Output von mehreren Decoder-Layern
        print(f"\n1. Anzahl Decoder-Layer Outputs: {len(predictions)}")
        print(f"   (9 Layers mit Deep Supervision + 1 Final Output)")
        
        # Letzter Output (Final)
        final_output = predictions[-1]
        print(f"\n2. Final Output Keys: {final_output.keys()}")
        
        # Instances extrahieren
        if "instances" in final_output:
            instances = final_output["instances"].to("cpu")
            print(f"\n3. INSTANCES (erkannte Objekte):")
            print(f"   - Anzahl erkannte Objekte: {len(instances)}")
            
            # Für jedes Objekt: Score, Maske, Ranking
            for i in range(min(5, len(instances))):  # Zeige erste 5
                print(f"\n   Objekt {i}:")
                print(f"     - Confidence Score: {instances[i].scores:.4f}")
                print(f"     - Maske Shape: {instances[i].pred_masks.shape}")
                print(f"     - Ranking Score: {instances[i].pred_rank:.4f}")
                print(f"     - Maske Pixel (min/max): {instances[i].pred_masks.min():.2f} / {instances[i].pred_masks.max():.2f}")
        
        # Seg Maps (Pixel-Level Predictions)
        if "sem_seg" in final_output:
            print(f"\n4. SEGMENTATION MAP:")
            sem_seg = final_output["sem_seg"]
            print(f"   - Shape: {sem_seg.shape}")
            print(f"   - Values Range: {sem_seg.min():.4f} - {sem_seg.max():.4f}")
        
        print("\n" + "=" * 80)
        print("VISUALISIERUNG")
        print("=" * 80)
        
        # Maske des ersten Objekts speichern
        if "instances" in final_output and len(instances) > 0:
            mask = instances[0].pred_masks[0].numpy()
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite("/tmp/debug_mask_object_0.png", mask_uint8)
            print(f"\n✓ Erste Objekt-Maske gespeichert: /tmp/debug_mask_object_0.png")
            
            # Ranking-basierte Färbung (wie im echten Code)
            all_masks = instances.pred_masks.numpy()
            ranking_scores = instances.pred_rank.numpy()
            
            # Farbzuweisung nach Ranking
            colored_map = np.zeros((instances.image_size[0], instances.image_size[1]), dtype=np.uint8)
            for i, (mask, rank) in enumerate(zip(all_masks, ranking_scores)):
                # Höheres Ranking = hellere Farbe
                color_value = int(255 * (rank / ranking_scores.max())) if ranking_scores.max() > 0 else 0
                colored_map[mask > 0.5] = color_value
            
            cv2.imwrite("/tmp/debug_ranking_visualization.png", colored_map)
            print(f"✓ Ranking-Visualisierung gespeichert: /tmp/debug_ranking_visualization.png")
            print(f"  (dunkel = niedriges Ranking, hell = hohes Ranking)")
        
        break  # Nur erstes Bild
