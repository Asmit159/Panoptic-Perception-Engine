import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class BDD100KDataset(Dataset):
    def __init__(self, images_dir, masks_dir, yolo_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.yolo_dir = Path(yolo_dir)
        self.samples = []
        self.labels = []

        try:
            yolo_files = set(os.listdir(yolo_dir))
            img_files = set(os.listdir(images_dir))
            mask_files = set(os.listdir(masks_dir))
        except FileNotFoundError as e:
            print(f"Path Error: {e}")
            return

        print(f"Caching labels into RAM from {yolo_dir}...")
        for label_name in tqdm(yolo_files, desc="Loading Labels"):
            if not label_name.endswith('.txt'): continue
            base_name = label_name.replace('.txt', '')
            img_name = f"{base_name}.jpg"
            mask_name = f"{base_name}.png"

            if img_name in img_files:
                has_seg = 1.0 if mask_name in mask_files else 0.0
                mask_path = str(self.masks_dir / mask_name) if has_seg else None

                self.samples.append({
                    'image': str(self.images_dir / img_name),
                    'mask': mask_path,
                    'has_seg': has_seg
                })
                boxes = []
                with open(self.yolo_dir / label_name, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            boxes.append([float(p) for p in parts])
                self.labels.append(torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 5)))

        print(f"Loaded {len(self.samples)} total images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 384))

        if sample['has_seg'] == 1.0:
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (640, 384), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((384, 640), dtype=np.float32)

        # Instantly fetch pre-loaded RAM targets instead of hitting the disk
        boxes = self.labels[idx]

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        has_seg_tensor = torch.tensor(sample['has_seg'], dtype=torch.float32)

        return img_tensor, mask_tensor, boxes, has_seg_tensor

def panoptic_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    yolo_targets = [item[2] for item in batch]
    has_seg = torch.stack([item[3] for item in batch])
    return imgs, masks, yolo_targets, has_seg
