import time
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import PanopticPerceptionEngine
from dataset import BDD100KDataset, panoptic_collate_fn
from utils import calculate_seg_metrics, visualize_results

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_imgs = "./datasets/val/images"
    test_yolo = "./datasets/val/labels"
    test_masks = "./datasets/bdd100k/drivable_area_annotations/val"

    test_dataset = BDD100KDataset(test_imgs, test_masks, test_yolo)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, pin_memory=True, collate_fn=panoptic_collate_fn)

    model = PanopticPerceptionEngine().to(device)
    if os.path.exists('panoptic_best.pth'):
        checkpoint = torch.load('panoptic_best.pth', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded weights successfully.")

    model.eval()
    total_iou = 0.0
    total_acc = 0.0
    total_time = 0.0
    samples_tested = 0
    latencies = []
    dummy = torch.randn(1, 3, 384, 640).to(device)

    for _ in range(10):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(dummy)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Benchmarking")
        for idx, (imgs, seg_masks, yolo_targets, has_seg) in enumerate(loop):
            imgs = imgs.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, non_blocking=True)

            start = time.time()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                seg_pred, yolo_cls, yolo_reg = model(imgs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.time()
            step_time = (end - start)
            total_time += step_time
            latencies.append((step_time / imgs.size(0)) * 1000)
            samples_tested += imgs.size(0)

            iou, acc = calculate_seg_metrics(seg_pred, seg_masks)
            total_iou += iou * imgs.size(0)
            total_acc += acc * imgs.size(0)

            if idx == 0:
                visualize_results(imgs, seg_masks, seg_pred)

    avg_latency = (total_time / samples_tested) * 1000
    fps = 1.0 / (total_time / samples_tested)

    print(f"\nSegmentation mIoU: {total_iou / samples_tested:.4f}")
    print(f"Segmentation Pixel Accuracy: {total_acc / samples_tested:.4f}")
    print(f"Inference Latency: {avg_latency:.2f} ms per frame")
    print(f"Throughput: {fps:.2f} FPS")

    if torch.cuda.is_available():
        print(f"Peak VRAM Usage: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

    plt.figure(figsize=(10, 5))
    plt.hist(latencies, bins=50, color='blue', alpha=0.7)
    plt.title('Inference Latency Distribution')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('latency_distribution.png')
    plt.show()

if __name__ == "__main__":
    test_model()
