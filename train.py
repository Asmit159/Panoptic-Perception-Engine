import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
from model import PanopticPerceptionEngine
from dataset import BDD100KDataset, panoptic_collate_fn
from utils import PanopticLoss, initialize_weights

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} with bfloat16")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Note: Adjust paths for your local machine
    train_imgs = "./datasets/train/images"
    train_yolo = "./datasets/train/labels"
    train_masks = "./datasets/bdd100k/drivable_area_annotations/train"
    val_imgs = "./datasets/val/images"
    val_yolo = "./datasets/val/labels"
    val_masks = "./datasets/bdd100k/drivable_area_annotations/val"

    train_dataset = BDD100KDataset(train_imgs, train_masks, train_yolo)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=4, collate_fn=panoptic_collate_fn
    )

    val_dataset = BDD100KDataset(val_imgs, val_masks, val_yolo)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=8, pin_memory=True, prefetch_factor=4, collate_fn=panoptic_collate_fn
    )

    model = PanopticPerceptionEngine().to(device)
    model.apply(initialize_weights)

    criterion = PanopticLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    epochs = 100
    warmup_epochs = 3
    iters_per_epoch = len(train_loader)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=iters_per_epoch * warmup_epochs)
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=iters_per_epoch * (epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[iters_per_epoch * warmup_epochs])

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

        for imgs, seg_masks, yolo_targets, has_seg in loop:
            imgs = imgs.to(device, non_blocking=True)
            seg_masks = seg_masks.to(device, non_blocking=True)
            has_seg = has_seg.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(imgs)
                loss, l_seg, l_yolo = criterion(preds, (seg_masks, yolo_targets), has_seg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, seg_masks, yolo_targets, has_seg in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                seg_masks = seg_masks.to(device, non_blocking=True)
                has_seg = has_seg.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    preds = model(imgs)
                    loss, _, _ = criterion(preds, (seg_masks, yolo_targets), has_seg)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, 'panoptic_best.pth')
            print("--> Model Saved")

if __name__ == "__main__":
    train_model()
