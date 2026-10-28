import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_giou(pred_boxes, target_boxes):
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.float()
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
    pred_area = torch.clamp((pred_x2 - pred_x1) * (pred_y2 - pred_y1), min=0.0)
    tgt_area = torch.clamp((tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1), min=0.0)
    union = pred_area + tgt_area - inter_area + 1e-4
    iou = inter_area / union
    c_x1 = torch.min(pred_x1, tgt_x1)
    c_y1 = torch.min(pred_y1, tgt_y1)
    c_x2 = torch.max(pred_x2, tgt_x2)
    c_y2 = torch.max(pred_y2, tgt_y2)
    c_area = torch.clamp((c_x2 - c_x1) * (c_y2 - c_y1), min=0.0) + 1e-4
    return iou - (c_area - union) / c_area

class PanopticLoss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.seg_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.num_classes = num_classes
        self.strides = [8, 16, 32]

    def focal_loss(self, preds, targets, alpha=0.25, gamma=2.0):
        preds = preds.float()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.sum()

    def compute_yolo_loss(self, cls_preds, reg_preds, targets):
        device = cls_preds[0].device
        batch_size = cls_preds[0].shape[0]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        batch_targets_device = []
        total_targets = 0
        for b in range(batch_size):
            bt = targets[b].to(device)
            batch_targets_device.append(bt)
            total_targets += len(bt)

        if total_targets == 0:
            return total_cls_loss + total_reg_loss

        for level, stride in enumerate(self.strides):
            B, C, H, W = cls_preds[level].shape
            batched_tgt_cls = torch.zeros((B, self.num_classes, H, W), device=device)
            level_pred_boxes = []
            level_true_boxes = []

            for b in range(batch_size):
                bt = batch_targets_device[b]
                if len(bt) == 0:
                    continue

                grid_x = (bt[:, 1] * W).long()
                grid_y = (bt[:, 2] * H).long()
                grid_x = torch.clamp(grid_x, 0, W - 1)
                grid_y = torch.clamp(grid_y, 0, H - 1)

                classes = bt[:, 0].long()
                batched_tgt_cls[b, classes, grid_y, grid_x] = 1.0

                pred_boxes = torch.sigmoid(reg_preds[level][b][:, grid_y, grid_x]).t()
                true_boxes = bt[:, 1:5]

                level_pred_boxes.append(pred_boxes)
                level_true_boxes.append(true_boxes)

            total_cls_loss += self.focal_loss(cls_preds[level], batched_tgt_cls)

            if len(level_pred_boxes) > 0:
                all_pred_boxes = torch.cat(level_pred_boxes, dim=0)
                all_true_boxes = torch.cat(level_true_boxes, dim=0)
                giou = compute_giou(all_pred_boxes, all_true_boxes)
                total_reg_loss += (1.0 - giou).sum()

        num_targets = max(1, total_targets)
        return (total_cls_loss / num_targets) + (total_reg_loss * 2.0 / num_targets)

    def forward(self, preds, targets, has_seg):
        seg_pred, yolo_cls_preds, yolo_reg_preds = preds
        seg_target, yolo_target = targets
        raw_seg_loss = self.seg_criterion(seg_pred.float(), seg_target.float())
        has_seg_mask = has_seg.view(-1, 1, 1, 1).to(seg_pred.device).float()
        masked_seg_loss = (raw_seg_loss * has_seg_mask).mean()
        loss_yolo = self.compute_yolo_loss(yolo_cls_preds, yolo_reg_preds, yolo_target)
        total_loss = masked_seg_loss + loss_yolo
        return total_loss, masked_seg_loss, loss_yolo

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def calculate_seg_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    pixel_acc = (pred == target).float().mean(dim=(2, 3))
    return iou.mean().item(), pixel_acc.mean().item()

def visualize_results(imgs, gt_masks, pred_masks, num_samples=4):
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(min(num_samples, imgs.size(0))):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        gt = gt_masks[i].squeeze().cpu().numpy()
        pred = (torch.sigmoid(pred_masks[i]).squeeze() > 0.5).float().cpu().numpy()

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(gt, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('evaluation_visuals.png')
    plt.show()
