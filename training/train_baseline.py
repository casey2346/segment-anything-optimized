"""
train_baseline.py (Advanced)

Author: Your Name (you@example.com)
License: MIT

This script trains the Segment Anything Model (SAM) from scratch or checkpoint with:
- AMP mixed precision
- Early stopping
- WandB experiment tracking
- YAML-based CLI config
- DDP-ready structure

Run example:
    python train_baseline.py --config configs/sam_config.yaml
"""

import os
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb
from torchinfo import summary

from models.sam_original import SegmentAnythingModel
from data.custom_loader import get_custom_dataloader
from data.coco_loader import get_coco_dataloader

DATASET_MAP = {
    "custom": get_custom_dataloader,
    "coco": get_coco_dataloader
}

def compute_dice(preds, targets, epsilon=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2. * intersection + epsilon) / (union + epsilon)).mean().item()

def compute_miou(preds, targets, epsilon=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + epsilon) / (union + epsilon)).mean().item()

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="🔁 Training"):
        images, masks = images.to(device), masks.to(device).float()
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / (len(dataloader) + 1e-8)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_scores, miou_scores = [], []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="🧪 Validating"):
            images, masks = images.to(device), masks.to(device).float()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item()
            dice_scores.append(compute_dice(outputs, masks))
            miou_scores.append(compute_miou(outputs, masks))
    return val_loss / (len(dataloader) + 1e-8), sum(dice_scores)/len(dice_scores), sum(miou_scores)/len(miou_scores)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

    wandb.init(project=cfg['project'], config=cfg, name=cfg.get("run_name", None))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    scaler = GradScaler()

    # Data
    get_loader = DATASET_MAP[cfg['dataset']]
    train_loader = get_loader(
        os.path.join(cfg['data_dir'], "images"),
        os.path.join(cfg['data_dir'], "masks"),
        cfg['batch_size'], True, cfg['num_workers'], cfg['image_size']
    )
    val_loader = get_loader(
        os.path.join(cfg['data_dir'], "images"),
        os.path.join(cfg['data_dir'], "masks"),
        cfg['batch_size'], False, cfg['num_workers'], cfg['image_size']
    )

    # Model
    model = SegmentAnythingModel(cfg['encoder'], cfg['decoder']).to(device)
    summary(model, input_size=(cfg['batch_size'], 3, cfg['image_size'], cfg['image_size']))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    start_epoch, best_dice, patience_counter = 0, 0.0, 0

    if cfg.get('resume'):
        start_epoch = load_checkpoint(model, optimizer, cfg['resume'])
        _, best_dice, _ = validate(model, val_loader, criterion, device)

    for epoch in range(start_epoch, cfg['epochs']):
        print(f"\n📆 Epoch {epoch + 1}/{cfg['epochs']}")
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, val_miou = validate(model, val_loader, criterion, device)

        print(f"📈 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | mIoU: {val_miou:.4f}")
        print(f"⏱️ Time: {time.time() - start_time:.2f}s")

        wandb.log({
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Dice/val": val_dice,
            "mIoU/val": val_miou,
            "epoch": epoch + 1
        })

        save_checkpoint(model, optimizer, epoch + 1, os.path.join(cfg['checkpoint_dir'], f"sam_epoch_{epoch + 1}.pth"))

        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, os.path.join(cfg['checkpoint_dir'], "best_model.pth"))
            torch.save(model.state_dict(), os.path.join(cfg['checkpoint_dir'], "deploy_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.get('patience', 5):
                print("❌ Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
