"""
train_finetune.py (Ultra Enhanced Version)

Supports:
✅ DDP with torchrun
✅ AMP + EarlyStopping
✅ Resume from checkpoint
✅ WandB / TensorBoard
✅ Git hash logging
✅ Checkpoint compress + optional upload
✅ Config validation via YAML + JSON schema

Usage:
    torchrun --nproc_per_node=2 train_finetune.py --config configs/finetune.yaml
"""

import os
import time
import csv
import yaml
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from jsonschema import validate
from utils.notify import notify_slack, notify_email
from utils.schema import CONFIG_SCHEMA
from models.sam_original import SegmentAnythingModel
from data.custom_loader import get_custom_dataloader
from accelerate import Accelerator
import wandb
import shutil
import tarfile

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return "unknown"

def compress_and_upload(path, upload_fn=None):
    archive = path + ".tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(path, arcname=os.path.basename(path))
    if upload_fn:
        upload_fn(archive)

def compute_dice(preds, targets, eps=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2. * intersection + eps) / (union + eps)).mean().item()

def compute_miou(preds, targets, eps=1e-6):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    validate(instance=cfg, schema=CONFIG_SCHEMA)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator()
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    git_hash = get_git_hash()
    scaler = GradScaler()

    writer = SummaryWriter(log_dir=os.path.join(cfg['checkpoint_dir'], 'runs'))
    if cfg.get("use_wandb"):
        wandb.init(project=cfg.get("wandb_project", "SAM_Finetune"), config=cfg)

    encoder_cfg = cfg['encoder']
    decoder_cfg = cfg['decoder']
    model = SegmentAnythingModel(encoder_cfg, decoder_cfg)

    if cfg.get('freeze_encoder'):
        for p in model.encoder.parameters():
            p.requires_grad = False

    if cfg.get('resume') and os.path.exists(cfg['resume']):
        accelerator.print(f"🔄 Resuming from {cfg['resume']}")
        model.load_state_dict(torch.load(cfg['resume'], map_location='cpu'))
    else:
        model.load_state_dict(torch.load(cfg['pretrained_path'], map_location='cpu'))

    train_loader = get_custom_dataloader(cfg['data_dir'], "train", cfg['batch_size'], cfg['num_workers'], cfg['image_size'])
    val_loader = get_custom_dataloader(cfg['data_dir'], "val", cfg['batch_size'], cfg['num_workers'], cfg['image_size'])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    metrics_path = os.path.join(cfg['checkpoint_dir'], "metrics.csv")
    with open(metrics_path, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "dice", "miou", "git_hash"])

    best_dice, patience_counter = 0.0, 0

    for epoch in range(cfg['epochs']):
        accelerator.print(f"\n📅 Epoch {epoch+1}/{cfg['epochs']}")
        start = time.time()

        model.train()
        total_loss = 0.0
        for images, masks in tqdm(train_loader, disable=not accelerator.is_main_process):
            images, masks = images.to(device), masks.to(device).float()
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        train_loss = total_loss / (len(train_loader) + 1e-8)

        model.eval()
        val_loss, dice_scores, miou_scores = 0.0, [], []
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device).float()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice_scores.append(compute_dice(outputs, masks))
                miou_scores.append(compute_miou(outputs, masks))
                if i == 0 and epoch % 5 == 0 and accelerator.is_main_process:
                    save_image(images[0].cpu(), f"{cfg['checkpoint_dir']}/ep{epoch}_input.png")
                    save_image(torch.sigmoid(outputs[0]).cpu(), f"{cfg['checkpoint_dir']}/ep{epoch}_pred.png")

        val_loss /= (len(val_loader) + 1e-8)
        val_dice = sum(dice_scores) / len(dice_scores)
        val_miou = sum(miou_scores) / len(miou_scores)

        if accelerator.is_main_process:
            print(f"📉 Train: {train_loss:.4f} | Val: {val_loss:.4f} | Dice: {val_dice:.4f} | mIoU: {val_miou:.4f}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Dice/val", val_dice, epoch)
            writer.add_scalar("mIoU/val", val_miou, epoch)
            if cfg.get("use_wandb"): wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice, "val_miou": val_miou, "epoch": epoch})
            with open(metrics_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch + 1, train_loss, val_loss, val_dice, val_miou, git_hash])
            ckpt_path = os.path.join(cfg['checkpoint_dir'], f"model_ep{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

            if val_dice > best_dice:
                best_dice = val_dice
                patience_counter = 0
                best_path = os.path.join(cfg['checkpoint_dir'], "best_model.pth")
                torch.save(model.state_dict(), best_path)
                compress_and_upload(best_path, upload_fn=None)  # Replace with uploader function if needed
            else:
                patience_counter += 1
                if patience_counter >= cfg['patience']:
                    print("⛔ Early stopping.")
                    break

        accelerator.print(f"⏱️ Time: {time.time() - start:.2f}s")

    if accelerator.is_main_process:
        if cfg.get("notify_slack"): notify_slack("🎉 Finetuning complete")
        if cfg.get("notify_email"): notify_email("SAM Finetune", "Training finished successfully")

if __name__ == "__main__":
    main()
