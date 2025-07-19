# evaluate.py

import torch  

def evaluate_model(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            preds = (outputs > 0.5).float()
            correct += (preds == masks).sum().item()
            total += masks.numel()
    return correct / total  # e.g., accuracy, Dice, mIoU
