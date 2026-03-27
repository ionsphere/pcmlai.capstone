import sys
import argparse
import json
import time
import copy
from pathlib import Path
from typing import Sequence, Optional, Dict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.vision import MultiTaskClothingModel, MultiTaskLoss, count_parameters


DEFAULT_IMAGE_SIZE = 380
DEMO_EPOCHS = 2
DEFAULT_CONFIG = {
    "model": {
        "backbone": "efficientnet_b4",
        "num_clothing_types": 1,
        "condition_scale": 10,
        "condition_mode": "regression",
        "pretrained": True,
        "freeze_backbone": False,
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "min_lr": 1e-6,
        "early_stopping_patience": 10,
        "mixed_precision": True,
    },
}


def build_config(quick_test: bool = False, demo: bool = False) -> Dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if demo:
        config["training"]["epochs"] = DEMO_EPOCHS
    if quick_test:
        config["training"]["epochs"] = DEMO_EPOCHS
        config["training"]["batch_size"] = 8
    return config


class ConditionDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, base_dir: str = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else Path(csv_path).parent
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.resolve_image_path(row['image_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        condition = torch.tensor(row['condition_score'], dtype=torch.float32)
        type_label = torch.tensor(0, dtype=torch.long)
        
        return image, type_label, condition

    def resolve_image_path(self, image_path_value) -> Path:
        raw_path = Path(str(image_path_value))
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                self.base_dir / raw_path,
                project_root / raw_path,
                raw_path,
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

def get_transforms(image_size: int = DEFAULT_IMAGE_SIZE, augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_condition_loss = 0.0
    all_predictions = []
    all_targets = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, type_labels, condition_targets) in enumerate(pbar):
        images = images.to(device)
        type_labels = type_labels.to(device)
        condition_targets = condition_targets.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            type_logits, condition_pred = model(images)
            loss, loss_type, loss_condition = criterion(
                type_logits, condition_pred, type_labels, condition_targets
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_condition_loss += loss_condition.item()
        
        all_predictions.extend(condition_pred.squeeze().detach().cpu().numpy())
        all_targets.extend(condition_targets.detach().cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cond_loss': f"{loss_condition.item():.4f}"
        })
        
        if writer:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/ConditionLoss', loss_condition.item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_condition_loss = total_condition_loss / len(dataloader)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'loss': avg_loss,
        'condition_loss': avg_condition_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_condition_loss = 0.0
    all_predictions = []
    all_targets = []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for images, type_labels, condition_targets in pbar:
        images = images.to(device)
        type_labels = type_labels.to(device)
        condition_targets = condition_targets.to(device)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            type_logits, condition_pred = model(images)
            loss, loss_type, loss_condition = criterion(
                type_logits, condition_pred, type_labels, condition_targets
            )
        
        total_loss += loss.item()
        total_condition_loss += loss_condition.item()
        
        all_predictions.extend(condition_pred.squeeze().detach().cpu().numpy())
        all_targets.extend(condition_targets.detach().cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cond_loss': f"{loss_condition.item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_condition_loss = total_condition_loss / len(dataloader)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    return {
        'loss': avg_loss,
        'condition_loss': avg_condition_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def train_model(config: Dict, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_config = config["training"]
    model_config = config["model"]
    use_amp = bool(training_config["mixed_precision"])
    image_size = DEFAULT_IMAGE_SIZE
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["num_workers"])
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    epochs = int(training_config["epochs"])
    min_lr = float(training_config["min_lr"])
    early_stopping_patience = int(training_config["early_stopping_patience"])
    pin_memory = device.type == 'cuda'

    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print(f"Loading datasets...")
    train_dataset = ConditionDataset(
        args.train_csv,
        transform=get_transforms(image_size, augment=True),
        base_dir=args.data_dir
    )
    val_dataset = ConditionDataset(
        args.val_csv,
        transform=get_transforms(image_size, augment=False),
        base_dir=args.data_dir
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Creating model...")
    model = MultiTaskClothingModel(
        backbone_name=model_config['backbone'],
        num_clothing_types=model_config['num_clothing_types'],
        condition_scale=model_config['condition_scale'],
        condition_mode=model_config['condition_mode'],
        pretrained=model_config['pretrained'],
        freeze_backbone=model_config['freeze_backbone']
    )
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Backbone: {model_config['backbone']}")
    print(f"Parameters: {params['trainable']:,} trainable, {params['frozen']:,} frozen")
    
    criterion = MultiTaskLoss(condition_mode=model_config['condition_mode'])
    criterion = criterion.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and use_amp))
    writer = SummaryWriter(log_dir)
    
    print(f"Starting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {early_stopping_patience}")
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer
        )
        
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch} Results:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R^2: {train_metrics['r2']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, R^2: {val_metrics['r2']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/MAE', train_metrics['mae'], epoch)
        writer.add_scalar('Train/RMSE', train_metrics['rmse'], epoch)
        writer.add_scalar('Train/R2', train_metrics['r2'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/MAE', val_metrics['mae'], epoch)
        writer.add_scalar('Val/RMSE', val_metrics['rmse'], epoch)
        writer.add_scalar('Val/R2', val_metrics['r2'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': current_lr
        })
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        torch.save(checkpoint, checkpoint_dir / "last.pth")
        
        if val_metrics['mae'] < best_val_mae:
            print(f"New best MAE: {val_metrics['mae']:.4f} (previous: {best_val_mae:.4f})")
            best_val_mae = val_metrics['mae']
            torch.save(checkpoint, checkpoint_dir / "best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    elapsed_time = time.time() - start_time
    print(f"TRAINING COMPLETE")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Best val MAE: {best_val_mae:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        history_serializable = []
        for h in history:
            h_copy = {
                'epoch': h['epoch'],
                'lr': h['lr'],
                'train': {k: float(v) for k, v in h['train'].items()},
                'val': {k: float(v) for k, v in h['val'].items()}
            }
            history_serializable.append(h_copy)
        json.dump(history_serializable, f, indent=2)
    print(f"History saved to: {history_path}")
    writer.close()
    return model, history


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train condition assessment model")
    parser.add_argument('--train-csv', type=str, 
                        default='data/processed/condition_assessment/train.csv',
                        help='Training CSV file')
    parser.add_argument('--val-csv', type=str,
                        default='data/processed/condition_assessment/val.csv',
                        help='Validation CSV file')
    parser.add_argument('--data-dir', type=str,
                        default='data/processed/condition_assessment',
                        help='Base directory for image paths')
    parser.add_argument('--output-dir', type=str,
                        default='models/condition_assessment',
                        help='Output directory for model and logs')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (2 epochs, small batch)')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode (2 epochs on the real dataset)')
    
    args = parser.parse_args(argv)
    
    print("CONDITION ASSESSMENT TRAINING")
    config = build_config(quick_test=args.quick_test, demo=args.demo)
    if args.quick_test:
        print("Quick test mode enabled")
    if args.demo:
        print(f"Demo mode enabled ({DEMO_EPOCHS} epochs on the real dataset)")
    
    model, history = train_model(config, args)
    
    print(f"Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print(f"View training progress:")
    print(f"tensorboard --logdir {args.output_dir}/logs")


if __name__ == '__main__':
    main()
