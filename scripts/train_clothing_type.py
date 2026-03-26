import sys
import json
import argparse
import time
import copy
from pathlib import Path
from typing import Sequence, Optional, Dict


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.models.vision import ModelFactory, count_parameters
from scripts.prepare_clothing_type_dataset import ClothingTypeDataset


DEFAULT_IMAGE_SIZE = 380
DEFAULT_TRAINING_CONFIG = {
    "model": {
        "backbone": "efficientnet_b4",
        "num_clothing_types": 20,
        "condition_scale": 10,
        "condition_mode": "regression",
        "pretrained": True,
        "freeze_backbone": False,
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "scheduler_params": {
            "T_max": 50,
            "eta_min": 1e-6,
        },
        "mixed_precision": True,
        "gradient_clip": 1.0,
        "early_stopping_patience": 10,
        "checkpoint_every_n_epochs": 5,
        "type_weight": None,
        "condition_weight": None,
        "num_workers": 0,
    },
}


def build_config(batch_size: Optional[int] = None, epochs: Optional[int] = None, quick_test: bool = False) -> Dict:
    config = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if epochs is not None:
        config["training"]["num_epochs"] = epochs
        config["training"]["scheduler_params"]["T_max"] = epochs
    if quick_test:
        config["training"]["batch_size"] = 4
        config["training"]["num_epochs"] = 2
        config["training"]["scheduler_params"]["T_max"] = 2
    return config


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    def __init__(self, config: Dict, output_dir: Path, device: str = 'cuda'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu':
            print("CUDA not available, using CPU (this will be slow)")
        else:
            print(f"Using device: {self.device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = ModelFactory.create_model(config)
        self.model = self.model.to(self.device)
        
        params = count_parameters(self.model)
        print(f"Model Parameters:")
        print(f"Total: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,}")
        
        self.criterion = ModelFactory.create_loss(config)
        self.criterion = self.criterion.to(self.device)
        
        train_config = config['training']
        self.optimizer = self.create_optimizer(train_config)
        self.scheduler = self.create_scheduler(train_config)
        self.use_amp = train_config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = train_config.get('early_stopping_patience', 10)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_history = []
        self.val_history = []
    
    def create_optimizer(self, train_config: Dict) -> optim.Optimizer:
        optimizer_name = train_config.get('optimizer', 'AdamW')
        lr = train_config.get('learning_rate', 1e-4)
        weight_decay = train_config.get('weight_decay', 1e-5)
        if optimizer_name == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            momentum = train_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print(f"Optimizer: {optimizer_name} (lr={lr}, wd={weight_decay})")
        return optimizer
    
    def create_scheduler(self, train_config: Dict) -> Optional[optim.lr_scheduler._LRScheduler]:
        scheduler_name = train_config.get('scheduler', 'CosineAnnealingLR')
        if scheduler_name == 'CosineAnnealingLR':
            scheduler_params = train_config.get('scheduler_params', {})
            T_max = scheduler_params.get('T_max', 50)
            eta_min = scheduler_params.get('eta_min', 1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_name is None or scheduler_name == 'None':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        if scheduler:
            print(f"Scheduler: {scheduler_name}")
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        type_losses = AverageMeter()
        type_accs = AverageMeter()
        
        end = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            data_time.update(time.time() - end)
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    type_logits, condition_pred = self.model(images)
                    loss = nn.CrossEntropyLoss()(type_logits, labels)
            else:
                type_logits, condition_pred = self.model(images)
                loss = nn.CrossEntropyLoss()(type_logits, labels)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config['training'].get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                self.optimizer.step()
            
            acc = (type_logits.argmax(dim=1) == labels).float().mean()
            
            losses.update(loss.item(), batch_size)
            type_accs.update(acc.item(), batch_size)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{type_accs.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return {
            'loss': losses.avg,
            'accuracy': type_accs.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        losses = AverageMeter()
        type_accs = AverageMeter()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        type_logits, _ = self.model(images)
                        loss = nn.CrossEntropyLoss()(type_logits, labels)
                else:
                    type_logits, _ = self.model(images)
                    loss = nn.CrossEntropyLoss()(type_logits, labels)
                
                preds = type_logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                losses.update(loss.item(), batch_size)
                type_accs.update(acc.item(), batch_size)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        per_class_acc = {}
        for label in unique_labels:
            mask = all_labels == label
            if mask.sum() > 0:
                acc = (all_preds[mask] == all_labels[mask]).mean()
                per_class_acc[int(label)] = float(acc)
        
        return {
            'loss': losses.avg,
            'accuracy': type_accs.avg,
            'per_class_accuracy': per_class_acc
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model (acc={self.best_val_acc:.4f})")
        
        if epoch % self.config['training'].get('checkpoint_every_n_epochs', 5) == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best val acc: {self.best_val_acc:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        print("STARTING TRAINING")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        start_time = time.time()
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Train/LR', train_metrics['lr'], epoch)
            
            self.train_history.append({
                'epoch': epoch,
                **train_metrics,
                'time': time.time() - epoch_start
            })
            self.val_history.append({
                'epoch': epoch,
                **val_metrics
            })
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} Summary ({epoch_time:.1f}s):")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered (patience={self.early_stopping_patience})")
                break
        
        total_time = time.time() - start_time
        
        print("TRAINING COMPLETE")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best val accuracy: {self.best_val_acc:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
                'best_val_acc': self.best_val_acc,
                'total_time': total_time
            }, f, indent=2)
        
        self.writer.close()


def resolve_image_root(data_dir: Path, mapping: Dict) -> Path:
    candidates = []
    mapping_root = mapping.get('image_root')
    if mapping_root:
        candidates.append(Path(mapping_root))
    candidates.extend([
        project_root / "data" / "deepfashion",
        project_root / "data" / "deepfashion" / "original",
        project_root / "data" / "raw" / "deepfashion" / "original",
        data_dir,
    ])

    train_csv = data_dir / 'train.csv'
    sample_image_path = None
    if train_csv.exists():
        sample_df = pd.read_csv(train_csv, nrows=1)
        if not sample_df.empty and 'image_path' in sample_df.columns:
            sample_image_path = Path(sample_df.iloc[0]['image_path'])

    for candidate in candidates:
        candidate = Path(candidate)
        if sample_image_path is None:
            if candidate.exists():
                return candidate
            continue

        if (candidate / sample_image_path).exists():
            return candidate

    raise FileNotFoundError(
        "Could not resolve the image root for clothing type training. "
        "Re-run dataset preparation or verify data/deepfashion is organized."
    )


def create_transforms(image_size: int = DEFAULT_IMAGE_SIZE, is_train: bool = True):
    if is_train:
        transform_list = [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]
    
    return transforms.Compose(transform_list)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train clothing type classification model")
    parser.add_argument('--data-dir', type=str, default='data/processed/clothing_type',
                       help='Path to prepared dataset')
    parser.add_argument('--output-dir', type=str, default='models/clothing_type',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test run with small dataset and few epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args(argv)
    config = build_config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        quick_test=args.quick_test,
    )
    if args.quick_test:
        print("Quick test mode enabled")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    mapping_file = data_dir / 'category_mapping.json'
    
    if not mapping_file.exists():
        print(f"Category mapping not found: {mapping_file}")
        print("Please run: python scripts/prepare_clothing_type_dataset.py")
        return
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    category_to_idx = mapping['category_to_idx']
    config['model']['num_clothing_types'] = len(category_to_idx)
    print(f"Loaded {len(category_to_idx)} categories")
    image_root = resolve_image_root(data_dir, mapping)
    print(f"Image root: {image_root}")

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    train_transform = create_transforms(is_train=True)
    val_transform = create_transforms(is_train=False)
    
    print("Loading datasets...")
    
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    
    if args.quick_test:
        train_df = train_df.head(20)
        val_df = val_df.head(10)
    
    train_dataset = ClothingTypeDataset(
        train_df,
        root_dir=image_root,
        transform=train_transform,
        category_to_idx=category_to_idx
    )
    
    val_dataset = ClothingTypeDataset(
        val_df,
        root_dir=image_root,
        transform=val_transform,
        category_to_idx=category_to_idx
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    trainer = Trainer(config, output_dir, device=args.device)
    
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    num_epochs = config['training']['num_epochs']
    trainer.train(train_loader, val_loader, num_epochs)


if __name__ == '__main__':
    main()
