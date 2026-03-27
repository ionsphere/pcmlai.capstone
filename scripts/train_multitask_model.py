import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence, Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image


sys.path.append(str(Path(__file__).parent.parent))

from src.models.vision import MultiTaskClothingModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 380
DEMO_EPOCHS = 2
DEFAULT_CONFIG = {
    'train_csv': None,
    'val_csv': None,
    'image_size': DEFAULT_IMAGE_SIZE,
    'backbone': 'efficientnet_b4',
    'pretrained': True,
    'condition_mode': 'regression',
    'condition_scale': 10,
    'num_clothing_types': None,
    'freeze_backbone': False,
    'batch_size': 32,
    'epochs': 50,
    'num_workers': 0,
    'seed': 42,
    'learn_task_weights': True,
    'optimizer': 'adamw',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'grad_clip': 1.0,
    'scheduler': 'cosine',
    'min_lr': 1e-6,
    'step_size': 10,
    'gamma': 0.1,
    'factor': 0.5,
    'lr_patience': 5,
    'early_stopping_patience': 10,
    'save_every': 10,
    'pin_memory': False,
}


def build_training_config(
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    quick_test: bool = False,
    demo: bool = False,
) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['train_csv'] = train_csv
    config['val_csv'] = val_csv
    if demo:
        config['epochs'] = DEMO_EPOCHS
        config['save_every'] = 1
    if quick_test:
        config['batch_size'] = 8
        config['epochs'] = DEMO_EPOCHS
        config['save_every'] = 1
    return config


class MultiTaskClothingDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[transforms.Compose] = None,
        is_training: bool = True
    ):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_training = is_training
        
        required_cols = ['image_path', 'category', 'condition_score']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"CSV must contain '{col}' column")
        
        self.categories = sorted(self.data['category'].unique())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.num_categories = len(self.categories)
        
        logger.info(f"Loaded {len(self.data)} samples from {csv_path}")
        logger.info(f"Categories: {self.num_categories} - {self.categories}")
        logger.info(f"Condition range: {self.data['condition_score'].min():.2f} - {self.data['condition_score'].max():.2f}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.data.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        category_idx = self.category_to_idx[row['category']]
        condition_score = float(row['condition_score'])
        targets = {
            'category': torch.tensor(category_idx, dtype=torch.long),
            'condition': torch.tensor(condition_score, dtype=torch.float32)
        }
        return image, targets


def get_transforms(config: dict, is_training: bool = True) -> transforms.Compose:
    img_size = int(config['image_size'])
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int = 2, learn_weights: bool = True):
        super().__init__()
        self.num_tasks = num_tasks
        self.learn_weights = learn_weights
        if learn_weights:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.register_buffer('log_vars', torch.zeros(num_tasks))
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        classification_loss = task_losses['classification']
        regression_loss = task_losses['regression']
        if self.learn_weights:
            weighted_cls_loss = classification_loss / (2 * torch.exp(self.log_vars[0])) + self.log_vars[0] / 2
            weighted_reg_loss = regression_loss / (2 * torch.exp(self.log_vars[1])) + self.log_vars[1] / 2
            total_loss = weighted_cls_loss + weighted_reg_loss
            cls_weight = 1.0 / (2 * torch.exp(self.log_vars[0]))
            reg_weight = 1.0 / (2 * torch.exp(self.log_vars[1]))
        else:
            total_loss = classification_loss + regression_loss
            cls_weight = 1.0
            reg_weight = 1.0
        
        loss_dict = {
            'total': total_loss.item(),
            'classification': classification_loss.item(),
            'regression': regression_loss.item(),
            'cls_weight': cls_weight.item() if torch.is_tensor(cls_weight) else cls_weight,
            'reg_weight': reg_weight.item() if torch.is_tensor(reg_weight) else reg_weight
        }
        
        return total_loss, loss_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    multitask_loss: MultiTaskLoss,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
    config: dict
) -> Dict[str, float]:
    model.train()
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    condition_mae = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        category_targets = targets['category'].to(device)
        condition_targets = targets['condition'].to(device)
        
        with autocast():
            type_logits, condition_pred = model(images)
            cls_loss = classification_criterion(type_logits, category_targets)
            reg_loss = regression_criterion(condition_pred.squeeze(), condition_targets)
            task_losses = {
                'classification': cls_loss,
                'regression': reg_loss
            }
            outputs = {
                'category': type_logits,
                'condition': condition_pred
            }
            loss, loss_dict = multitask_loss(outputs, targets, task_losses)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        if config.get('grad_clip', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss_dict['total']
        total_cls_loss += loss_dict['classification']
        total_reg_loss += loss_dict['regression']
        
        _, predicted = type_logits.max(1)
        correct_predictions += predicted.eq(category_targets).sum().item()
        total_samples += category_targets.size(0)
        
        condition_mae += torch.abs(condition_pred.squeeze() - condition_targets).sum().item()
        
        progress_bar.set_postfix({
            'loss': loss_dict['total'],
            'acc': 100. * correct_predictions / total_samples,
            'mae': condition_mae / total_samples
        })
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'classification_loss': total_cls_loss / len(dataloader),
        'regression_loss': total_reg_loss / len(dataloader),
        'accuracy': 100. * correct_predictions / total_samples,
        'condition_mae': condition_mae / total_samples
    }
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    multitask_loss: MultiTaskLoss,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.eval()
    
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    condition_mae = 0.0
    condition_rmse = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Validation {epoch}"):
            images = images.to(device)
            category_targets = targets['category'].to(device)
            condition_targets = targets['condition'].to(device)
            
            type_logits, condition_pred = model(images)
            
            cls_loss = classification_criterion(type_logits, category_targets)
            reg_loss = regression_criterion(condition_pred.squeeze(), condition_targets)
            
            task_losses = {
                'classification': cls_loss,
                'regression': reg_loss
            }
            outputs = {
                'category': type_logits,
                'condition': condition_pred
            }
            loss, loss_dict = multitask_loss(outputs, targets, task_losses)
            
            total_loss += loss_dict['total']
            total_cls_loss += loss_dict['classification']
            total_reg_loss += loss_dict['regression']
            
            _, predicted = type_logits.max(1)
            correct_predictions += predicted.eq(category_targets).sum().item()
            total_samples += category_targets.size(0)
            
            condition_errors = torch.abs(condition_pred.squeeze() - condition_targets)
            condition_mae += condition_errors.sum().item()
            condition_rmse += (condition_errors ** 2).sum().item()
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'classification_loss': total_cls_loss / len(dataloader),
        'regression_loss': total_reg_loss / len(dataloader),
        'accuracy': 100. * correct_predictions / total_samples,
        'condition_mae': condition_mae / total_samples,
        'condition_rmse': np.sqrt(condition_rmse / total_samples)
    }
    
    return metrics


def train_model(config: dict, output_dir: Path) -> Dict[str, any]:
    if not config['train_csv'] or not config['val_csv']:
        raise ValueError("Both train_csv and val_csv are required.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    train_dataset = MultiTaskClothingDataset(
        config['train_csv'],
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = MultiTaskClothingDataset(
        config['val_csv'],
        transform=val_transform,
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'] if device.type == 'cuda' else False
    )

    if len(train_loader) == 0:
        raise ValueError(f"Training dataset is empty: {config['train_csv']}")
    if len(val_loader) == 0:
        raise ValueError(f"Validation dataset is empty: {config['val_csv']}")
    
    # Create model
    num_categories = train_dataset.num_categories
    model = MultiTaskClothingModel(
        backbone_name=config['backbone'],
        num_clothing_types=int(config['num_clothing_types'] or num_categories),
        condition_scale=config['condition_scale'],
        condition_mode=config['condition_mode'],
        pretrained=config['pretrained'],
        freeze_backbone=config['freeze_backbone']
    )
    model = model.to(device)
    
    logger.info(f"Model: {config['backbone']}")
    logger.info(f"Number of categories: {num_categories}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    multitask_loss = MultiTaskLoss(
        num_tasks=2,
        learn_weights=config['learn_task_weights']
    )
    
    optimizer_name = config['optimizer'].lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    scheduler_name = config['scheduler'].lower()
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    elif scheduler_name == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['factor'],
            patience=config['lr_patience']
        )
    else:
        scheduler = None
    
    scaler = GradScaler()
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = config['early_stopping_patience']
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_condition_mae': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_condition_mae': [],
        'val_condition_rmse': [],
        'learning_rates': []
    }
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, multitask_loss,
            device, scaler, epoch, config
        )
        val_metrics = validate(
            model, val_loader, multitask_loss, device, epoch
        )
        if scheduler:
            if scheduler_name == 'reduce':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{config['epochs']}")
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.2f}%, "
                   f"MAE: {train_metrics['condition_mae']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%, "
                   f"MAE: {val_metrics['condition_mae']:.4f}, "
                   f"RMSE: {val_metrics['condition_rmse']:.4f}")
        logger.info(f"LR: {current_lr:.6f}, "
                   f"Time: {time.time() - epoch_start:.2f}s")
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('ConditionMAE/train', train_metrics['condition_mae'], epoch)
        writer.add_scalar('ConditionMAE/val', val_metrics['condition_mae'], epoch)
        writer.add_scalar('ConditionRMSE/val', val_metrics['condition_rmse'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_condition_mae'].append(train_metrics['condition_mae'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_condition_mae'].append(val_metrics['condition_mae'])
        history['val_condition_rmse'].append(val_metrics['condition_rmse'])
        history['learning_rates'].append(current_lr)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        
        torch.save(checkpoint, checkpoint_dir / 'last.pth')
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, checkpoint_dir / 'best_loss.pth')
            logger.info(f"Saved best model (loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(checkpoint, checkpoint_dir / 'best_accuracy.pth')
        
        if epoch % config['save_every'] == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')
        
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training complete! Total time: {total_time / 3600:.2f} hours")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    writer.close()
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_acc,
        'total_epochs': epoch,
        'total_time': total_time
    }


def create_mock_dataset(output_dir: Path, num_samples: int = 100):
    logger.info(f"Creating mock dataset with {num_samples} samples...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    categories = [
        't-shirt', 'shirt', 'sweater', 'jacket', 'dress',
        'jeans', 'pants', 'skirt', 'shoes', 'bag'
    ]
    
    data = []
    for i in range(num_samples):
        category = np.random.choice(categories)
        condition = np.random.uniform(1, 10)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img = Image.new('RGB', (380, 380), color=color)
        img_path = images_dir / f'mock_{i:04d}.jpg'
        img.save(img_path)
        data.append({
            'image_path': str(img_path),
            'category': category,
            'condition_score': condition
        })
    
    np.random.shuffle(data)
    split_idx = int(0.8 * len(data))
    
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    pd.DataFrame(train_data).to_csv(output_dir / 'train.csv', index=False)
    pd.DataFrame(val_data).to_csv(output_dir / 'val.csv', index=False)
    
    logger.info(f"Created {len(train_data)} training samples and {len(val_data)} validation samples")
    logger.info(f"Dataset saved to {output_dir}")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Multi-Task Model Training")
    parser.add_argument(
        '--train-csv',
        type=str,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--val-csv',
        type=str,
        help='Path to validation CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/multitask',
        help='Output directory for models and logs'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with mock data (2 epochs)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode on the real dataset (2 epochs)'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Create mock dataset for testing'
    )
    parser.add_argument(
        '--mock-samples',
        type=int,
        default=100,
        help='Number of mock samples to create'
    )
    
    args = parser.parse_args(argv)
    if args.quick_test:
        logger.info("Running quick test mode...")
        mock_dir = Path('data/processed/multitask_mock')
        create_mock_dataset(mock_dir, num_samples=100)
        config = build_training_config(
            train_csv=str(mock_dir / 'train.csv'),
            val_csv=str(mock_dir / 'val.csv'),
            quick_test=True,
        )
        
        train_model(config, Path(args.output_dir) / 'quick_test')
        return
    
    if args.mock:
        mock_dir = Path('data/processed/multitask_mock')
        create_mock_dataset(mock_dir, num_samples=args.mock_samples)
        return
    
    if not args.train_csv or not args.val_csv:
        parser.error("Both --train-csv and --val-csv are required")
    
    config = build_training_config(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        demo=args.demo,
    )
    if args.demo:
        logger.info(f"Running demo mode ({DEMO_EPOCHS} epochs on the real dataset)...")
    
    results = train_model(config, Path(args.output_dir))
    
    logger.info("Training Results:")
    logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
    logger.info(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    logger.info(f"Total Epochs: {results['total_epochs']}")
    logger.info(f"Total Time: {results['total_time'] / 3600:.2f} hours")


if __name__ == '__main__':
    main()
