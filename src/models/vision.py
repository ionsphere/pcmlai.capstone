import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any, Dict, Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import CLIPModel, ViTModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _get_torchvision_weights(weight_name: str, pretrained: bool):
    if not pretrained:
        return None
    return getattr(models, weight_name).DEFAULT


class MultiTaskClothingModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b4',
        num_clothing_types: int = 20,
        condition_scale: int = 10,
        condition_mode: str = 'regression',
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.condition_mode = condition_mode
        self.num_clothing_types = num_clothing_types
        self.condition_scale = condition_scale

        self.backbone, num_features = self._create_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.type_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_clothing_types),
        )

        if condition_mode == 'regression':
            self.condition_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.condition_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, condition_scale),
            )

    def _create_backbone(self, backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        if backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(
                weights=_get_torchvision_weights('EfficientNet_B0_Weights', pretrained)
            )
            num_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b4':
            backbone = models.efficientnet_b4(
                weights=_get_torchvision_weights('EfficientNet_B4_Weights', pretrained)
            )
            num_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b7':
            backbone = models.efficientnet_b7(
                weights=_get_torchvision_weights('EfficientNet_B7_Weights', pretrained)
            )
            num_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(
                weights=_get_torchvision_weights('ResNet50_Weights', pretrained)
            )
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(
                weights=_get_torchvision_weights('ResNet101_Weights', pretrained)
            )
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == 'resnet152':
            backbone = models.resnet152(
                weights=_get_torchvision_weights('ResNet152_Weights', pretrained)
            )
            num_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name.startswith('convnext_') and TIMM_AVAILABLE:
            backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            num_features = backbone.num_features
        elif backbone_name.startswith('vit_') and TRANSFORMERS_AVAILABLE:
            if backbone_name == 'vit_base':
                backbone = ViTModel.from_pretrained('google/vit-base-patch16-224')
                num_features = 768
            elif backbone_name == 'vit_large':
                backbone = ViTModel.from_pretrained('google/vit-large-patch16-224')
                num_features = 1024
            else:
                raise ValueError(f'Unknown ViT variant: {backbone_name}')
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}')
        return backbone, num_features

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name.startswith('vit_'):
            return self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_backbone_features(x)
        shared = self.shared_features(features)
        type_logits = self.type_classifier(shared)

        if self.condition_mode == 'regression':
            condition_score = self.condition_head(shared)
            condition_score = condition_score * (self.condition_scale - 1) + 1
        else:
            condition_score = self.condition_head(shared)

        return type_logits, condition_score

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_features(self._extract_backbone_features(x))


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        condition_mode: str = 'regression',
        type_weight: Optional[float] = None,
        condition_weight: Optional[float] = None,
    ):
        super().__init__()
        self.condition_mode = condition_mode
        self.classification_loss = nn.CrossEntropyLoss()
        self.condition_loss = nn.SmoothL1Loss() if condition_mode == 'regression' else nn.CrossEntropyLoss()
        self.log_var_type = nn.Parameter(torch.tensor([type_weight]).log()) if type_weight is not None else nn.Parameter(torch.zeros(1))
        self.log_var_condition = nn.Parameter(torch.tensor([condition_weight]).log()) if condition_weight is not None else nn.Parameter(torch.zeros(1))

    def forward(
        self,
        type_logits: torch.Tensor,
        condition_pred: torch.Tensor,
        type_targets: torch.Tensor,
        condition_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_type = self.classification_loss(type_logits, type_targets)
        if self.condition_mode == 'regression':
            loss_condition = self.condition_loss(condition_pred.squeeze(), condition_targets.float())
        else:
            loss_condition = self.condition_loss(condition_pred, condition_targets)

        precision_type = torch.exp(-self.log_var_type)
        precision_condition = torch.exp(-self.log_var_condition)
        total_loss = (
            precision_type * loss_type + self.log_var_type
            + precision_condition * loss_condition + self.log_var_condition
        )
        return total_loss, loss_type, loss_condition

    def get_task_weights(self) -> Dict[str, float]:
        return {
            'type_weight': torch.exp(-self.log_var_type).item(),
            'condition_weight': torch.exp(-self.log_var_condition).item(),
            'type_log_var': self.log_var_type.item(),
            'condition_log_var': self.log_var_condition.item(),
        }


class ModelFactory:
    @staticmethod
    def create_model(config: Dict[str, Any]) -> MultiTaskClothingModel:
        model_config = config.get('model', {})
        return MultiTaskClothingModel(
            backbone_name=model_config.get('backbone', 'efficientnet_b4'),
            num_clothing_types=model_config.get('num_clothing_types', 20),
            condition_scale=model_config.get('condition_scale', 10),
            condition_mode=model_config.get('condition_mode', 'regression'),
            pretrained=model_config.get('pretrained', True),
            freeze_backbone=model_config.get('freeze_backbone', False),
        )

    @staticmethod
    def create_loss(config: Dict[str, Any]) -> MultiTaskLoss:
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        return MultiTaskLoss(
            condition_mode=model_config.get('condition_mode', 'regression'),
            type_weight=training_config.get('type_weight'),
            condition_weight=training_config.get('condition_weight'),
        )

    @staticmethod
    def load_pretrained(checkpoint_path: str, config: Dict[str, Any]) -> MultiTaskClothingModel:
        model = ModelFactory.create_model(config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }
