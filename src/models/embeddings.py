import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


def get_config_value(config: dict, paths, default=None):
    for path in paths:
        value = config
        found = True
        for key in path:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break
        if found:
            return value
    return default


class VisionEmbeddingExtractor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        layer_name: str = 'backbone',
    ):
        self.device = device
        self.layer_name = layer_name
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self._get_embedding_dim()
        
    def _load_model(self, model_path: str) -> nn.Module:
        from .vision import MultiTaskClothingModel
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = {
                'model': {
                    'backbone': 'efficientnet_b4',
                    'num_clothing_types': 20,
                    'condition_mode': 'regression',
                    'condition_scale': 10,
                    'pretrained': False
                }
            }

        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        inferred_num_classes = None
        classifier_weight = state_dict.get('type_classifier.4.weight')
        if classifier_weight is not None and hasattr(classifier_weight, 'shape'):
            inferred_num_classes = int(classifier_weight.shape[0])
        
        model = MultiTaskClothingModel(
            backbone_name=get_config_value(config, [('model', 'backbone'), ('backbone',)], 'efficientnet_b4'),
            num_clothing_types=int(get_config_value(
                config,
                [('model', 'num_clothing_types'), ('num_clothing_types',), ('num_categories',), ('num_classes',)],
                inferred_num_classes or 20
            )),
            condition_scale=int(get_config_value(config, [('model', 'condition_scale'), ('condition_scale',)], 10)),
            condition_mode=get_config_value(config, [('model', 'condition_mode'), ('condition_mode',)], 'regression'),
            pretrained=False,
            freeze_backbone=bool(get_config_value(config, [('model', 'freeze_backbone'), ('freeze_backbone',)], False))
        )
        
        model.load_state_dict(state_dict)
        return model
    
    def _get_embedding_dim(self) -> int:
        dummy_input = torch.randn(1, 3, 380, 380).to(self.device)
        with torch.no_grad():
            features = self.model.backbone(dummy_input)
            if isinstance(features, tuple):
                features = features[0]
            if len(features.shape) == 4:
                features = torch.mean(features, dim=[2, 3])
            embedding_dim = features.shape[1]
        return embedding_dim
    
    def extract(
        self,
        images: Union[List[str], List[Image.Image], torch.Tensor],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        if isinstance(images, torch.Tensor):
            image_tensors = images
        else:
            image_tensors = []
            for img in tqdm(images, desc="Loading images"):
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                img_tensor = transform(img)
                image_tensors.append(img_tensor)
            image_tensors = torch.stack(image_tensors)
        
        embeddings_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting embeddings"):
                batch = image_tensors[i:i+batch_size].to(self.device)
                features = self.model.backbone(batch)
                if isinstance(features, tuple):
                    features = features[0]
                if len(features.shape) == 4:
                    features = torch.mean(features, dim=[2, 3])
                embeddings_list.append(features.cpu().numpy())
        embeddings = np.vstack(embeddings_list)
        if normalize:
            embeddings = self._normalize(embeddings)
        return embeddings
    
    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return embeddings / norms


class TextEmbeddingExtractor:
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def extract(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        texts = [str(t) if t else "" for t in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )


class MultiModalEmbedding:
    def __init__(
        self,
        vision_extractor: Optional[VisionEmbeddingExtractor] = None,
        text_extractor: Optional[TextEmbeddingExtractor] = None,
        fusion_method: str = 'concat',
        vision_weight: float = 0.5,
        text_weight: float = 0.5,
    ):
        self.vision_extractor = vision_extractor
        self.text_extractor = text_extractor
        self.fusion_method = fusion_method
        self.vision_weight = vision_weight
        self.text_weight = text_weight
        self.embedding_dim = self._get_embedding_dim()
    
    def _get_embedding_dim(self) -> int:
        if self.fusion_method == 'concat':
            dim = 0
            if self.vision_extractor:
                dim += self.vision_extractor.embedding_dim
            if self.text_extractor:
                dim += self.text_extractor.embedding_dim
            return dim
        elif self.fusion_method == 'weighted':
            if self.vision_extractor and self.text_extractor:
                assert self.vision_extractor.embedding_dim == self.text_extractor.embedding_dim, \
                    "Vision and text embeddings must have same dimension for 'weighted' fusion"
                return self.vision_extractor.embedding_dim
            elif self.vision_extractor:
                return self.vision_extractor.embedding_dim
            else:
                return self.text_extractor.embedding_dim
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def extract(
        self,
        images: Optional[List] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        embeddings_list = []
        if images is not None and self.vision_extractor is not None:
            vision_emb = self.vision_extractor.extract(
                images, batch_size=batch_size, normalize=False
            )
            embeddings_list.append(vision_emb)
        if texts is not None and self.text_extractor is not None:
            text_emb = self.text_extractor.extract(
                texts, batch_size=batch_size, normalize=False
            )
            embeddings_list.append(text_emb)
        if len(embeddings_list) == 0:
            raise ValueError("No embeddings extracted. Provide images or texts.")
        if self.fusion_method == 'concat':
            combined = np.concatenate(embeddings_list, axis=1)
        elif self.fusion_method == 'weighted':
            if len(embeddings_list) == 2:
                combined = (self.vision_weight * embeddings_list[0] + 
                          self.text_weight * embeddings_list[1])
            else:
                combined = embeddings_list[0]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        if normalize:
            combined = VisionEmbeddingExtractor._normalize(combined)
        return combined


class DimensionalityReducer:
    def __init__(
        self,
        method: str = 'pca',
        n_components: int = 128,
        random_state: int = 42,
        **kwargs,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        
        self.reducer = PCA(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
        
        self.is_fitted = False
    
    def fit(self, embeddings: np.ndarray) -> 'DimensionalityReducer':
        self.reducer.fit(embeddings)
        self.is_fitted = True
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        return self.reducer.transform(embeddings)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.reducer, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.reducer = pickle.load(f)
        self.is_fitted = True


class EmbeddingPipeline:
    def __init__(
        self,
        vision_model_path: Optional[str] = None,
        text_model_name: str = 'all-MiniLM-L6-v2',
        fusion_method: str = 'concat',
        reduce_dim: bool = False,
        reduction_method: str = 'pca',
        target_dim: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.vision_extractor = None
        if vision_model_path:
            self.vision_extractor = VisionEmbeddingExtractor(
                model_path=vision_model_path,
                device=device
            )
        
        self.text_extractor = None
        if text_model_name:
            self.text_extractor = TextEmbeddingExtractor(
                model_name=text_model_name,
                device=device
            )
        self.multi_modal = MultiModalEmbedding(
            vision_extractor=self.vision_extractor,
            text_extractor=self.text_extractor,
            fusion_method=fusion_method,
        )
        self.reducer = None
        if reduce_dim:
            self.reducer = DimensionalityReducer(
                method=reduction_method,
                n_components=target_dim,
            )
        self.device = device
    
    def generate(
        self,
        images: Optional[List] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        embeddings = self.multi_modal.extract(
            images=images,
            texts=texts,
            batch_size=batch_size,
            normalize=False,
        )
        if self.reducer is not None:
            if not self.reducer.is_fitted:
                embeddings = self.reducer.fit_transform(embeddings)
            else:
                embeddings = self.reducer.transform(embeddings)
        if normalize:
            embeddings = VisionEmbeddingExtractor._normalize(embeddings)
        return embeddings
    
    def save(self, output_dir: str, embeddings: np.ndarray, metadata: Dict):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_path = output_dir / 'embeddings.npy'
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to {embeddings_path}")
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
        
        if self.reducer is not None and self.reducer.is_fitted:
            reducer_path = output_dir / f'{self.reducer.method}_reducer.pkl'
            self.reducer.save(str(reducer_path))
    
    @staticmethod
    def load(embeddings_dir: str) -> Tuple[np.ndarray, Dict]:
        embeddings_dir = Path(embeddings_dir)
        
        embeddings_path = embeddings_dir / 'embeddings.npy'
        embeddings = np.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path}: {embeddings.shape}")
        
        metadata_path = embeddings_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
        
        return embeddings, metadata
