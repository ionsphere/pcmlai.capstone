"""Inference pipeline."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from PIL import Image
import io
import base64
import requests
import json

from ..models.vision import MultiTaskClothingModel


logger = logging.getLogger(__name__)


def get_config_value(config: dict, paths, default=None):
    """Return the first matching nested config value from a list of key paths."""
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


class InferencePipeline:
    def __init__(
        self,
        vision_model_path: Optional[str] = None,
        price_model_path: Optional[str] = None,
        feature_extractor_path: Optional[str] = None,
        vector_index_path: Optional[str] = None,
        items_data_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = device
        self.models_loaded = {
            'vision': False,
            'price': False,
            'feature_extractor': False,
            'vector_index': False,
        }
        self.price_binner_info = None
        
        if vision_model_path:
            self.load_vision_model(vision_model_path)
        
        if price_model_path and feature_extractor_path:
            self.load_price_model(price_model_path, feature_extractor_path)
        
        if vector_index_path and items_data_path:
            self.load_vector_index(vector_index_path, items_data_path)
        
        logger.info(f"InferencePipeline initialized on {device}")
        logger.info(f"Models loaded: {self.models_loaded}")
    
    def load_vision_model(self, model_path: str):
        """Load vision model for clothing type and condition prediction."""
        checkpoint = torch.load(model_path, map_location=self.device)
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
        classifier_weight = state_dict.get('type_classifier.4.weight')
        inferred_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None and hasattr(classifier_weight, 'shape') else None
        self.vision_model = MultiTaskClothingModel(
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
        self.vision_model.load_state_dict(state_dict)
        self.vision_model.to(self.device)
        self.vision_model.eval()
        self.category_names = checkpoint.get('category_names', [
            't-shirt', 'shirt', 'sweater', 'hoodie', 'jacket', 'coat',
            'dress', 'skirt', 'jeans', 'pants', 'shorts', 'leggings',
            'shoes', 'boots', 'sneakers', 'bag', 'hat', 'scarf', 'gloves', 'accessories'
        ])
        self.models_loaded['vision'] = True
        logger.info(f"Vision model loaded from {model_path}")
    
    def load_price_model(self, model_path: str, feature_extractor_path: str):
        """Load price classifier and feature extractor."""
        from ..models.price import PriceClassifier, FeatureExtractor
        self.price_model = PriceClassifier.load(model_path)
        self.feature_extractor = FeatureExtractor.load(feature_extractor_path)
        for candidate in [
            Path(feature_extractor_path).parent / 'price_binner.json',
            Path(feature_extractor_path).parent.parent / 'price_classification' / 'price_binner.json',
            Path(model_path).parent / 'price_binner.json',
            Path(model_path).parent.parent / 'data' / 'price_classification' / 'price_binner.json',
        ]:
            if candidate.exists():
                with open(candidate, 'r') as f:
                    self.price_binner_info = json.load(f)
                logger.info(f"Price binner loaded from {candidate}")
                break
        self.models_loaded['price'] = True
        self.models_loaded['feature_extractor'] = True
        logger.info(f"Price model loaded from {model_path}")
        logger.info(f"Feature extractor loaded from {feature_extractor_path}")
    
    def load_vector_index(self, index_path: str, items_data_path: str):
        """Load vector index and items data."""
        from ..vector_search import FAISSIndex, SimilaritySearch
        self.vector_index = FAISSIndex()
        self.vector_index.load(index_path)
        self.items_data = pd.read_csv(items_data_path)
        self.similarity_search = SimilaritySearch(
            index=self.vector_index,
            items_data=self.items_data
        )
        self.models_loaded['vector_index'] = True
        logger.info(f"Vector index loaded from {index_path}")
        logger.info(f"Items data loaded: {len(self.items_data)} items")
    
    def preprocess_image(self, image_source: str, is_base64: bool = False) -> torch.Tensor:
        if is_base64:
            image_data = base64.b64decode(image_source)
            image = Image.open(io.BytesIO(image_data))
        else:
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def predict_vision(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        if not self.models_loaded['vision']:
            raise RuntimeError("Vision model not loaded")
        with torch.no_grad():
            type_logits, condition_score = self.vision_model(image_tensor)
            type_probs = torch.softmax(type_logits, dim=1)
            type_confidence, type_idx = torch.max(type_probs, dim=1)
            category = self.category_names[type_idx.item()]
            category_confidence = type_confidence.item()
            condition = condition_score.item()
            if condition >= 9:
                condition_label = "pristine"
            elif condition >= 8:
                condition_label = "excellent"
            elif condition >= 6:
                condition_label = "good"
            elif condition >= 4:
                condition_label = "fair"
            else:
                condition_label = "poor"
            embeddings = self.vision_model.backbone(image_tensor)
            embeddings = embeddings.cpu().numpy().flatten()
            return {
                'category': category,
                'category_confidence': category_confidence,
                'condition_score': condition,
                'condition_label': condition_label,
                'embeddings': embeddings.tolist()
            }
    
    def predict_price(
        self,
        category: str,
        condition_score: float,
        title: Optional[str] = None,
        description: Optional[str] = None,
        brand: Optional[str] = None,
        vision_embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not self.models_loaded['price']:
            raise RuntimeError("Price model not loaded")
        item_data = {
            'category': category,
            'condition_score': condition_score,
            'title': title or '',
            'description': description or '',
            'brand': brand or ''
        }
        features = self.feature_extractor.transform(
            [item_data],
            vision_embeddings=vision_embeddings.reshape(1, -1) if vision_embeddings is not None else None
        )
        prediction = self.price_model.predict(features)[0]
        probas = self.price_model.predict_proba(features)[0]
        bins_dict = (self.price_binner_info or {}).get('bins', {})
        labels_dict = (self.price_binner_info or {}).get('labels', {})
        if 'global' in bins_dict:
            price_bins = bins_dict['global']
            range_labels = labels_dict.get('global', [])
        elif bins_dict:
            first_cat = next(iter(bins_dict))
            price_bins = bins_dict[first_cat]
            range_labels = labels_dict.get(first_cat, [])
        else:
            raise RuntimeError("Price binner configuration not loaded")
        min_price = price_bins[prediction]
        max_price = price_bins[prediction + 1] if prediction + 1 < len(price_bins) else price_bins[-1] * 1.5
        recommended_price = (min_price + max_price) / 2
        range_label = (
            range_labels[prediction]
            if prediction < len(range_labels)
            else f"${min_price:.0f}-${max_price:.0f}"
        )
        semantic_ranges = ['budget', 'low', 'medium', 'high', 'premium']
        range_name = semantic_ranges[min(int(prediction), len(semantic_ranges) - 1)]
        return {
            'range': range_name,
            'range_label': range_label,
            'confidence': float(probas[prediction]),
            'min_price': float(min_price),
            'max_price': float(max_price),
            'recommended_price': float(recommended_price)
        }
    
    def find_similar_items(
        self,
        embedding: np.ndarray,
        k: int = 10,
        category: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
        condition_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar items using vector search.
        
        Args:
            embedding: Query embedding
            k: Number of similar items
            category: Filter by category
            price_range: Filter by price range
            condition_range: Filter by condition range
        
        Returns:
            List of similar items
        """
        if not self.models_loaded['vector_index']:
            raise RuntimeError("Vector index not loaded")
        
        try:
            results = self.similarity_search.search(
                query_embedding=embedding,
                k=k,
                category=category,
                price_range=price_range,
                condition_range=condition_range
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Similar items search failed: {e}")
            raise
    
    def get_pricing_context(
        self,
        embedding: np.ndarray,
        category: Optional[str] = None,
        k: int = 20,
    ) -> Dict[str, Any]:
        """
        Get pricing context from similar items.
        
        Args:
            embedding: Query embedding
            category: Filter by category
            k: Number of similar items to analyze
        
        Returns:
            Pricing context dictionary
        """
        if not self.models_loaded['vector_index']:
            return None
        
        try:
            context = self.similarity_search.get_pricing_context(
                query_embedding=embedding,
                k=k,
                category=category
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Pricing context generation failed: {e}")
            return None
    
    def predict(
        self,
        image_source: str,
        is_base64: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
        include_similar: bool = True,
        k_similar: int = 10,
    ) -> Dict[str, Any]:
        """
        Full end-to-end prediction pipeline.
        
        Args:
            image_source: Image URL or base64 string
            is_base64: Whether image_source is base64
            title: Item title
            description: Item description
            brand: Brand name
            category: Known category (optional, will predict if not provided)
            include_similar: Whether to include similar items
            k_similar: Number of similar items to return
        
        Returns:
            Complete prediction results
        """
        start_time = time.time()
        
        try:
            # 1. Preprocess image
            image_tensor = self.preprocess_image(image_source, is_base64)
            
            # 2. Vision prediction
            vision_results = self.predict_vision(image_tensor)
            
            # Use provided category or predicted category
            final_category = category or vision_results['category']
            
            # 3. Price prediction
            vision_embeddings = np.array(vision_results['embeddings'])
            
            price_prediction = self.predict_price(
                category=final_category,
                condition_score=vision_results['condition_score'],
                title=title,
                description=description,
                brand=brand,
                vision_embeddings=vision_embeddings
            )
            
            # 4. Pricing context and similar items
            pricing_context = None
            similar_items = None
            
            if self.models_loaded['vector_index']:
                pricing_context = self.get_pricing_context(
                    embedding=vision_embeddings,
                    category=final_category,
                    k=20
                )
                
                if include_similar:
                    similar_items = self.find_similar_items(
                        embedding=vision_embeddings,
                        k=k_similar,
                        category=final_category
                    )
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'vision': vision_results,
                'price': {
                    'prediction': price_prediction,
                    'context': pricing_context,
                    'similar_items': similar_items
                },
                'inference_time_ms': inference_time
            }
            
        except Exception as e:
            logger.error(f"Full prediction failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            'models_loaded': self.models_loaded,
            'device': str(self.device),
            'ready': all(self.models_loaded.values())
        }
