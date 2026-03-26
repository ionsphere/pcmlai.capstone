import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import inspect
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pickle


class FeatureExtractor:
    
    def __init__(
        self,
        vision_model: Optional[nn.Module] = None,
        text_features: bool = True,
        categorical_features: bool = True,
        embedding_dim: int = 1792,
        text_max_features: int = 500,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.vision_model = vision_model
        self.text_features = text_features
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim
        self.text_max_features = text_max_features
        self.device = device
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        if self.vision_model is not None:
            self.vision_model.eval()
            self.vision_model.to(device)

    @staticmethod
    def condition_label_from_score(condition_score: float) -> str:
        if condition_score >= 9:
            return 'like_new'
        if condition_score >= 6:
            return 'good'
        return 'fair'
    
    def extract_vision_embeddings(
        self,
        images: Union[torch.Tensor, np.ndarray],
        layer_name: str = 'features'
    ) -> np.ndarray:
        if self.vision_model is None:
            raise ValueError("Vision model not provided")
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        images = images.to(self.device)
        with torch.no_grad():
            if hasattr(self.vision_model, 'backbone'):
                embeddings = self.vision_model.backbone(images)
            else:
                embeddings = self.vision_model(images)
            if len(embeddings.shape) == 4:
                embeddings = embeddings.mean(dim=[2, 3])
            return embeddings.cpu().numpy()
    
    def extract_text_features(
        self,
        texts: List[str],
        field_name: str = 'description',
        fit: bool = False
    ) -> np.ndarray:
        if not self.text_features:
            return np.array([])
        if field_name not in self.tfidf_vectorizers:
            self.tfidf_vectorizers[field_name] = TfidfVectorizer(
                max_features=self.text_max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        vectorizer = self.tfidf_vectorizers[field_name]
        
        if fit:
            features = vectorizer.fit_transform(texts).toarray()
        else:
            features = vectorizer.transform(texts).toarray()
        return features
    
    def encode_categorical(
        self,
        data: pd.DataFrame,
        column: str,
        fit: bool = False
    ) -> np.ndarray:
        if not self.categorical_features:
            return np.array([])
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
        encoder = self.label_encoders[column]
        values = data[column].fillna('unknown').astype(str)
        if fit:
            encoded = encoder.fit_transform(values)
        else:
            encoded = np.array([
                encoder.transform([v])[0] if v in encoder.classes_ else -1
                for v in values
            ])
        return encoded.reshape(-1, 1)
    
    def extract_all_features(
        self,
        data: Dict[str, Any],
        images: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fit: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        feature_list = []
        feature_names = []
        if images is not None:
            if self.vision_model is not None:
                embeddings = self.extract_vision_embeddings(images)
            else:
                embeddings = np.asarray(images)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
            feature_list.append(embeddings)
            feature_names.extend([f'vision_emb_{i}' for i in range(embeddings.shape[1])])
        categorical_columns = ['category', 'condition', 'brand']
        for col in categorical_columns:
            if col in data.columns:
                encoded = self.encode_categorical(data, col, fit=fit)
                if encoded.size > 0:
                    feature_list.append(encoded)
                    feature_names.append(f'{col}_encoded')
        if 'description' in data.columns:
            desc_features = self.extract_text_features(
                data['description'].fillna('').astype(str).tolist(),
                field_name='description',
                fit=fit
            )
            if desc_features.size > 0:
                feature_list.append(desc_features)
                if fit:
                    vocab = self.tfidf_vectorizers['description'].vocabulary_
                    feature_names.extend([f'desc_tfidf_{word}' for word in sorted(vocab, key=vocab.get)])
                else:
                    feature_names.extend([f'desc_tfidf_{i}' for i in range(desc_features.shape[1])])
        if 'title' in data.columns:
            title_features = self.extract_text_features(
                data['title'].fillna('').astype(str).tolist(),
                field_name='title',
                fit=fit
            )
            if title_features.size > 0:
                feature_list.append(title_features)
                if fit:
                    vocab = self.tfidf_vectorizers['title'].vocabulary_
                    feature_names.extend([f'title_tfidf_{word}' for word in sorted(vocab, key=vocab.get)])
                else:
                    feature_names.extend([f'title_tfidf_{i}' for i in range(title_features.shape[1])])
        numerical_columns = ['condition_score']
        for col in numerical_columns:
            if col in data.columns:
                values = data[col].fillna(data[col].median()).values.reshape(-1, 1)
                feature_list.append(values)
                feature_names.append(col)
        if len(feature_list) == 0:
            raise ValueError("No features extracted")
        features = np.hstack(feature_list)
        if fit:
            features = self.scaler.fit_transform(features)
            self.feature_names = feature_names
        else:
            features = self.scaler.transform(features)
        return features, feature_names

    def transform(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
        vision_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            data = pd.DataFrame([data])
        else:
            data = data.copy()

        if 'condition' not in data.columns and 'condition_score' in data.columns:
            data['condition'] = data['condition_score'].apply(self.condition_label_from_score)

        if 'brand' not in data.columns:
            data['brand'] = 'unknown'
        if 'title' not in data.columns:
            data['title'] = ''
        if 'description' not in data.columns:
            data['description'] = ''
        if 'category' not in data.columns:
            data['category'] = 'unknown'

        features, _ = self.extract_all_features(
            data,
            images=vision_embeddings,
            fit=False
        )
        return features
    
    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'tfidf_vectorizers': self.tfidf_vectorizers,
            'feature_names': self.feature_names,
            'text_features': self.text_features,
            'categorical_features': self.categorical_features,
            'embedding_dim': self.embedding_dim,
            'text_max_features': self.text_max_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        extractor = cls(
            vision_model=None,
            text_features=state['text_features'],
            categorical_features=state['categorical_features'],
            embedding_dim=state['embedding_dim'],
            text_max_features=state['text_max_features']
        )
        extractor.scaler = state['scaler']
        extractor.label_encoders = state['label_encoders']
        extractor.tfidf_vectorizers = state['tfidf_vectorizers']
        extractor.feature_names = state['feature_names']
        return extractor


class PriceRangeBinner:
    
    def __init__(
        self,
        strategy: str = 'quantile',
        n_bins: int = 5,
        custom_bins: Optional[List[float]] = None,
        category_specific: bool = False
    ):
        self.strategy = strategy
        self.n_bins = n_bins
        self.custom_bins = custom_bins
        self.category_specific = category_specific
        self.bins = {}
        self.labels = {}
    
    def fit(self, prices: np.ndarray, categories: Optional[np.ndarray] = None):
        if self.strategy == 'custom':
            if self.custom_bins is None:
                raise ValueError("Custom bins must be provided for 'custom' strategy")
            self.bins['global'] = self.custom_bins
            self.labels['global'] = [
                f"${self.custom_bins[i]:.0f}-${self.custom_bins[i+1]:.0f}"
                for i in range(len(self.custom_bins) - 1)
            ]
            return
        
        if self.category_specific and categories is not None:
            unique_cats = np.unique(categories)
        else:
            unique_cats = ['global']
            categories = np.array(['global'] * len(prices))
        
        for cat in unique_cats:
            cat_prices = prices[categories == cat] if cat != 'global' else prices
            
            if self.strategy == 'quantile':
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bins = np.quantile(cat_prices, quantiles)
            elif self.strategy == 'uniform':
                bins = np.linspace(cat_prices.min(), cat_prices.max(), self.n_bins + 1)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self.bins[cat] = bins
            self.labels[cat] = [
                f"${bins[i]:.0f}-${bins[i+1]:.0f}"
                for i in range(len(bins) - 1)
            ]
    
    def transform(
        self,
        prices: np.ndarray,
        categories: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.bins:
            raise ValueError("Binner not fitted. Call fit() first.")
        
        bin_indices = np.zeros(len(prices), dtype=int)
        bin_labels = np.array([''] * len(prices), dtype=object)
        
        if self.category_specific and categories is not None:
            unique_cats = np.unique(categories)
        else:
            unique_cats = ['global']
            categories = np.array(['global'] * len(prices))
        
        for cat in unique_cats:
            mask = categories == cat
            cat_prices = prices[mask]
            
            bins = self.bins.get(cat, self.bins['global'])
            labels = self.labels.get(cat, self.labels['global'])
            
            indices = np.digitize(cat_prices, bins) - 1
            indices = np.clip(indices, 0, len(labels) - 1)
            
            bin_indices[mask] = indices
            bin_labels[mask] = [labels[i] for i in indices]
        
        return bin_indices, bin_labels
    
    def fit_transform(
        self,
        prices: np.ndarray,
        categories: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(prices, categories)
        return self.transform(prices, categories)
    
    def get_bin_info(self) -> Dict[str, Any]:
        return {
            'strategy': self.strategy,
            'n_bins': self.n_bins,
            'category_specific': self.category_specific,
            'bins': self.bins,
            'labels': self.labels
        }


class PriceClassifier:
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_classes: int = 5,
        **model_params
    ):
        self.model_type = model_type
        self.n_classes = n_classes
        self.model_params = model_params
        self.model = None
        self.feature_importance = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=self.n_classes,
                eval_metric='mlogloss',
                **self.model_params
            )
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=self.n_classes,
                **self.model_params
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'mlp':
            raise NotImplementedError("MLP model coming soon")
        elif self.model_type == 'ensemble':
            raise NotImplementedError("Ensemble model coming soon")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        if self.model_type in ['xgboost', 'lightgbm']:
            if X_val is not None and y_val is not None:
                fit_kwargs = {
                    'eval_set': [(X_val, y_val)]
                }
                fit_signature = inspect.signature(self.model.fit)
                if 'verbose' in fit_signature.parameters:
                    fit_kwargs['verbose'] = verbose
                self.model.fit(X_train, y_train, **fit_kwargs)
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_macro = f1_score(y, y_pred, average='macro')
        
        within_1 = np.mean(np.abs(y - y_pred) <= 1)
        
        report = classification_report(
            y, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'within_1_accuracy': within_1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20
    ) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Model not trained or doesn't support feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_k)
        
        return importance_df
    
    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'xgboost':
            self.model.save_model(str(path))
        elif self.model_type == 'lightgbm':
            if hasattr(self.model, 'booster_') and self.model.booster_ is not None:
                self.model.booster_.save_model(str(path))
            else:
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        
        metadata = {
            'model_type': self.model_type,
            'n_classes': self.n_classes,
            'model_params': self.model_params,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        
        with open(path.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)

        classifier = cls(
            model_type=metadata['model_type'],
            n_classes=metadata['n_classes'],
            **metadata['model_params']
        )
        classifier.feature_importance = (
            np.array(metadata['feature_importance'])
            if metadata['feature_importance'] else None
        )
        
        if classifier.model_type == 'xgboost':
            classifier.model = xgb.XGBClassifier()
            classifier.model.load_model(str(path))
        elif classifier.model_type == 'lightgbm':
            if path.suffix in ['.pkl', '.pickle']:
                with open(path, 'rb') as f:
                    classifier.model = pickle.load(f)
            else:
                classifier.model = lgb.Booster(model_file=str(path))
        else:
            with open(path, 'rb') as f:
                classifier.model = pickle.load(f)

        return classifier
