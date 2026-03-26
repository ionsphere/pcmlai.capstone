import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.price import FeatureExtractor
from src.models.vision import MultiTaskClothingModel


def get_config_value(config: dict, paths, default=None):
    for path in paths:
        value = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    return default


class ClothingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, image_column: str = "image_path"):
        self.df = df
        self.transform = transform
        self.image_column = image_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx][self.image_column]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), idx


def load_vision_model(model_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get(
        "config",
        {
            "model": {
                "num_clothing_types": 20,
                "backbone": "efficientnet_b4",
                "condition_mode": "regression",
                "condition_scale": 10,
                "pretrained": False,
            }
        },
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    classifier_weight = state_dict.get("type_classifier.4.weight")
    inferred_num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None and hasattr(classifier_weight, "shape") else None
    model = MultiTaskClothingModel(
        backbone_name=get_config_value(config, [("model", "backbone"), ("backbone",)], "efficientnet_b4"),
        num_clothing_types=int(
            get_config_value(
                config,
                [("model", "num_clothing_types"), ("num_clothing_types",), ("num_categories",)],
                inferred_num_classes or 20,
            )
        ),
        condition_scale=int(get_config_value(config, [("model", "condition_scale"), ("condition_scale",)], 10)),
        condition_mode=get_config_value(config, [("model", "condition_mode"), ("condition_mode",)], "regression"),
        pretrained=False,
        freeze_backbone=bool(get_config_value(config, [("model", "freeze_backbone"), ("freeze_backbone",)], False)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_features_from_dataset(
    data_csv: Path,
    output_dir: Path,
    vision_model_path: Optional[Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv)

    required_cols = {"title", "description", "brand", "category", "condition", "price"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    vision_model = load_vision_model(vision_model_path, device) if vision_model_path else None
    feature_extractor = FeatureExtractor(
        vision_model=vision_model,
        text_features=True,
        categorical_features=True,
        device=device,
    )

    vision_embeddings = None
    if vision_model is not None:
        if "image_path" not in df.columns:
            raise ValueError("image_path column is required when vision embeddings are enabled")
        dataset = ClothingDataset(
            df,
            transform=transforms.Compose(
                [
                    transforms.Resize((380, 380)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device == "cuda",
        )
        embeddings = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="embeddings"):
                embeddings.append(feature_extractor.extract_vision_embeddings(images.to(device)))
        vision_embeddings = np.vstack(embeddings)
        np.save(output_dir / "vision_embeddings.npy", vision_embeddings)

    data_dict = {
        "title": df["title"].fillna("").astype(str).tolist(),
        "description": df["description"].fillna("").astype(str).tolist(),
        "brand": df["brand"].fillna("unknown").astype(str).tolist(),
        "category": df["category"].fillna("unknown").astype(str).tolist(),
        "condition": df["condition"].fillna("good").astype(str).tolist(),
        "condition_score": df.get("condition_score", pd.Series([5.0] * len(df))).fillna(5.0).values,
    }
    features, feature_names = feature_extractor.extract_all_features(data_dict, images=None, fit=True)

    np.save(output_dir / "features.npy", features)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    feature_extractor.save(output_dir / "feature_extractor.pkl")

    metadata = {
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "has_vision_embeddings": vision_embeddings is not None,
        "vision_embedding_dim": int(vision_embeddings.shape[1]) if vision_embeddings is not None else 0,
        "feature_names": feature_names,
        "data_csv": str(data_csv),
        "vision_model": str(vision_model_path) if vision_model_path else None,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"samples={features.shape[0]} features={features.shape[1]}")
    print(output_dir)


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Extract features for price classification")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--vision-model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args(argv)

    data_csv = Path(args.data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(data_csv)

    vision_model_path = None
    if args.vision_model and not args.no_vision:
        vision_model_path = Path(args.vision_model)
        if not vision_model_path.exists():
            raise FileNotFoundError(vision_model_path)

    extract_features_from_dataset(
        data_csv=data_csv,
        output_dir=Path(args.output_dir),
        vision_model_path=vision_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
