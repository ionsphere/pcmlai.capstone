"""
API Schemas Module

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class ClothingCategory(str, Enum):
    """Supported clothing categories."""
    TSHIRT = "t-shirt"
    SHIRT = "shirt"
    SWEATER = "sweater"
    HOODIE = "hoodie"
    JACKET = "jacket"
    COAT = "coat"
    DRESS = "dress"
    SKIRT = "skirt"
    JEANS = "jeans"
    PANTS = "pants"
    SHORTS = "shorts"
    LEGGINGS = "leggings"
    SHOES = "shoes"
    BOOTS = "boots"
    SNEAKERS = "sneakers"
    BAG = "bag"
    HAT = "hat"
    SCARF = "scarf"
    GLOVES = "gloves"
    ACCESSORIES = "accessories"


class PriceRange(str, Enum):
    """Price range categories."""
    BUDGET = "budget"  # <$20
    LOW = "low"  # $20-$50
    MEDIUM = "medium"  # $50-$100
    HIGH = "high"  # $100-$200
    PREMIUM = "premium"  # >$200


class ConditionLabel(str, Enum):
    """Condition quality labels."""
    POOR = "poor"  # 1-3
    FAIR = "fair"  # 4-5
    GOOD = "good"  # 6-7
    EXCELLENT = "excellent"  # 8-9
    PRISTINE = "pristine"  # 10


# ==================== Request Models ====================

class PredictRequest(BaseModel):
    """Full prediction pipeline request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_url": "https://example.com/image.jpg",
                "title": "Vintage Levi's Jeans",
                "description": "Classic vintage denim jeans in excellent condition",
                "brand": "Levi's",
                "category": "jeans"
            }
        }
    )

    image_url: Optional[str] = Field(None, description="URL of clothing image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    title: Optional[str] = Field(None, description="Item title")
    description: Optional[str] = Field(None, description="Item description")
    brand: Optional[str] = Field(None, description="Brand name")
    category: Optional[ClothingCategory] = Field(None, description="Clothing category (if known)")
    
    @model_validator(mode='after')
    def check_image_provided(self):
        """Ensure at least one image source is provided."""
        if not self.image_url and not self.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")
        return self


class VisionRequest(BaseModel):
    """Vision model only request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_url": "https://example.com/image.jpg"
            }
        }
    )

    image_url: Optional[str] = Field(None, description="URL of clothing image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    
    @model_validator(mode='after')
    def check_image_provided(self):
        if not self.image_url and not self.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")
        return self


class PriceRequest(BaseModel):
    """Price prediction request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "jeans",
                "condition_score": 8.5,
                "title": "Vintage Levi's 501",
                "description": "Classic straight leg jeans",
                "brand": "Levi's"
            }
        }
    )

    category: ClothingCategory = Field(..., description="Clothing category")
    condition_score: float = Field(..., ge=1, le=10, description="Condition score (1-10)")
    title: Optional[str] = Field(None, description="Item title")
    description: Optional[str] = Field(None, description="Item description")
    brand: Optional[str] = Field(None, description="Brand name")
    vision_features: Optional[List[float]] = Field(None, description="Pre-computed vision embeddings")


class SimilarItemsRequest(BaseModel):
    """Similar items search request."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_url": "https://example.com/image.jpg",
                "k": 10,
                "category": "jeans",
                "price_min": 20,
                "price_max": 100
            }
        }
    )

    item_id: Optional[str] = Field(None, description="Item ID to find similar items for")
    image_url: Optional[str] = Field(None, description="URL of clothing image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    k: int = Field(10, ge=1, le=50, description="Number of similar items to return")
    category: Optional[ClothingCategory] = Field(None, description="Filter by category")
    price_min: Optional[float] = Field(None, ge=0, description="Minimum price filter")
    price_max: Optional[float] = Field(None, ge=0, description="Maximum price filter")
    condition_min: Optional[float] = Field(None, ge=1, le=10, description="Minimum condition score")
    condition_max: Optional[float] = Field(None, ge=1, le=10, description="Maximum condition score")


# ==================== Response Models ====================

class VisionResponse(BaseModel):
    """Vision model prediction response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "jeans",
                "category_confidence": 0.95,
                "condition_score": 8.5,
                "condition_label": "excellent"
            }
        }
    )

    category: ClothingCategory = Field(..., description="Predicted clothing category")
    category_confidence: float = Field(..., ge=0, le=1, description="Category prediction confidence")
    condition_score: float = Field(..., ge=1, le=10, description="Condition score (1-10)")
    condition_label: ConditionLabel = Field(..., description="Condition quality label")
    embeddings: Optional[List[float]] = Field(None, description="Vision embeddings (optional)")


class PriceRangePrediction(BaseModel):
    """Price range prediction."""
    range: PriceRange = Field(..., description="Predicted price range")
    range_label: Optional[str] = Field(None, description="Concrete trained price-bin label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    min_price: float = Field(..., description="Minimum price in range")
    max_price: float = Field(..., description="Maximum price in range")
    recommended_price: float = Field(..., description="Recommended listing price")


class PricingContext(BaseModel):
    """Pricing context from similar items."""
    n_similar_items: int = Field(..., description="Number of similar items analyzed")
    mean_price: float = Field(..., description="Mean price of similar items")
    median_price: float = Field(..., description="Median price of similar items")
    std_price: float = Field(..., description="Standard deviation of prices")
    price_range: tuple = Field(..., description="(min, max) price range")
    confidence: float = Field(..., ge=0, le=1, description="Pricing confidence score")


class SimilarItem(BaseModel):
    """Similar item information."""
    item_id: str = Field(..., description="Item ID")
    similarity: float = Field(..., ge=0, le=1, description="Similarity score")
    category: Optional[str] = Field(None, description="Item category")
    price: Optional[float] = Field(None, description="Item price")
    condition_score: Optional[float] = Field(None, description="Condition score")
    title: Optional[str] = Field(None, description="Item title")
    image_url: Optional[str] = Field(None, description="Item image URL")


class PriceResponse(BaseModel):
    """Price prediction response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": {
                    "range": "medium",
                    "confidence": 0.85,
                    "min_price": 50,
                    "max_price": 100,
                    "recommended_price": 75
                },
                "context": {
                    "n_similar_items": 20,
                    "mean_price": 72.50,
                    "median_price": 70.00,
                    "std_price": 15.20,
                    "price_range": (35, 110),
                    "confidence": 0.80
                }
            }
        }
    )

    prediction: PriceRangePrediction = Field(..., description="Price range prediction")
    context: Optional[PricingContext] = Field(None, description="Pricing context from similar items")
    similar_items: Optional[List[SimilarItem]] = Field(None, description="Top similar items")


class PredictResponse(BaseModel):
    """Full prediction pipeline response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vision": {
                    "category": "jeans",
                    "category_confidence": 0.95,
                    "condition_score": 8.5,
                    "condition_label": "excellent"
                },
                "price": {
                    "prediction": {
                        "range": "medium",
                        "confidence": 0.85,
                        "min_price": 50,
                        "max_price": 100,
                        "recommended_price": 75
                    }
                },
                "inference_time_ms": 250
            }
        }
    )

    vision: VisionResponse = Field(..., description="Vision model predictions")
    price: PriceResponse = Field(..., description="Price predictions")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")


class SimilarItemsResponse(BaseModel):
    """Similar items search response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "item_id": "12345",
                        "similarity": 0.95,
                        "category": "jeans",
                        "price": 65.00,
                        "condition_score": 8.0,
                        "title": "Levi's 501 Original Jeans"
                    }
                ],
                "count": 10,
                "search_time_ms": 15
            }
        }
    )

    items: List[SimilarItem] = Field(..., description="List of similar items")
    count: int = Field(..., description="Number of items returned")
    search_time_ms: float = Field(..., description="Search time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Invalid image format",
                "detail": "Could not decode base64 image",
                "code": "IMAGE_DECODE_ERROR"
            }
        }
    )

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.11.0",
                "models_loaded": {
                    "vision": True,
                    "price": True,
                    "vector_index": True
                },
                "uptime_seconds": 3600.5
            }
        }
    )

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
