import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from .pipeline import InferencePipeline
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PriceRequest,
    PriceResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
    VisionRequest,
    VisionResponse,
)
from ..utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

pipeline: Optional[InferencePipeline] = None
start_time = time.time()

API_VERSION = "0.11.0"
API_TITLE = "Clothing Price Predictor API"
API_DESCRIPTION = """
# Clothing Condition & Price Prediction API
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting Clothing Price Predictor API")
    vision_model_path = os.getenv("VISION_MODEL_PATH", "models/multitask/checkpoints/best_loss.pth")
    price_model_path = os.getenv("PRICE_MODEL_PATH", "models/price_model/xgboost_model.pkl")
    feature_extractor_path = os.getenv("FEATURE_EXTRACTOR_PATH", "data/features/feature_extractor.pkl")
    vector_index_path = os.getenv("VECTOR_INDEX_PATH", "data/vector_index/clothing_index")
    items_data_path = os.getenv("ITEMS_DATA_PATH", "data/vector_index/items_data.csv")
    pipeline = InferencePipeline(
        vision_model_path=vision_model_path if os.path.exists(vision_model_path) else None,
        price_model_path=price_model_path if os.path.exists(price_model_path) else None,
        feature_extractor_path=feature_extractor_path if os.path.exists(feature_extractor_path) else None,
        vector_index_path=vector_index_path if os.path.exists(f"{vector_index_path}.index") else None,
        items_data_path=items_data_path if os.path.exists(items_data_path) else None,
    )
    logger.info(f"Pipeline initialized: {pipeline.get_status()}")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str((time.time() - started) * 1000)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "code": f"HTTP_{exc.status_code}"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc), "code": "INTERNAL_ERROR"},
    )


@app.get("/", tags=["General"])
async def root():
    return {"name": API_TITLE, "version": API_VERSION, "status": "running", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    if pipeline is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "version": API_VERSION,
                "models_loaded": {},
                "uptime_seconds": time.time() - start_time,
                "error": "Pipeline not initialized",
            },
        )
    pipeline_status = pipeline.get_status()
    return {
        "status": "healthy" if pipeline_status["ready"] else "degraded",
        "version": API_VERSION,
        "models_loaded": pipeline_status["models_loaded"],
        "uptime_seconds": time.time() - start_time,
    }


def get_image_source(image_url: Optional[str], image_base64: Optional[str]):
    if image_url:
        return image_url, False
    if image_base64:
        return image_base64, True
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either image_url or image_base64 must be provided")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialized")
    image_source, is_base64 = get_image_source(request.image_url, request.image_base64)
    return pipeline.predict(
        image_source=image_source,
        is_base64=is_base64,
        title=request.title,
        description=request.description,
        brand=request.brand,
        category=request.category,
        include_similar=True,
        k_similar=10,
    )


@app.post("/vision", response_model=VisionResponse, tags=["Prediction"])
async def predict_vision(request: VisionRequest):
    if pipeline is None or not pipeline.models_loaded["vision"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vision model not loaded")
    image_source, is_base64 = get_image_source(request.image_url, request.image_base64)
    result = pipeline.predict_vision(pipeline.preprocess_image(image_source, is_base64))
    return {k: v for k, v in result.items() if k != "embeddings"}


@app.post("/price", response_model=PriceResponse, tags=["Prediction"])
async def predict_price(request: PriceRequest):
    if pipeline is None or not pipeline.models_loaded["price"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Price model not loaded")
    vision_embeddings = np.array(request.vision_features) if request.vision_features else None
    price_prediction = pipeline.predict_price(
        category=request.category,
        condition_score=request.condition_score,
        title=request.title,
        description=request.description,
        brand=request.brand,
        vision_embeddings=vision_embeddings,
    )
    pricing_context = None
    if vision_embeddings is not None and pipeline.models_loaded["vector_index"]:
        pricing_context = pipeline.get_pricing_context(
            embedding=vision_embeddings,
            category=request.category,
            k=20,
        )
    return {"prediction": price_prediction, "context": pricing_context, "similar_items": None}


@app.post("/similar", response_model=SimilarItemsResponse, tags=["Search"])
async def find_similar_items(request: SimilarItemsRequest):
    if pipeline is None or not pipeline.models_loaded["vector_index"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector index not loaded")
    start_time_search = time.time()
    if request.item_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Item ID lookup not yet implemented")
    image_source, is_base64 = get_image_source(request.image_url, request.image_base64)
    vision_result = pipeline.predict_vision(pipeline.preprocess_image(image_source, is_base64))
    embedding = np.array(vision_result["embeddings"])
    items = pipeline.find_similar_items(
        embedding=embedding,
        k=request.k,
        category=request.category,
        price_range=(request.price_min, request.price_max) if request.price_min is not None and request.price_max is not None else None,
        condition_range=(request.condition_min, request.condition_max) if request.condition_min is not None and request.condition_max is not None else None,
    )
    return {
        "items": [
            {
                "item_id": result.get("item_id", str(i)),
                "similarity": result.get("similarity", 0.0),
                "category": result.get("category"),
                "price": result.get("price"),
                "condition_score": result.get("condition_score"),
                "title": result.get("title"),
                "image_url": result.get("image_url"),
            }
            for i, result in enumerate(items)
        ],
        "count": len(items),
        "search_time_ms": (time.time() - start_time_search) * 1000,
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )
    openapi_schema["components"]["schemas"]["ErrorResponse"] = ErrorResponse.model_json_schema()
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
