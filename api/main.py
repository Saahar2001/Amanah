from __future__ import annotations

import os

from fastapi import FastAPI

from amanah_engine.classifier import ClassifierAdapter
from amanah_engine.references import JsonReferenceStore
from amanah_engine.service import AmanahAnalysisService
from api.routes import router


def build_default_service() -> AmanahAnalysisService:
    model_dir = os.getenv('MODEL_DIR', 'checkpoints/amanah-drift-v0.1')
    registry = os.getenv('SOURCE_REGISTRY_PATH', 'data/reference_store.json')
    return AmanahAnalysisService(ClassifierAdapter(model_dir), JsonReferenceStore(registry))


def create_app(*, service=None, api_token: str | None = None) -> FastAPI:
    app = FastAPI(title='AMANAH Semantic Integrity API', version='0.1.0')
    app.state.service = service or build_default_service()
    app.state.api_token = api_token if api_token is not None else os.getenv('AMANAH_API_TOKEN')
    app.include_router(router)
    return app


app = create_app()
