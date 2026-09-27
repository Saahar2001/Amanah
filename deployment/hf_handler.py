from __future__ import annotations

import sys
from pathlib import Path


class EndpointHandler:
    """Hugging Face Inference Endpoint custom handler for AMANAH."""

    def __init__(self, path: str = ""):
        root = Path(path or ".").resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from amanah_engine.classifier import ClassifierAdapter
        from amanah_engine.references import JsonReferenceStore
        from amanah_engine.service import AmanahAnalysisService
        self._AnalyzeRequest = __import__("amanah_engine.schemas", fromlist=["AnalyzeRequest"]).AnalyzeRequest
        self.service = AmanahAnalysisService(
            ClassifierAdapter(root),
            JsonReferenceStore(root / 'reference_store.json'),
        )

    def __call__(self, data: dict) -> dict:
        payload = data.get("inputs", data)
        request = self._AnalyzeRequest.model_validate(payload)
        return self.service.analyze(request).model_dump(mode="json")
