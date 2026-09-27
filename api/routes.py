from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from amanah_engine.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@router.get('/version')
def version() -> dict[str, str]:
    return {'version': '0.1.0'}


@router.get('/ready')
def ready(request: Request) -> dict:
    service = request.app.state.service
    status = service.readiness() if hasattr(service, 'readiness') else {'ready': False, 'model_available': False, 'reference_count': 0}
    if not status.get('ready'):
        raise HTTPException(status_code=503, detail=status)
    return status


@router.post('/v1/analyze', response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    required = getattr(request.app.state, 'api_token', None)
    if required:
        supplied = request.headers.get('authorization', '')
        if supplied != f'Bearer {required}':
            raise HTTPException(status_code=401, detail='Unauthorized')
    service = request.app.state.service
    return service.analyze(payload)
