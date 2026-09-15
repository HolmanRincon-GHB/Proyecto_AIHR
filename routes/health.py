"""
============================================================
  routes/health.py — Endpoint GET /health
  Seminario: Agentes de IA · Sesión 5

  Endpoint de monitoreo — verifica que el servicio
  está vivo y devuelve información del estado.
  Railway y otros servicios cloud usan este endpoint
  para saber si el servicio está funcionando.
============================================================
"""

from fastapi import APIRouter
from models.schemas import HealthResponse
from core.agent_core import agente_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Verifica que el servicio está activo y devuelve información del agente.",
)
async def health_check():
    """
    GET /health

    Returns:
        HealthResponse con el estado del servicio
    """
    return HealthResponse(
        status="ok",
        agente=agente_service.nombre,
        proveedor=agente_service.proveedor,
    )


# ✏️ MODIFICA AQUÍ: agrega más info al health check
@router.get(
    "/health/detail",
    summary="Estado detallado",
    description="Información detallada del servicio incluyendo herramientas disponibles.",
)
async def health_detail():
    """GET /health/detail — estado completo"""
    return {
        "status":       "ok",
        "agente":       agente_service.nombre,
        "proveedor":    agente_service.proveedor,
        "modelo":       agente_service.modelo,
        "herramientas": agente_service.herramientas,
    }
