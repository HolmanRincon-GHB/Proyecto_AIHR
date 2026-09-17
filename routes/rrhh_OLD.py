"""
============================================================
  routes/rrhh.py — Endpoint POST /rrhh
  Seminario: Agentes de IA · Sesión 6 (Despliegue)

  Endpoint dedicado al sistema multiagente + RAG sobre
  Gestión de RRHH y Retención de Talento (TechnoDistrib S.A.S.)

  Independiente de /chat y /agent — no comparte memoria ni
  lógica con el agente genérico del seminario.
============================================================
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import RRHHRequest, RRHHResponse, ErrorResponse
from core.rrhh_service import rrhh_service

router = APIRouter()


@router.post(
    "/rrhh",
    response_model=RRHHResponse,
    summary="Consultar al equipo multiagente de RRHH",
    description=(
        "Ejecuta el flujo Coordinador → Investigador (RAG) → Analista → "
        "Redactor sobre documentos internos de RRHH y retención de talento."
    ),
    responses={500: {"model": ErrorResponse}},
)
async def consultar_rrhh(request: RRHHRequest):
    """
    POST /rrhh

    Body:
        pregunta:   Pregunta sobre clima laboral, rotación, desempeño,
                    beneficios o retención de talento
        session_id: Identificador informativo (no se usa memoria aún)

    Returns:
        RRHHResponse con la respuesta del equipo multiagente y sus fuentes
    """
    try:
        resultado = rrhh_service.responder(request.pregunta)
        return RRHHResponse(
            respuesta=resultado["respuesta"],
            pregunta=request.pregunta,
            session_id=request.session_id,
            fuentes=resultado.get("fuentes", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/rrhh/stream",
    summary="Consultar al equipo multiagente de RRHH con progreso en vivo",
    description=(
        "Igual que POST /rrhh, pero transmite (Server-Sent Events) el avance "
        "de cada agente — coordinador, investigador, analista, redactor — "
        "antes de enviar la respuesta final."
    ),
)
async def consultar_rrhh_stream(request: RRHHRequest):
    """
    POST /rrhh/stream

    Body: igual que POST /rrhh (pregunta, session_id).

    Devuelve un stream `text/event-stream` donde cada línea `data: {...}`
    es uno de estos eventos JSON:
        {"tipo": "paso",  "linea": "🎯 [COORDINADOR] → INVESTIGADOR"}
        {"tipo": "final", "respuesta": "...", "fuentes": [...]}
        {"tipo": "error", "mensaje": "..."}
    """
    def generar_eventos():
        try:
            for evento in rrhh_service.responder_stream(request.pregunta):
                yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
        except Exception as e:
            error = {"tipo": "error", "mensaje": str(e)}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generar_eventos(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/rrhh/documentos",
    summary="Listar documentos indexados",
    description="Devuelve los títulos de los documentos internos de RRHH disponibles en el RAG.",
)
async def listar_documentos():
    """GET /rrhh/documentos — inventario de fuentes disponibles"""
    return {"documentos": rrhh_service.documentos_disponibles}
