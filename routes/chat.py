"""
============================================================
  routes/chat.py — Endpoint POST /chat
  Seminario: Agentes de IA · Sesión 5

  Maneja conversaciones con el agente.
  Mantiene memoria por session_id.
============================================================
"""

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, ErrorResponse
from core.agent_core import agente_service
import os

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Conversar con el agente",
    description="Envía un mensaje al agente y recibe su respuesta. Mantiene memoria por session_id.",
    responses={500: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest):
    """
    POST /chat

    Body:
        mensaje:    El texto del usuario
        session_id: Identificador de sesión (opcional, default: "default")

    Returns:
        ChatResponse con la respuesta del agente
    """
    try:
        resultado = agente_service.chat(
            mensaje=request.mensaje,
            session_id=request.session_id,
        )
        return ChatResponse(
            respuesta=resultado["respuesta"],
            session_id=request.session_id,
            modelo=agente_service.modelo,
            categoria=resultado.get("categoria"),
            fuentes=resultado.get("fuentes", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✏️ MODIFICA AQUÍ: agrega endpoints relacionados con chat
@router.delete(
    "/chat/{session_id}",
    summary="Limpiar sesión",
    description="Elimina el historial de conversación de una sesión.",
)
async def limpiar_sesion(session_id: str):
    """DELETE /chat/{session_id} — limpia el historial"""
    eliminado = agente_service.limpiar_sesion(session_id)
    if eliminado:
        return {"mensaje": f"Sesión '{session_id}' eliminada correctamente"}
    return {"mensaje": f"Sesión '{session_id}' no encontrada"}
