"""
============================================================
  models/schemas.py — Modelos Pydantic de entrada y salida
  Seminario: Agentes de IA · Sesión 5

  Define los tipos de datos que la API acepta y devuelve.
  FastAPI usa estos modelos para:
    - Validar automáticamente los datos entrantes
    - Generar documentación Swagger automática
    - Serializar las respuestas a JSON

  ✏️ MODIFICA AQUÍ: agrega o quita campos según tu caso de uso.
============================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────────────────
#  MODELOS DE ENTRADA (Request)
#  Lo que el cliente envía al servidor
# ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Cuerpo del POST /chat"""
    # ✏️ MODIFICA AQUÍ: agrega campos si necesitas más contexto
    mensaje:    str  = Field(..., description="Mensaje del usuario", min_length=1, max_length=2000)
    session_id: str  = Field(default="default", description="ID de sesión para memoria")

    class Config:
        json_schema_extra = {
            "example": {
                "mensaje":    "¿Qué temperatura hace en Bogotá?",
                "session_id": "usuario_001"
            }
        }


class AgentRequest(BaseModel):
    """Cuerpo del POST /agent — para tareas específicas"""
    # ✏️ MODIFICA AQUÍ: define los campos de tu tarea
    tarea:      str  = Field(..., description="Tarea a ejecutar por el agente")
    contexto:   Optional[str] = Field(default=None, description="Contexto adicional opcional")
    session_id: str  = Field(default="default", description="ID de sesión")

    class Config:
        json_schema_extra = {
            "example": {
                "tarea":    "Analiza las ventajas de los implantes dentales vs prótesis removibles",
                "contexto": "Paciente de 55 años con buena salud ósea",
                "session_id": "caso_clinico_001"
            }
        }


# ─────────────────────────────────────────────────────────
#  MODELOS DE SALIDA (Response)
#  Lo que el servidor devuelve al cliente
# ─────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Respuesta del POST /chat"""
    # ✏️ MODIFICA AQUÍ: agrega campos de respuesta si necesitas
    respuesta:  str            = Field(..., description="Respuesta del agente")
    session_id: str            = Field(..., description="ID de sesión usado")
    timestamp:  datetime       = Field(default_factory=datetime.now)
    modelo:     str            = Field(..., description="Modelo LLM usado")
    categoria:  Optional[str]  = Field(default=None, description="Camino que tomó el coordinador: conversacion | documentos | analisis | reporte")
    fuentes:    List[str]      = Field(default_factory=list, description="Fuentes citadas cuando la respuesta usó RAG")

    class Config:
        json_schema_extra = {
            "example": {
                "respuesta":  "En Bogotá actualmente hay 16°C con cielo parcialmente nublado.",
                "session_id": "usuario_001",
                "timestamp":  "2025-01-15T10:30:00",
                "modelo":     "openai/gpt-oss-120b",
                "categoria":  "conversacion",
                "fuentes":    []
            }
        }


class AgentResponse(BaseModel):
    """Respuesta del POST /agent"""
    resultado:  str            = Field(..., description="Resultado de la tarea")
    tarea:      str            = Field(..., description="Tarea ejecutada")
    session_id: str            = Field(..., description="ID de sesión")
    timestamp:  datetime       = Field(default_factory=datetime.now)
    categoria:  Optional[str]  = Field(default=None, description="Camino que tomó el coordinador")
    fuentes:    List[str]      = Field(default_factory=list, description="Fuentes citadas cuando aplique")


class HealthResponse(BaseModel):
    """Respuesta del GET /health"""
    status:   str = Field(..., description="Estado del servicio")
    agente:   str = Field(..., description="Nombre del agente")
    proveedor:str = Field(..., description="Proveedor LLM activo")
    version:  str = Field(default="1.0.0")

    class Config:
        json_schema_extra = {
            "example": {
                "status":    "ok",
                "agente":    "MiAgente",
                "proveedor": "groq",
                "version":   "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Respuesta de error estándar"""
    error:   str = Field(..., description="Descripción del error")
    detalle: Optional[str] = Field(default=None, description="Detalle técnico")
