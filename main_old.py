"""
============================================================
  main.py — Servidor FastAPI principal
  Seminario: Agentes de IA · Sesión 6 (Despliegue)

  Punto de entrada de la aplicación.
  Registra todos los routers, sirve el frontend de chat
  (frontend/index.html) y configura la API.

  Un solo despliegue en Railway sirve las dos cosas:
    - La interfaz de chat en  /
    - La API REST en          /chat, /agent, /health...
    - La documentación en     /docs

  Arrancar el servidor:
      uvicorn main:app --reload --port 8000

  Ver documentación Swagger:
      http://localhost:8000/docs

  ✏️ MODIFICA AQUÍ: título, descripción y versión de la API
============================================================
"""

import os
import sys
from pathlib import Path

# ── Fix Windows + uvicorn --reload ──────────────────────────
# El proceso hijo que crea --reload (via multiprocessing/spawn)
# no siempre hereda la carpeta del proyecto en sys.path, lo que
# rompe imports locales como "s6_llm_factory" o "core.graph".
# Esta línea lo garantiza sin importar cómo se lance el proceso.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from routes.chat   import router as chat_router
from routes.agent  import router as agent_router
from routes.health import router as health_router
from routes.rrhh   import router as rrhh_router


from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


load_dotenv()

# ═══════════════════════════════════════════════════════════
#  ✏️ MODIFICA AQUÍ — Metadata de tu API
#  Esto aparece en la documentación Swagger
# ═══════════════════════════════════════════════════════════
NOMBRE_AGENTE = os.getenv("AGENT_NAME", "Asistente")

app = FastAPI(
    title=f"API — {NOMBRE_AGENTE}",
    description=f"""
## Agente Inteligente: {NOMBRE_AGENTE}

API REST para interactuar con el agente de IA construido durante el seminario.

### Endpoints disponibles:
- **POST /chat** — Conversar con el agente (con memoria por sesión)
- **POST /agent** — Ejecutar tareas específicas con contexto
- **POST /rrhh** — Consultar al equipo multiagente de RRHH (clima laboral, rotación, desempeño, retención)
- **GET /rrhh/documentos** — Ver documentos internos de RRHH indexados
- **GET /health** — Verificar estado del servicio
- **GET /health/detail** — Estado detallado con herramientas

### Documentación:
- Swagger UI: `/docs`
- ReDoc: `/redoc`
    """,
    version="1.0.0",
    # ✏️ MODIFICA AQUÍ: cambia los metadatos
    contact={
        "name":  "Seminario Agentes IA",
        "email": "tu@email.com",
    },
    license_info={
        "name": "MIT",
    },
)

# ═══════════════════════════════════════════════════════════
#  CORS — permite que frontends externos llamen a la API
#  ✏️ MODIFICA AQUÍ: restringe los orígenes en producción
# ═══════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ✏️ en producción: ["https://tusitioweb.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ═══════════════════════════════════════════════════════════
#  ROUTERS — registra todos los endpoints
# ═══════════════════════════════════════════════════════════
app.include_router(health_router, tags=["Monitoreo"])
app.include_router(chat_router,   tags=["Chat"])
app.include_router(agent_router,  tags=["Agente"])
app.include_router(rrhh_router,   tags=["RRHH"])




# ═══════════════════════════════════════════════════════════
#  FRONTEND — sirve la interfaz de chat en la ruta raíz
#  El archivo vive en frontend/index.html; ver ese archivo
#  para modificar la interfaz visual del chat.
# ═══════════════════════════════════════════════════════════
FRONTEND_INDEX = Path(__file__).parent / "frontend" / "index.html"


@app.get("/", tags=["Info"])
async def root():
    """GET / — sirve la interfaz de chat (frontend/index.html)"""
    return FileResponse(FRONTEND_INDEX, media_type="text/html; charset=utf-8")


@app.get("/api", tags=["Info"])
async def info():
    """GET /api — información básica de la API, en JSON"""
    return {
        "api":      f"Agente IA — {NOMBRE_AGENTE}",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/health",
    }


# ═══════════════════════════════════════════════════════════
#  ARRANQUE DIRECTO
#  Ejecuta con:  python main.py
#  O con:        uvicorn main:app --reload
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    host  = os.getenv("API_HOST", "0.0.0.0")
    port  = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "True").lower() == "true"

    print(f"\n{'─'*50}")
    print(f"  🚀 Servidor arrancando...")
    print(f"  Agente:  {NOMBRE_AGENTE}")
    print(f"  URL:     http://localhost:{port}")
    print(f"  Docs:    http://localhost:{port}/docs")
    print(f"{'─'*50}\n")

    uvicorn.run("main:app", host=host, port=port, reload=debug)
