import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routes.chat   import router as chat_router
from routes.agent  import router as agent_router
from routes.health import router as health_router
from routes.rrhh   import router as rrhh_router

load_dotenv()

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "Asistente")

app = FastAPI(
    title=f"API — {NOMBRE_AGENTE}",
    description=f"""
## Agente Inteligente: {NOMBRE_AGENTE}

API REST para interactuar con el agente de IA construido durante el seminario.
    """,
    version="1.0.0",
    contact={
        "name":  "Seminario Agentes IA",
        "email": "tu@email.com",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health_router, tags=["Monitoreo"])
app.include_router(chat_router,   tags=["Chat"])
app.include_router(agent_router,  tags=["Agente"])
app.include_router(rrhh_router,   tags=["RRHH"])

# Carpeta de estáticos (para imágenes, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Frontend principal
FRONTEND_INDEX = Path(__file__).parent / "frontend" / "index.html"

@app.get("/", tags=["Info"])
async def root():
    """Sirve la interfaz de chat"""
    return FileResponse(FRONTEND_INDEX, media_type="text/html; charset=utf-8")

@app.get("/api", tags=["Info"])
async def info():
    return {
        "api":      f"Agente IA — {NOMBRE_AGENTE}",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/health",
    }

if __name__ == "__main__":
    import uvicorn
    host  = os.getenv("API_HOST", "0.0.0.0")
    port  = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "True").lower() == "true"

    uvicorn.run("main:app", host=host, port=port, reload=debug)
