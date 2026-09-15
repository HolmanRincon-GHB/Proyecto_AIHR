"""
core/tools.py — Herramientas del agente
Seminario: Agentes de IA · Sesión 6 (Despliegue)

Reúne las herramientas vistas en el curso:
  - calculadora, fecha_hora_actual, consultar_clima  (Sesión 2-3)
  - buscar_documentos                                (RAG — Sesión 4)
  - analizar_ventas                                  (datos reales en CSV)
"""

import os
import sys
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
from langchain_core.tools import tool

from core.rag import obtener_motor


@tool
def calculadora(expresion: str) -> str:
    """Evalúa expresiones matemáticas. Úsala para cualquier cálculo numérico.
    Ejemplos: '2**10', '1024 / 8', 'sqrt(256)'"""
    try:
        ctx = {"sqrt": math.sqrt, "log": math.log, "pi": math.pi,
               "e": math.e, "abs": abs, "round": round,
               "sin": math.sin, "cos": math.cos, "tan": math.tan}
        return f"{expresion} = {eval(expresion, {'__builtins__': {}}, ctx)}"
    except Exception as e:
        return f"Error al calcular: {e}"


@tool
def fecha_hora_actual(zona: str = "America/Bogota") -> str:
    """Devuelve la fecha y hora actual. Úsala cuando pregunten qué hora o qué día es."""
    try:
        now = datetime.now(ZoneInfo(zona))
        return now.strftime(f"%A %d de %B de %Y, %H:%M en {zona}")
    except Exception:
        return datetime.now().strftime("%A %d de %B de %Y, %H:%M")


@tool
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima actual de una ciudad. Úsala para temperatura o condiciones climáticas."""
    key = os.getenv("OPENWEATHER_API_KEY", "")
    if not key or "TU_CLAVE" in key:
        return "La herramienta de clima no está configurada (falta OPENWEATHER_API_KEY)."
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": ciudad, "appid": key, "units": "metric", "lang": "es"},
            timeout=10,
        )
        if r.status_code == 404:
            return f"Ciudad '{ciudad}' no encontrada."
        r.raise_for_status()
        d = r.json()
        return (
            f"{d['name']}, {d['sys']['country']}: {d['main']['temp']:.1f}°C, "
            f"{d['weather'][0]['description']}, humedad {d['main']['humidity']}%"
        )
    except Exception as e:
        return f"Error al consultar el clima: {e}"


@tool
def buscar_documentos(pregunta: str) -> str:
    """
    Busca en los documentos internos de TechnoDistrib S.A.S.: política
    comercial, informe de RRHH, protocolo de atención al cliente, plan
    estratégico y catálogo de productos. Úsala para preguntas sobre
    políticas, garantías, precios de catálogo, indicadores de RRHH o
    la estrategia de la empresa.
    """
    motor = obtener_motor()
    texto, fuentes = motor.buscar(pregunta)
    if not texto:
        return "No se encontró información relevante en los documentos internos."
    return texto + "\n\nFuentes consultadas: " + ", ".join(fuentes)


@tool
def analizar_ventas(agrupar_por: str = "region") -> str:
    """
    Analiza las ventas reales de TechnoDistrib S.A.S. desde el archivo
    de ventas regionales. agrupar_por puede ser: 'region', 'producto' o 'mes'.
    Úsala para preguntas sobre cifras de ventas, qué región vende más,
    producto más vendido, o tendencias por mes.
    """
    ruta = Path(os.getenv("DOCUMENTOS_DIR", "data/documentos")) / "06_ventas_regionales.csv"
    if not ruta.exists():
        return "No se encontró el archivo de ventas."

    df = pd.read_csv(ruta, parse_dates=["fecha"])
    agrupar_por = agrupar_por.strip().lower()

    if agrupar_por == "mes":
        df["mes"] = df["fecha"].dt.strftime("%Y-%m")
        resumen = df.groupby("mes")["valor_total_cop"].sum().sort_index()
    elif agrupar_por == "producto":
        resumen = df.groupby("producto")["valor_total_cop"].sum().sort_values(ascending=False)
    else:
        agrupar_por = "region"
        resumen = df.groupby("region")["valor_total_cop"].sum().sort_values(ascending=False)

    total = df["valor_total_cop"].sum()
    lineas = [f"- {idx}: ${val:,.0f} COP" for idx, val in resumen.items()]
    return f"Ventas totales del periodo: ${total:,.0f} COP\n\nDesglose por {agrupar_por}:\n" + "\n".join(lineas)


# ✏️ MODIFICA AQUÍ: agrega o quita herramientas del agente conversacional
TOOLS_BASE = [calculadora, fecha_hora_actual, consultar_clima, buscar_documentos, analizar_ventas]
