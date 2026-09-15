"""
============================================================
  05_agente_completo_s4.py — Agente BASE para Sesión 5
  Seminario: Agentes de IA · Sesión 4

  Combina todo lo aprendido en S1 → S4:
    ✅ Switch Groq / Ollama
    ✅ Grafo LangGraph con coordinador
    ✅ Agentes especializados configurables
    ✅ Memoria de conversación
    ✅ Herramientas externas (clima, países, cálculo)
    ✅ Routing inteligente por tipo de tarea
    ✅ Comandos especiales: /grafo /agentes /limpiar

  En la Sesión 5 este agente se envuelve en una API
  REST con FastAPI.

  ✏️ Puntos de modificación marcados en cada sección.

  Uso:
      python 05_agente_completo_s4.py
============================================================
"""

import sys, os, math, requests
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal, Annotated
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from s4_llm_factory import crear_llm

load_dotenv()

# ═══════════════════════════════════════════════════════════
#  ✏️ SECCIÓN 1 — Identidad del sistema
# ═══════════════════════════════════════════════════════════
NOMBRE_SISTEMA = os.getenv("AGENT_NAME", "SistemaIA")
CIUDAD_DEFAULT = os.getenv("CIUDAD_DEFAULT", "Bogota")


# ═══════════════════════════════════════════════════════════
#  ✏️ SECCIÓN 2 — Herramientas disponibles
# ═══════════════════════════════════════════════════════════

@tool
def consultar_clima(ciudad: str) -> str:
    """Clima actual de una ciudad. Para preguntas de temperatura o tiempo."""
    key = os.getenv("OPENWEATHER_API_KEY", "")
    if not key or "TU_CLAVE" in key:
        return "OPENWEATHER_API_KEY no configurada en .env"
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
            params={"q":ciudad,"appid":key,"units":"metric","lang":"es"}, timeout=10)
        d = r.json()
        return f"🌤 {d['name']}: {d['main']['temp']:.1f}°C, {d['weather'][0]['description']}, humedad {d['main']['humidity']}%"
    except Exception as e:
        return f"Error clima: {e}"

@tool
def calculadora(expresion: str) -> str:
    """Calcula expresiones matemáticas. Para cualquier cálculo."""
    try:
        ctx = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        ctx.update({"abs": abs, "round": round})
        return f"{expresion} = {eval(expresion, {'__builtins__': {}}, ctx)}"
    except Exception as e:
        return f"Error: {e}"

@tool
def fecha_hora_actual(zona: str = "America/Bogota") -> str:
    """Fecha y hora actual. Para preguntas sobre tiempo."""
    try:
        return datetime.now(ZoneInfo(zona)).strftime(f"%A %d de %B de %Y, %H:%M en {zona}")
    except Exception:
        return datetime.now().strftime("%A %d de %B de %Y, %H:%M")

@tool
def info_pais(nombre_pais: str) -> str:
    """Información de un país. Para preguntas geográficas."""
    try:
        r = requests.get(f"https://restcountries.com/v3.1/name/{nombre_pais}", timeout=10)
        if r.status_code == 404:
            return f"País no encontrado: {nombre_pais}"
        d = r.json()[0]
        return f"🌍 {d['name']['common']}: cap. {d.get('capital',['?'])[0]}, {d.get('population',0):,} hab."
    except Exception as e:
        return f"Error: {e}"

# ✏️ Lista de tools activas — agrega o quita según tu proyecto
TOOLS = [consultar_clima, calculadora, fecha_hora_actual, info_pais]


# ═══════════════════════════════════════════════════════════
#  STATE DEL GRAFO
# ═══════════════════════════════════════════════════════════
class EstadoSistema(TypedDict):
    mensajes:         Annotated[list, add_messages]
    entrada_usuario:  str
    tipo_tarea:       str   # "conversacion" | "investigacion" | "calculo"
    respuesta_final:  str
    iteraciones:      int


# ═══════════════════════════════════════════════════════════
#  NODOS DEL GRAFO
# ═══════════════════════════════════════════════════════════

def nodo_clasificar_tarea(estado: EstadoSistema) -> dict:
    """Clasifica el tipo de tarea para enrutar al agente correcto."""
    llm  = crear_llm(temperature=0.0)
    resp = llm.invoke([
        SystemMessage(content="""Clasifica en UNA categoría:
- calculo: matemáticas, números, fórmulas
- investigacion: análisis, explicaciones largas, comparaciones
- conversacion: preguntas simples, saludos, consultas directas
Responde SOLO la categoría."""),
        HumanMessage(content=estado["entrada_usuario"]),
    ])
    categoria = resp.content.strip().lower()
    if categoria not in ["calculo", "investigacion", "conversacion"]:
        categoria = "conversacion"
    return {"tipo_tarea": categoria}


def nodo_agente_con_tools(estado: EstadoSistema) -> dict:
    """Agente con herramientas para conversación y cálculos.

    create_agent (LangChain 1.x) reemplaza create_tool_calling_agent
    + AgentExecutor. La memoria se pasa como lista de "messages",
    no con MessagesPlaceholder.
    """
    llm = crear_llm()
    agente = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=f"""Eres {NOMBRE_SISTEMA}, asistente inteligente con herramientas.
Usa las herramientas cuando necesites información real o calcular algo.
Responde en español.""",
    )
    historial = [m for m in estado["mensajes"] if not isinstance(m, str)]
    resultado = agente.invoke({
        "messages": historial + [HumanMessage(content=estado["entrada_usuario"])]
    })
    respuesta = resultado["messages"][-1].content
    return {
        "respuesta_final": respuesta,
        "mensajes": [HumanMessage(content=estado["entrada_usuario"]),
                     AIMessage(content=respuesta)],
    }


def nodo_investigacion(estado: EstadoSistema) -> dict:
    """Agente especializado en análisis profundo."""
    llm  = crear_llm(temperature=0.5)
    # ✏️ Cambia el system prompt para tu dominio de investigación
    resp = llm.invoke([
        SystemMessage(content=f"""Eres {NOMBRE_SISTEMA}, experto en análisis.
Proporciona respuestas detalladas, estructuradas y basadas en evidencia.
Usa encabezados y puntos clave. Responde en español."""),
        *[m for m in estado["mensajes"]],
        HumanMessage(content=estado["entrada_usuario"]),
    ])
    return {
        "respuesta_final": resp.content,
        "mensajes": [HumanMessage(content=estado["entrada_usuario"]),
                     AIMessage(content=resp.content)],
    }


def router_tarea(estado: EstadoSistema) -> Literal["agente_tools", "investigacion"]:
    """Enruta según el tipo de tarea."""
    if estado.get("tipo_tarea") == "investigacion":
        return "investigacion"
    return "agente_tools"


# ═══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO CON MEMORIA PERSISTENTE
# ═══════════════════════════════════════════════════════════
def construir_sistema():
    grafo = StateGraph(EstadoSistema)

    grafo.add_node("clasificar",    nodo_clasificar_tarea)
    grafo.add_node("agente_tools",  nodo_agente_con_tools)
    grafo.add_node("investigacion", nodo_investigacion)

    grafo.add_edge(START, "clasificar")
    grafo.add_conditional_edges("clasificar", router_tarea,
        {"agente_tools": "agente_tools", "investigacion": "investigacion"})
    grafo.add_edge("agente_tools",  END)
    grafo.add_edge("investigacion", END)

    # MemorySaver guarda el estado entre invocaciones
    memoria = MemorySaver()
    return grafo.compile(checkpointer=memoria)


def main():
    app = construir_sistema()

    # ✏️ Cambia el thread_id para simular diferentes usuarios
    THREAD_ID = "usuario_principal"
    config    = {"configurable": {"thread_id": THREAD_ID}}

    print(f"\n{'─'*55}")
    print(f"  🤖  {NOMBRE_SISTEMA}  —  Sistema LangGraph Completo")
    print(f"  Agentes: Clasificador + Tools + Investigación")
    print(f"  Herramientas: {', '.join(t.name for t in TOOLS)}")
    print(f"{'─'*55}")
    print("  Comandos: /grafo | /agentes | /limpiar | salir\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not entrada:
            continue
        if entrada.lower() == "salir":
            break
        if entrada.lower() == "/grafo":
            print("\n  Nodos del grafo: clasificar → [agente_tools | investigacion]\n")
            continue
        if entrada.lower() == "/agentes":
            print(f"\n  Agentes activos:")
            print(f"    • Clasificador → decide el tipo de tarea")
            print(f"    • Agente Tools → conversación + {len(TOOLS)} herramientas")
            print(f"    • Investigación → análisis profundo\n")
            continue
        if entrada.lower() == "/limpiar":
            THREAD_ID_nuevo = f"sesion_{datetime.now().strftime('%H%M%S')}"
            config = {"configurable": {"thread_id": THREAD_ID_nuevo}}
            print(f"\n  🔄 Nueva sesión iniciada.\n")
            continue

        estado_inicial = {
            "mensajes":        [],
            "entrada_usuario": entrada,
            "tipo_tarea":      "",
            "respuesta_final": "",
            "iteraciones":     0,
        }

        try:
            resultado = app.invoke(estado_inicial, config=config)
            print(f"\n{NOMBRE_SISTEMA}: {resultado['respuesta_final']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    print(f"\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()
