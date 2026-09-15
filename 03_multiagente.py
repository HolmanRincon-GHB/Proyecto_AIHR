"""
============================================================
  03_multiagente.py — Sistema con 4 agentes coordinados
  Seminario: Agentes de IA · Sesión 4

  Sistema multiagente con roles especializados:

    ┌─────────────┐
    │ COORDINADOR │  ← decide qué agente activar
    └──────┬──────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
  INVESTIGADOR  ANALISTA  REDACTOR
  (busca info) (analiza) (escribe)
    └──────┴──────────────┘
           │
           ▼
        RESPUESTA FINAL

  Cada agente tiene:
    - Su propio system prompt (personalidad y rol)
    - Acceso al historial compartido
    - Capacidad de pasar el trabajo al siguiente

  ✏️ Puntos de personalización marcados en cada agente.

  Uso:
      python 03_multiagente.py
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from s4_llm_factory import crear_llm

load_dotenv()


# ═══════════════════════════════════════════════════════════
#  STATE COMPARTIDO
#  Todos los agentes leen y escriben en este diccionario.
#  add_messages es un reducer especial que acumula mensajes
#  en lugar de sobreescribirlos.
#
#  ✏️ MODIFICA AQUÍ: agrega campos según tu caso de uso
# ═══════════════════════════════════════════════════════════
class EstadoMultiagente(TypedDict):
    mensajes:          Annotated[list, add_messages]  # historial acumulativo
    tarea_original:    str    # lo que pidió el usuario
    siguiente_agente:  str    # quién actúa next: investigador|analista|redactor|fin
    contexto_reunido:  str    # información recopilada por el investigador
    analisis:          str    # análisis del analista
    respuesta_final:   str    # producto del redactor


# ═══════════════════════════════════════════════════════════
#  AGENTE COORDINADOR
#  Decide qué agente debe actuar según el estado actual.
#  ✏️ MODIFICA AQUÍ: ajusta la lógica de coordinación
# ═══════════════════════════════════════════════════════════
def agente_coordinador(estado: EstadoMultiagente) -> dict:
    """
    El coordinador analiza la tarea y decide el flujo.
    Primera vez: manda al investigador.
    Si hay contexto pero no análisis: manda al analista.
    Si hay análisis: manda al redactor.
    Si hay respuesta final: termina.
    """
    print("\n  🎯 [COORDINADOR] Evaluando estado del sistema...")

    tiene_contexto  = bool(estado.get("contexto_reunido", "").strip())
    tiene_analisis  = bool(estado.get("analisis", "").strip())
    tiene_respuesta = bool(estado.get("respuesta_final", "").strip())

    if tiene_respuesta:
        siguiente = "fin"
    elif tiene_analisis:
        siguiente = "redactor"
    elif tiene_contexto:
        siguiente = "analista"
    else:
        siguiente = "investigador"

    print(f"  🎯 [COORDINADOR] → Siguiente agente: {siguiente.upper()}")
    return {"siguiente_agente": siguiente}


# ═══════════════════════════════════════════════════════════
#  AGENTE INVESTIGADOR
#  Recopila información relevante sobre la tarea.
#  ✏️ MODIFICA AQUÍ: conecta a fuentes reales (APIs, RAG, etc.)
# ═══════════════════════════════════════════════════════════
def agente_investigador(estado: EstadoMultiagente) -> dict:
    """Recopila y organiza información sobre la tarea."""
    print("\n  🔍 [INVESTIGADOR] Recopilando información...")

    llm = crear_llm(temperature=0.3)

    # ✏️ Cambia este system prompt para tu dominio
    system = """Eres un investigador experto. Tu trabajo es:
1. Identificar los aspectos clave de la tarea
2. Organizar el conocimiento relevante disponible
3. Señalar qué información adicional podría ser útil
Sé exhaustivo pero conciso. Responde en español."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Tarea a investigar: {estado['tarea_original']}\n\nOrgánica la información clave sobre este tema."),
    ])

    print(f"  🔍 [INVESTIGADOR] Contexto recopilado ({len(respuesta.content)} chars)")
    return {
        "contexto_reunido": respuesta.content,
        "mensajes": [AIMessage(content=f"[Investigador]: {respuesta.content[:200]}...")],
    }


# ═══════════════════════════════════════════════════════════
#  AGENTE ANALISTA
#  Analiza el contexto y extrae conclusiones.
#  ✏️ MODIFICA AQUÍ: adapta el tipo de análisis
# ═══════════════════════════════════════════════════════════
def agente_analista(estado: EstadoMultiagente) -> dict:
    """Analiza el contexto recopilado y genera insights."""
    print("\n  📊 [ANALISTA] Analizando información...")

    llm = crear_llm(temperature=0.4)

    # ✏️ Cambia el enfoque del análisis
    system = """Eres un analista experto. Tu trabajo es:
1. Analizar la información proporcionada críticamente
2. Identificar patrones, fortalezas y debilidades
3. Extraer conclusiones accionables
4. Priorizar los puntos más importantes
Responde en español con estructura clara."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Tarea original: {estado['tarea_original']}

Información recopilada:
{estado['contexto_reunido']}

Realiza un análisis crítico y extrae las conclusiones más importantes."""),
    ])

    print(f"  📊 [ANALISTA] Análisis completado ({len(respuesta.content)} chars)")
    return {
        "analisis": respuesta.content,
        "mensajes": [AIMessage(content=f"[Analista]: {respuesta.content[:200]}...")],
    }


# ═══════════════════════════════════════════════════════════
#  AGENTE REDACTOR
#  Genera la respuesta final para el usuario.
#  ✏️ MODIFICA AQUÍ: cambia el formato y tono de la respuesta
# ═══════════════════════════════════════════════════════════
def agente_redactor(estado: EstadoMultiagente) -> dict:
    """Redacta la respuesta final clara y útil para el usuario."""
    print("\n  ✍️  [REDACTOR] Elaborando respuesta final...")

    llm = crear_llm(temperature=0.6)

    # ✏️ Ajusta el formato de salida deseado
    system = """Eres un redactor experto. Tu trabajo es:
1. Sintetizar la investigación y el análisis
2. Crear una respuesta clara, útil y bien estructurada
3. Adaptar el lenguaje al usuario final
4. Ser directo y evitar repetición innecesaria
Responde en español. Usa formato Markdown si es útil."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Solicitud del usuario: {estado['tarea_original']}

Investigación realizada:
{estado['contexto_reunido']}

Análisis:
{estado['analisis']}

Redacta la respuesta final al usuario."""),
    ])

    print(f"  ✍️  [REDACTOR] Respuesta final lista")
    return {
        "respuesta_final": respuesta.content,
        "mensajes": [AIMessage(content=respuesta.content)],
    }


# ═══════════════════════════════════════════════════════════
#  ROUTER DEL COORDINADOR
# ═══════════════════════════════════════════════════════════
def router_coordinador(estado: EstadoMultiagente) -> Literal[
    "investigador", "analista", "redactor", "__end__"
]:
    sig = estado.get("siguiente_agente", "investigador")
    if sig == "fin":
        return "__end__"
    return sig


# ═══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO
# ═══════════════════════════════════════════════════════════
def construir_grafo():
    grafo = StateGraph(EstadoMultiagente)

    # Registrar nodos
    grafo.add_node("coordinador",  agente_coordinador)
    grafo.add_node("investigador", agente_investigador)
    grafo.add_node("analista",     agente_analista)
    grafo.add_node("redactor",     agente_redactor)

    # Flujo: START → coordinador (siempre primero)
    grafo.add_edge(START, "coordinador")

    # El coordinador decide a dónde ir
    grafo.add_conditional_edges(
        "coordinador",
        router_coordinador,
        {
            "investigador": "investigador",
            "analista":     "analista",
            "redactor":     "redactor",
            "__end__":      END,
        }
    )

    # Después de cada agente especializado → vuelve al coordinador
    # Esto permite el ciclo: coordinador → especialista → coordinador → ...
    grafo.add_edge("investigador", "coordinador")
    grafo.add_edge("analista",     "coordinador")
    grafo.add_edge("redactor",     "coordinador")

    return grafo.compile()


def main():
    app = construir_grafo()

    print("\n" + "═" * 55)
    print("  Sistema Multiagente — 4 agentes coordinados")
    print("  Coordinador → Investigador → Analista → Redactor")
    print("═" * 55)
    print("  Prueba con tareas que requieran análisis:")
    print("    • 'Explícame las ventajas de LangGraph vs LangChain'")
    print("    • 'Analiza el impacto de la IA en la educación'")
    print("    • '¿Qué framework debo aprender primero para IA?'")
    print("  Para salir: salir\n")

    while True:
        try:
            tarea = input("Tarea: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not tarea or tarea.lower() == "salir":
            break

        print(f"\n  Procesando con 4 agentes...\n")

        estado_inicial = {
            "mensajes":         [],
            "tarea_original":   tarea,
            "siguiente_agente": "",
            "contexto_reunido": "",
            "analisis":         "",
            "respuesta_final":  "",
        }

        try:
            resultado = app.invoke(estado_inicial)
            print("\n" + "─" * 55)
            print("  RESPUESTA FINAL:")
            print("─" * 55)
            print(resultado["respuesta_final"])
            print("─" * 55 + "\n")
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")

    print("\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()
