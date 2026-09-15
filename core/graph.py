"""
core/graph.py — Grafo coordinador (LangGraph)
Seminario: Agentes de IA · Sesión 6 (Despliegue)

Este es el "cerebro" que junta TODO lo visto en el curso en un solo
punto de entrada: el COORDINADOR lee la pregunta y decide sola cuál
camino tomar — como en NEXUS.

      START → coordinador → [conversacion | documentos | analisis | reporte] → END

  • conversacion → agente con memoria + herramientas       (Sesiones 1-3)
  • documentos   → RAG sobre los PDFs/CSV internos          (Sesión 4)
  • analisis     → equipo Investigador→Analista→Redactor    (Sesión 4)
  • reporte      → Salida Estructurada con Pydantic         (Sesión 2)

✏️ MODIFICA AQUÍ: ajusta las categorías, los prompts o agrega un
   nuevo camino siguiendo el mismo patrón.
"""

import os
import sys
from typing import TypedDict, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

from s6_llm_factory import crear_llm
from core.tools import TOOLS_BASE, buscar_documentos, analizar_ventas
from core.rag import obtener_motor
from core.structured import ReporteEjecutivo

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "Asistente")
ROL_AGENTE = os.getenv("AGENT_ROLE", "asistente de TechnoDistrib S.A.S.")


def limpiar_pensamiento(texto: str) -> str:
    """
    Algunos modelos 'razonadores' (ej. qwen3.6) anteponen un bloque
    <think>...</think> con su razonamiento interno antes de la
    respuesta final. Nos quedamos solo con lo que viene después.
    """
    if texto and "</think>" in texto:
        return texto.split("</think>")[-1].strip()
    return (texto or "").strip()


# ═══════════════════════════════════════════════════════════
#  STATE — el diccionario compartido entre todos los nodos
# ═══════════════════════════════════════════════════════════
class EstadoAgente(TypedDict):
    entrada_usuario: str
    historial: list
    categoria: str
    respuesta_final: str
    fuentes: list


# ═══════════════════════════════════════════════════════════
#  COORDINADOR — clasifica y decide el camino (edge condicional)
# ═══════════════════════════════════════════════════════════
def nodo_coordinador(estado: EstadoAgente) -> dict:
    llm = crear_llm(temperature=0.0)
    instrucciones = """Clasifica la pregunta del usuario en UNA categoría:
- documentos: preguntas sobre políticas, garantías, precios de catálogo,
  indicadores de RRHH o el plan estratégico de la empresa.
- reporte: el usuario pide explícitamente un reporte, informe ejecutivo,
  resumen estructurado, o un "reporte de ventas".
- analisis: el usuario pide analizar, comparar, investigar o evaluar
  algo en profundidad.
- conversacion: cualquier otra cosa (saludos, cálculos, clima, hora,
  preguntas generales).

Responde SOLO con la categoría, en minúsculas, sin explicación ni puntuación.

Ejemplos:
"¿Qué garantía tiene el portátil Dell?" -> documentos
"¿Cuál es la política de devoluciones?" -> documentos
"Genera un reporte ejecutivo de ventas" -> reporte
"Necesito un informe ejecutivo de la situación de RRHH" -> reporte
"Dame un resumen estructurado del negocio" -> reporte
"Analiza el desempeño de ventas por región" -> analisis
"Compara las ventas de Bogotá y Medellín y saca conclusiones" -> analisis
"Investiga qué está pasando con la rotación de personal" -> analisis
"Hola, ¿cómo estás?" -> conversacion
"¿Cuánto es 45 * 12?" -> conversacion
"¿Qué hora es en Bogotá?" -> conversacion
"¿Cuáles fueron las ventas totales de mayo?" -> conversacion"""
    resp = llm.invoke([
        SystemMessage(content=instrucciones),
        HumanMessage(content=estado["entrada_usuario"]),
    ])
    contenido = limpiar_pensamiento(resp.content).lower()

    categoria = "conversacion"
    for opcion in ("documentos", "reporte", "analisis", "conversacion"):
        if opcion in contenido:
            categoria = opcion
            break

    print(f"  🎯 [COORDINADOR] → {categoria.upper()}")
    return {"categoria": categoria}


def router(estado: EstadoAgente) -> Literal["conversacion", "documentos", "reporte", "analisis"]:
    return estado.get("categoria", "conversacion")


# ═══════════════════════════════════════════════════════════
#  NODO: CONVERSACIÓN — agente con memoria + herramientas
# ═══════════════════════════════════════════════════════════
def nodo_conversacion(estado: EstadoAgente) -> dict:
    llm = crear_llm()
    system_prompt = f"""Eres {NOMBRE_AGENTE}, un {ROL_AGENTE}.
Responde siempre en español, claro y directo.
Usa tus herramientas cuando las necesites en vez de inventar datos."""
    agente = create_agent(model=llm, tools=TOOLS_BASE, system_prompt=system_prompt)
    resultado = agente.invoke({
        "messages": estado["historial"] + [HumanMessage(content=estado["entrada_usuario"])]
    })
    return {"respuesta_final": limpiar_pensamiento(resultado["messages"][-1].content)}


# ═══════════════════════════════════════════════════════════
#  NODO: DOCUMENTOS — RAG sobre los PDFs/CSV internos
# ═══════════════════════════════════════════════════════════
def nodo_documentos(estado: EstadoAgente) -> dict:
    motor = obtener_motor()
    contexto, fuentes = motor.buscar(estado["entrada_usuario"])

    llm = crear_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres {NOMBRE_AGENTE}, asistente de TechnoDistrib S.A.S.
Responde ÚNICAMENTE basándote en el contexto de los documentos internos
proporcionado. Si la información no está en el contexto, dilo claramente
en vez de inventar. Cita el archivo/página cuando sea posible.
Responde en español, claro y profesional."""),
        ("human", "Contexto de los documentos internos:\n{contexto}\n\nPregunta: {pregunta}"),
    ])
    chain = prompt | llm | StrOutputParser()
    respuesta = chain.invoke({
        "contexto": contexto or "(no se encontró contexto relevante)",
        "pregunta": estado["entrada_usuario"],
    })
    return {"respuesta_final": limpiar_pensamiento(respuesta), "fuentes": fuentes}


# ═══════════════════════════════════════════════════════════
#  NODO: ANÁLISIS — equipo Investigador → Analista → Redactor
# ═══════════════════════════════════════════════════════════
def nodo_analisis(estado: EstadoAgente) -> dict:
    print("  🔍 [INVESTIGADOR] Recopilando información...")
    agente_investigador = create_agent(
        model=crear_llm(temperature=0.3),
        tools=[buscar_documentos, analizar_ventas],
        system_prompt="""Eres un investigador. Usa tus herramientas para
recopilar datos reales relevantes a la pregunta ANTES de concluir nada.
Sé concreto. Responde en español, en lista de puntos clave.""",
    )
    investigacion = limpiar_pensamiento(agente_investigador.invoke({
        "messages": [HumanMessage(content=f"Investiga sobre: {estado['entrada_usuario']}")]
    })["messages"][-1].content)

    print("  📊 [ANALISTA] Analizando información...")
    analisis = limpiar_pensamiento(crear_llm(temperature=0.4).invoke([
        SystemMessage(content="""Eres un analista crítico. Extrae lo más
importante, identifica riesgos u oportunidades, y prioriza los hallazgos.
Responde en español, estructurado."""),
        HumanMessage(content=f"Pregunta: {estado['entrada_usuario']}\n\n"
                              f"Información recopilada:\n{investigacion}\n\nAnaliza y concluye."),
    ]).content)

    print("  ✍️  [REDACTOR] Redactando respuesta final...")
    redaccion = limpiar_pensamiento(crear_llm(temperature=0.5).invoke([
        SystemMessage(content=f"""Eres {NOMBRE_AGENTE}. Toma el análisis
recibido y conviértelo en una respuesta final clara y bien organizada
para el usuario. Responde en español."""),
        HumanMessage(content=f"Pregunta original: {estado['entrada_usuario']}\n\n"
                              f"Análisis:\n{analisis}\n\nRedacta la respuesta final."),
    ]).content)

    return {"respuesta_final": redaccion}


# ═══════════════════════════════════════════════════════════
#  NODO: REPORTE — Salida Estructurada con Pydantic
# ═══════════════════════════════════════════════════════════
def nodo_reporte(estado: EstadoAgente) -> dict:
    # ✏️ Algunos modelos (ej. qwen3.6) fallan con with_structured_output
    # ("tool_use_failed"). GROQ_MODEL_ESTRUCTURADO permite usar un
    # modelo distinto solo para este nodo — por defecto, gpt-oss-120b.
    modelo_estructurado = os.getenv("GROQ_MODEL_ESTRUCTURADO", "openai/gpt-oss-120b")
    llm_estructurado = crear_llm(temperature=0.4, modelo=modelo_estructurado).with_structured_output(ReporteEjecutivo)
    datos_ventas = analizar_ventas.invoke({"agrupar_por": "region"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Genera un reporte ejecutivo de TechnoDistrib S.A.S.
basado en los datos disponibles. Sé específico y accionable.
Responde en español."""),
        ("human", "Solicitud: {solicitud}\n\nDatos disponibles:\n{datos}"),
    ])
    chain = prompt | llm_estructurado
    reporte: ReporteEjecutivo = chain.invoke({
        "solicitud": estado["entrada_usuario"],
        "datos": datos_ventas,
    })

    salida = f"📋 REPORTE EJECUTIVO — TechnoDistrib S.A.S.\n\n{reporte.resumen_ejecutivo}\n"
    if reporte.alertas:
        salida += "\nALERTAS:\n"
        for a in reporte.alertas:
            salida += f"  • [{a.area}] {a.descripcion}\n    → {a.accion_recomendada}\n"
    if reporte.oportunidades:
        salida += "\nOPORTUNIDADES:\n" + "\n".join(f"  • {o}" for o in reporte.oportunidades) + "\n"
    if reporte.recomendaciones:
        salida += "\nRECOMENDACIONES:\n" + "\n".join(f"  • {r}" for r in reporte.recomendaciones)

    return {"respuesta_final": salida}


# ═══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO
# ═══════════════════════════════════════════════════════════
def construir_grafo():
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("coordinador", nodo_coordinador)
    grafo.add_node("conversacion", nodo_conversacion)
    grafo.add_node("documentos", nodo_documentos)
    grafo.add_node("analisis", nodo_analisis)
    grafo.add_node("reporte", nodo_reporte)

    grafo.add_edge(START, "coordinador")
    grafo.add_conditional_edges("coordinador", router, {
        "conversacion": "conversacion",
        "documentos": "documentos",
        "analisis": "analisis",
        "reporte": "reporte",
    })
    grafo.add_edge("conversacion", END)
    grafo.add_edge("documentos", END)
    grafo.add_edge("analisis", END)
    grafo.add_edge("reporte", END)

    return grafo.compile()
