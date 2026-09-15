# ============================================================
#  multiagente_rrhh.py — Sistema multiagente con RAG
#  Tema: Gestión de RRHH y retención de talento
# ============================================================

import sys, os
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from s4_llm_factory import crear_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# ✅ Librerías actualizadas
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
#  Preparar documentos y vector store (global)
# ============================================================
def preparar_vector_store():
    documentos = [
        Document(
            page_content="""INFORME DE GESTIÓN RRHH — Primer Semestre 2024
            Total empleados: 287
            Rotación voluntaria: 8.3% (objetivo: 7%)
            23 empleados en riesgo de renuncia, concentrados en Ventas y Logística.
            Brecha salarial del 18% en vendedores senior.
            Clima laboral bajo en Operaciones (6.1/10).""",
            metadata={"titulo": "Informe RRHH"}
        )
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    return Chroma.from_documents(documents=chunks, embedding=embeddings)

# Crear vector_store global
vector_store = preparar_vector_store()

# ============================================================
#  Estado compartido
# ============================================================
class EstadoRRHH(TypedDict):
    mensajes:          Annotated[list, add_messages]
    tarea_original:    str
    siguiente_agente:  str
    contexto_reunido:  str
    analisis:          str
    respuesta_final:   str

# ============================================================
#  Agente Coordinador
# ============================================================
def agente_coordinador(estado: EstadoRRHH) -> dict:
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

    return {"siguiente_agente": siguiente}

# ============================================================
#  Agente Investigador (usa RAG)
# ============================================================
def agente_investigador(estado: EstadoRRHH) -> dict:
    llm = crear_llm(temperature=0.2)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un investigador de RRHH.
Usa SOLO la información de los documentos internos.
Si no está en el contexto, dilo claramente.
Responde en español."""),
        ("human", """Contexto de documentos:
{contexto}

Pregunta: {pregunta}"""),
    ])

    def formatear_contexto(docs) -> str:
        resultado = ""
        for i, doc in enumerate(docs, 1):
            titulo = doc.metadata.get("titulo", "Documento")
            resultado += f"\n--- {titulo} ---\n"
            resultado += doc.page_content + "\n"
        return resultado

    chain = (
        {
            "contexto": retriever | formatear_contexto,
            "pregunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    respuesta = chain.invoke(estado["tarea_original"])
    return {
        "contexto_reunido": respuesta,
        "mensajes": [AIMessage(content=f"[Investigador]: {respuesta[:200]}...")],
    }

# ============================================================
#  Agente Analista
# ============================================================
def agente_analista(estado: EstadoRRHH) -> dict:
    llm = crear_llm(temperature=0.4)
    system = """Eres un analista de RRHH.
Tu trabajo es:
1. Evaluar críticamente la información recuperada
2. Identificar riesgos de rotación y fuga de talento
3. Proponer conclusiones accionables
Responde en español con estructura clara."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Tarea original: {estado['tarea_original']}

Información recopilada:
{estado['contexto_reunido']}

Realiza un análisis crítico y extrae las conclusiones más importantes."""),
    ])

    return {
        "analisis": respuesta.content,
        "mensajes": [AIMessage(content=f"[Analista]: {respuesta.content[:200]}...")],
    }

# ============================================================
#  Agente Redactor
# ============================================================
def agente_redactor(estado: EstadoRRHH) -> dict:
    llm = crear_llm(temperature=0.6)
    system = """Eres un redactor experto en RRHH.
Tu trabajo es:
1. Sintetizar investigación y análisis
2. Crear un informe claro y útil
3. Adaptar el lenguaje al comité directivo
Responde en español. Usa formato Markdown si es útil."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Solicitud: {estado['tarea_original']}

Investigación:
{estado['contexto_reunido']}

Análisis:
{estado['analisis']}

Redacta la respuesta final."""),
    ])

    return {
        "respuesta_final": respuesta.content,
        "mensajes": [AIMessage(content=respuesta.content)],
    }

# ============================================================
#  Construcción del grafo
# ============================================================
def router_coordinador(estado: EstadoRRHH) -> Literal["investigador","analista","redactor","__end__"]:
    sig = estado.get("siguiente_agente", "investigador")
    if sig == "fin":
        return "__end__"
    return sig

def construir_grafo():
    grafo = StateGraph(EstadoRRHH)
    grafo.add_node("coordinador", agente_coordinador)
    grafo.add_node("investigador", agente_investigador)
    grafo.add_node("analista", agente_analista)
    grafo.add_node("redactor", agente_redactor)

    grafo.add_edge(START, "coordinador")
    grafo.add_conditional_edges("coordinador", router_coordinador, {
        "investigador": "investigador",
        "analista": "analista",
        "redactor": "redactor",
        "__end__": END,
    })
    grafo.add_edge("investigador", "coordinador")
    grafo.add_edge("analista", "coordinador")
    grafo.add_edge("redactor", "coordinador")
    return grafo.compile()

# ============================================================
#  Main
# ============================================================
def main():
    app = construir_grafo()
    tarea = "Analiza la situación de rotación de empleados y propone un plan de retención"
    estado_inicial = {
        "mensajes": [],
        "tarea_original": tarea,
        "siguiente_agente": "",
        "contexto_reunido": "",
        "analisis": "",
        "respuesta_final": "",
    }
    resultado = app.invoke(estado_inicial)
    print("RESPUESTA FINAL:\n")
    print(resultado["respuesta_final"])

if __name__ == "__main__":
    main()


def consulta_rrhh(pregunta: str):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente de RRHH.
Responde SOLO con base en los documentos internos.
Si no está en el contexto, dilo claramente.
Responde en español."""),
        ("human", """Contexto:
{contexto}

Pregunta: {pregunta}"""),
    ])

    def formatear_contexto(docs) -> str:
        resultado = ""
        for i, doc in enumerate(docs, 1):
            titulo = doc.metadata.get("titulo", "Documento")
            resultado += f"\n--- {titulo} ---\n"
            resultado += doc.page_content + "\n"
        return resultado

    chain = (
        {
            "contexto": retriever | formatear_contexto,
            "pregunta": RunnablePassthrough(),
        }
        | prompt
        | crear_llm(temperature=0.2)
        | StrOutputParser()
    )

    respuesta = chain.invoke(pregunta)
    return respuesta


print(consulta_rrhh("¿Cuál fue el resultado de la encuesta de clima laboral de este año?"))
print(consulta_rrhh("¿Cuál es el área donde los empleados tienen riesgo de abandonar la empresa?"))