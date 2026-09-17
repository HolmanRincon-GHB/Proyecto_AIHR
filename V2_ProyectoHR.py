"""
============================================================
  04_multiagente_rag_rrhh.py — Multiagente + RAG
  Seminario: Agentes de IA · Sesión 4

  Tema: Gestión de Recursos Humanos y Retención de Talento
  Empresa ficticia: TechnoDistrib S.A.S.

  Combina los dos patrones anteriores:
    - RAG (00_rag_documentos.py) para dar al INVESTIGADOR
      acceso a documentos internos reales de la empresa.
    - Sistema multiagente (03_multiagente.py) para coordinar
      Investigador → Analista → Redactor.

    ┌─────────────┐
    │ COORDINADOR │
    └──────┬──────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
  INVESTIGADOR  ANALISTA  REDACTOR
  (consulta      (analiza   (redacta
   RAG/Chroma)   hallazgos) respuesta)
    └──────┴──────────────┘
           │
           ▼
        RESPUESTA FINAL

  Preguntas de ejemplo que debería poder responder:
    • "¿Cuál fue el resultado de la encuesta de clima laboral?"
    • "¿Qué área tiene más riesgo de renuncias?"
    • "¿Cuál es el factor más importante para mejorar el desempeño?"
    • "¿Qué beneficios ofrece la empresa para retener talento?"

  Uso:
      python 04_multiagente_rag_rrhh.py

  Dependencias adicionales:
      pip install langchain-community chromadb sentence-transformers
============================================================
"""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from s4_llm_factory import crear_llm

load_dotenv()


# ═══════════════════════════════════════════════════════════
#  DOCUMENTOS SINTÉTICOS — RRHH / Retención de Talento
#  ✏️ MODIFICA AQUÍ: reemplaza con documentos reales de tu empresa
# ═══════════════════════════════════════════════════════════
DOCUMENTOS_RRHH = [
    {
        "titulo": "Encuesta de Clima Laboral 2024",
        "contenido": """
        ENCUESTA DE CLIMA LABORAL — TechnoDistrib S.A.S. · Julio 2024
        Participación: 264 de 287 empleados (92%)

        RESULTADO GLOBAL: 7.1 / 10 (vs 7.4 en 2023)

        RESULTADOS POR ÁREA
        - Tecnología: 8.2/10
        - Ventas: 6.4/10
        - Logística: 6.0/10
        - Operaciones: 6.1/10
        - Administración: 7.8/10

        DIMENSIONES EVALUADAS
        - Reconocimiento y feedback: 6.0/10 (la más baja del estudio)
        - Balance vida-trabajo: 6.3/10
        - Compensación percibida como justa: 6.5/10
        - Relación con jefe directo: 7.6/10
        - Orgullo de pertenencia: 7.9/10
        - Oportunidades de crecimiento: 6.2/10

        COMENTARIOS DESTACADOS (cualitativo)
        Los empleados de Logística y Operaciones mencionan turnos extendidos
        y falta de reconocimiento como los principales motivos de insatisfacción.
        En Ventas se reporta presión por metas sin acompañamiento suficiente
        de los líderes de equipo.

        CONCLUSIÓN DEL ESTUDIO
        El principal factor a intervenir es "reconocimiento y feedback",
        seguido de "oportunidades de crecimiento". Ambos están directamente
        correlacionados con la intención de permanencia reportada por los
        empleados encuestados.
        """,
    },
    {
        "titulo": "Análisis de Riesgo de Rotación por Área",
        "contenido": """
        ANÁLISIS DE RIESGO DE ROTACIÓN — TechnoDistrib S.A.S. · Q3 2024
        Metodología: modelo predictivo basado en historial, clima y mercado

        RIESGO DE RENUNCIA POR ÁREA (próximos 6 meses)
        - Logística: 34% de riesgo promedio (ALTO) — 12 personas en riesgo alto
        - Ventas: 27% de riesgo promedio (MEDIO-ALTO) — 9 personas en riesgo alto
        - Operaciones: 22% de riesgo promedio (MEDIO) — 6 personas en riesgo alto
        - Administración: 9% de riesgo promedio (BAJO)
        - Tecnología: 6% de riesgo promedio (BAJO)

        ÁREA CRÍTICA: LOGÍSTICA
        Es el área con mayor riesgo de rotación de toda la compañía.
        Factores identificados:
        1. Turnos extendidos sin compensación adicional clara
        2. Salario 15% por debajo del mercado para el rol de coordinador
        3. Rotación de supervisores directos (3 cambios en 12 meses)
        4. Baja percepción de reconocimiento (alineado con encuesta de clima)

        PERFIL DE MAYOR RIESGO
        Empleados con 1-3 años de antigüedad, sin ascensos recientes,
        y calificación de clima laboral individual menor a 6/10.

        RECOMENDACIÓN PRIORITARIA
        Intervenir primero en Logística con ajuste salarial y estabilización
        de supervisión directa, dado que concentra el mayor número absoluto
        de personas en riesgo alto y el mayor porcentaje relativo.
        """,
    },
    {
        "titulo": "Estudio de Factores de Desempeño",
        "contenido": """
        ESTUDIO DE FACTORES DE DESEMPEÑO — TechnoDistrib S.A.S. · 2024
        Basado en evaluaciones de desempeño y correlación estadística

        FACTORES ANALIZADOS Y SU CORRELACIÓN CON DESEMPEÑO
        1. Reconocimiento y feedback frecuente: correlación 0.71 (ALTA)
        2. Claridad de objetivos (OKRs): correlación 0.64 (ALTA)
        3. Acceso a capacitación relevante: correlación 0.52 (MEDIA)
        4. Compensación económica: correlación 0.38 (MEDIA-BAJA)
        5. Ambiente físico de trabajo: correlación 0.21 (BAJA)

        HALLAZGO PRINCIPAL
        El reconocimiento y feedback frecuente es el factor con mayor
        correlación con el desempeño individual, incluso por encima de la
        compensación económica. Los equipos con feedback semanal muestran
        un desempeño 23% superior al promedio de la compañía.

        DATO COMPLEMENTARIO
        Este hallazgo es consistente con la Encuesta de Clima Laboral 2024,
        donde "reconocimiento y feedback" fue la dimensión peor calificada
        (6.0/10), sugiriendo una oportunidad de mejora con alto impacto
        potencial en el desempeño general de la organización.

        RECOMENDACIÓN
        Implementar un programa formal de feedback continuo (no solo
        evaluación anual) como palanca prioritaria de mejora de desempeño,
        antes que incrementos salariales generalizados.
        """,
    },
    {
        "titulo": "Programa de Beneficios y Retención de Talento",
        "contenido": """
        PROGRAMA DE BENEFICIOS Y RETENCIÓN — TechnoDistrib S.A.S. · 2024

        BENEFICIOS VIGENTES
        - Medicina prepagada familiar (100% empleado, 50% núcleo familiar)
        - Día libre por cumpleaños
        - Auxilio de estudio: hasta $1.500.000 COP/semestre
        - Convenio con gimnasios en 5 ciudades
        - Bono de desempeño trimestral (hasta 15% del salario base)

        PROGRAMA DE RETENCIÓN — PLAN 2024-2025 (en implementación)
        1. Revisión salarial focalizada
           - Vendedores senior y coordinadores de Logística
           - Presupuesto aprobado: $180.000.000 COP
        2. Programa de reconocimiento "Impulso"
           - Feedback quincenal estructurado líder-colaborador
           - Reconocimiento público mensual por equipo
           - Piloto iniciado en Logística y Ventas (agosto 2024)
        3. Plan de carrera visible
           - Rutas de ascenso documentadas para 8 cargos operativos
           - Mentoría cruzada entre áreas
        4. Estabilización de liderazgo en Logística
           - Contratación de un coordinador senior con plan de permanencia
             mínima de 18 meses

        SEGUIMIENTO
        Los 23 empleados identificados como de alto riesgo de renuncia
        (ver Análisis de Riesgo de Rotación Q3 2024) están incluidos en un
        plan de seguimiento individual con revisión mensual por parte de
        Recursos Humanos.
        """,
    },
    {
        "titulo": "Indicadores Generales de RRHH — Primer Semestre 2024",
        "contenido": """
        INDICADORES GENERALES DE RRHH — TechnoDistrib S.A.S. · S1 2024

        - Total empleados: 287 (vs 271 en S1 2023, +5.9%)
        - Rotación voluntaria acumulada: 8.3% (objetivo anual: 7%)
        - Ausentismo promedio: 4.2 días por empleado en el semestre
        - Satisfacción laboral general: 7.1/10
        - Capacitaciones realizadas: 34 programas, 892 participantes
        - Tiempo promedio de vacante sin cubrir: 27 días
        - Costo estimado de rotación por empleado: $8.400.000 COP
          (reclutamiento, inducción y curva de aprendizaje)

        COMPARATIVO POR ÁREA — ROTACIÓN VOLUNTARIA
        - Logística: 14.2%
        - Ventas: 10.8%
        - Operaciones: 8.5%
        - Administración: 3.1%
        - Tecnología: 2.4%

        NOTA
        Estos indicadores generales sirven como línea base para medir el
        impacto del Programa de Retención de Talento implementado a partir
        del segundo semestre de 2024.
        """,
    },
]


# ═══════════════════════════════════════════════════════════
#  DOCUMENTOS REALES (PDF) — se suman a los sintéticos de arriba
#  ✏️ MODIFICA AQUÍ: agrega una entrada por cada PDF que quieras
#  indexar. "archivo" es el nombre dentro de CARPETA_PDFS_RRHH,
#  "titulo" es como aparecerá citado en "fuentes".
# ═══════════════════════════════════════════════════════════
CARPETA_PDFS_RRHH = os.getenv("CARPETA_PDFS_RRHH", "data/documentos")

PDFS_RRHH = [
    {"archivo": "02_informe_rrhh.pdf", "titulo": "Informe de RRHH"},
]


def cargar_pdfs_rrhh() -> list[Document]:
    """
    Carga cada PDF listado en PDFS_RRHH y lo convierte en un Document,
    exactamente en el mismo formato que los documentos sintéticos, para
    que se indexen juntos en el mismo vector store.
    """
    documentos = []
    carpeta = Path(CARPETA_PDFS_RRHH)
    if not carpeta.is_absolute():
        carpeta = Path(__file__).resolve().parent / carpeta

    for entrada in PDFS_RRHH:
        ruta = carpeta / entrada["archivo"]
        if not ruta.exists():
            print(f"  ⚠️  PDF no encontrado, se omite: {ruta}")
            continue

        paginas = PyPDFLoader(str(ruta)).load()
        texto_completo = "\n\n".join(p.page_content for p in paginas)
        documentos.append(Document(
            page_content=texto_completo,
            metadata={
                "titulo": entrada["titulo"],
                "empresa": "TechnoDistrib S.A.S.",
                "archivo": entrada["archivo"],
            },
        ))
        print(f"  📄 PDF cargado: {entrada['archivo']} → \"{entrada['titulo']}\" ({len(paginas)} páginas)")

    return documentos


# ═══════════════════════════════════════════════════════════
#  RAG — Construcción del vector store (una sola vez, global)
#  Combina los documentos sintéticos con los PDFs reales.
# ═══════════════════════════════════════════════════════════
def construir_vector_store():
    documentos = [
        Document(
            page_content=doc["contenido"],
            metadata={"titulo": doc["titulo"], "empresa": "TechnoDistrib S.A.S."},
        )
        for doc in DOCUMENTOS_RRHH
    ] + cargar_pdfs_rrhh()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)
    print(f"  📄 Documentos RRHH: {len(documentos)} · Chunks: {len(chunks)}")

    print("  🔢 Generando embeddings (30-60 segundos primera vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print(f"  ✅ Vector store: {vector_store._collection.count()} vectores")
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# ═══════════════════════════════════════════════════════════
#  STATE COMPARTIDO
# ═══════════════════════════════════════════════════════════
class EstadoRRHH(TypedDict):
    mensajes:          Annotated[list, add_messages]
    tarea_original:    str
    siguiente_agente:  str
    contexto_reunido:  str   # fragmentos recuperados del RAG
    fuentes:           str   # títulos de documentos citados
    analisis:          str
    respuesta_final:   str


# ═══════════════════════════════════════════════════════════
#  AGENTE COORDINADOR
# ═══════════════════════════════════════════════════════════
def agente_coordinador(estado: EstadoRRHH) -> dict:
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
#  AGENTE INVESTIGADOR (con RAG)
#  En vez de "inventar" conocimiento, consulta el vector store
#  con los documentos internos de RRHH.
# ═══════════════════════════════════════════════════════════
def crear_agente_investigador(retriever):
    def agente_investigador(estado: EstadoRRHH) -> dict:
        print("\n  🔍 [INVESTIGADOR] Consultando documentos internos de RRHH...")

        docs = retriever.invoke(estado["tarea_original"])

        contexto = ""
        titulos = []
        for i, doc in enumerate(docs, 1):
            titulo = doc.metadata.get("titulo", "Documento")
            titulos.append(titulo)
            contexto += f"\n--- Fragmento {i} — {titulo} ---\n{doc.page_content}\n"

        fuentes = ", ".join(sorted(set(titulos)))
        print(f"  🔍 [INVESTIGADOR] {len(docs)} fragmentos recuperados de: {fuentes}")

        return {
            "contexto_reunido": contexto,
            "fuentes": fuentes,
        }
    return agente_investigador


# ═══════════════════════════════════════════════════════════
#  MEMORIA — formatea el historial de conversación previo
#  para que el analista y el redactor puedan usarlo como
#  contexto (ej. preguntas de seguimiento).
#  ✏️ MODIFICA AQUÍ: cuántos mensajes previos recordar
# ═══════════════════════════════════════════════════════════
MEMORIA_MAX_MENSAJES = 8  # ~4 intercambios pregunta/respuesta


def formatear_historial(mensajes: list) -> str:
    """Convierte los mensajes previos (sin incluir la pregunta actual) a texto plano."""
    previos = mensajes[:-1][-MEMORIA_MAX_MENSAJES:]  # excluye la pregunta actual
    if not previos:
        return "(sin conversación previa)"

    lineas = []
    for m in previos:
        if isinstance(m, HumanMessage):
            lineas.append(f"Usuario: {m.content}")
        elif isinstance(m, AIMessage):
            lineas.append(f"Asistente: {m.content}")
    return "\n".join(lineas) if lineas else "(sin conversación previa)"


# ═══════════════════════════════════════════════════════════
#  AGENTE ANALISTA
#  Analiza estrictamente el contexto recuperado por RAG.
# ═══════════════════════════════════════════════════════════
def agente_analista(estado: EstadoRRHH) -> dict:
    print("\n  📊 [ANALISTA] Analizando información de RRHH...")

    llm = crear_llm(temperature=0.3)

    system = """Eres un analista de Recursos Humanos experto en retención de
talento. Tu trabajo es analizar ÚNICAMENTE la información de los documentos
internos proporcionados (no inventes cifras ni datos que no aparezcan ahí):
1. Identifica los datos y hallazgos relevantes para la pregunta
2. Señala relaciones entre distintos documentos si existen
3. Si la información no está en el contexto, dilo explícitamente
4. Prioriza los hallazgos más accionables
Responde en español con estructura clara."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Pregunta del usuario: {estado['tarea_original']}

Conversación previa (úsala solo si la pregunta actual hace referencia a ella,
por ejemplo "y en esa área..." o "cuál fue mi pregunta anterior"):
{formatear_historial(estado.get('mensajes', []))}

Documentos internos recuperados:
{estado['contexto_reunido']}

Analiza esta información y extrae las conclusiones relevantes para responder
la pregunta."""),
    ])

    print(f"  📊 [ANALISTA] Análisis completado ({len(respuesta.content)} chars)")
    return {
        "analisis": respuesta.content,
    }


# ═══════════════════════════════════════════════════════════
#  AGENTE REDACTOR
#  Genera la respuesta final citando las fuentes usadas.
# ═══════════════════════════════════════════════════════════
def agente_redactor(estado: EstadoRRHH) -> dict:
    print("\n  ✍️  [REDACTOR] Elaborando respuesta final...")

    llm = crear_llm(temperature=0.5)

    system = """Eres un asistente de RRHH que comunica hallazgos a líderes de
la empresa. Tu trabajo es:
1. Responder de forma directa y clara a la pregunta del usuario
2. Basarte únicamente en el análisis proporcionado
3. Citar los documentos fuente relevantes al final
4. Usar formato Markdown si ayuda a la claridad (listas, negritas)
Responde en español."""

    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Pregunta del usuario: {estado['tarea_original']}

Conversación previa (úsala solo si la pregunta actual hace referencia a ella):
{formatear_historial(estado.get('mensajes', []))}

Análisis del equipo de RRHH:
{estado['analisis']}

Documentos fuente disponibles: {estado.get('fuentes', 'N/A')}

Redacta la respuesta final para el usuario, citando las fuentes al final."""),
    ])

    print(f"  ✍️  [REDACTOR] Respuesta final lista")
    return {
        "respuesta_final": respuesta.content,
    }


# ═══════════════════════════════════════════════════════════
#  ROUTER Y CONSTRUCCIÓN DEL GRAFO
# ═══════════════════════════════════════════════════════════
def router_coordinador(estado: EstadoRRHH) -> Literal[
    "investigador", "analista", "redactor", "__end__"
]:
    sig = estado.get("siguiente_agente", "investigador")
    if sig == "fin":
        return "__end__"
    return sig


def construir_grafo(retriever):
    grafo = StateGraph(EstadoRRHH)

    grafo.add_node("coordinador",  agente_coordinador)
    grafo.add_node("investigador", crear_agente_investigador(retriever))
    grafo.add_node("analista",     agente_analista)
    grafo.add_node("redactor",     agente_redactor)

    grafo.add_edge(START, "coordinador")
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
    grafo.add_edge("investigador", "coordinador")
    grafo.add_edge("analista",     "coordinador")
    grafo.add_edge("redactor",     "coordinador")

    return grafo.compile()


# ═══════════════════════════════════════════════════════════
#  INTERFAZ INTERACTIVA
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  Sistema Multiagente + RAG — RRHH y Retención de Talento")
    print("  TechnoDistrib S.A.S.")
    print("  Coordinador → Investigador (RAG) → Analista → Redactor")
    print("═" * 60)

    print("\n  Construyendo base de conocimiento (RAG)...")
    retriever = construir_vector_store()
    app = construir_grafo(retriever)

    print("\n  ✅ Sistema listo\n")
    print("  Documentos cargados:")
    for doc in DOCUMENTOS_RRHH:
        print(f"    📄 {doc['titulo']}")
    for pdf in PDFS_RRHH:
        print(f"    📄 {pdf['titulo']} (PDF)")

    print("\n  Preguntas de ejemplo:")
    print("    • '¿Cuál fue el resultado de la encuesta de clima laboral?'")
    print("    • '¿Qué área tiene más riesgo de renuncia?'")
    print("    • '¿Cuál es el factor más importante para mejorar el desempeño?'")
    print("    • '¿Qué beneficios ofrece la empresa para retener talento?'")
    print("    • '¿Cuánto cuesta la rotación de personal por empleado?'")
    print("  Para salir: salir\n")

    historial_mensajes = []  # ✏️ persiste entre preguntas mientras dure esta sesión de consola

    while True:
        try:
            tarea = input("Pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not tarea or tarea.lower() == "salir":
            break

        print(f"\n  Procesando con 4 agentes...\n")

        estado_inicial = {
            "mensajes":         historial_mensajes + [HumanMessage(content=tarea)],
            "tarea_original":   tarea,
            "siguiente_agente": "",
            "contexto_reunido": "",
            "fuentes":          "",
            "analisis":         "",
            "respuesta_final":  "",
        }

        try:
            resultado = app.invoke(estado_inicial)
            print("\n" + "─" * 60)
            print("  RESPUESTA FINAL:")
            print("─" * 60)
            print(resultado["respuesta_final"])
            print("─" * 60 + "\n")

            # Guarda el intercambio para que las próximas preguntas lo recuerden
            historial_mensajes = resultado["mensajes"] + [AIMessage(content=resultado["respuesta_final"])]
            historial_mensajes = historial_mensajes[-MEMORIA_MAX_MENSAJES:]
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")

    print("\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()