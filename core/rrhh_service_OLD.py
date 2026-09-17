"""
core/rrhh_service.py — Servicio multiagente + RAG sobre RRHH
Seminario: Agentes de IA · Sesión 6 (Despliegue)

Endpoint dedicado /rrhh — independiente de /chat y /agent.
Reutiliza el patrón de agent_core.py: se construye UNA sola vez
al arrancar el servidor (vector store + grafo compilado) y esa
misma instancia atiende todas las peticiones.

Tema: Gestión de RRHH y Retención de Talento — TechnoDistrib S.A.S.

Flujo: Coordinador → Investigador (RAG/Chroma) → Analista → Redactor
"""

import os
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from s6_llm_factory import crear_llm

load_dotenv()


# ═══════════════════════════════════════════════════════════
#  DOCUMENTOS SINTÉTICOS — RRHH / Retención de Talento
#  ✏️ MODIFICA AQUÍ: reemplaza con documentos reales de tu empresa
#  o cárgalos desde DOCUMENTOS_DIR (.env) con un loader real.
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
#  STATE COMPARTIDO DEL GRAFO MULTIAGENTE
# ═══════════════════════════════════════════════════════════
class EstadoRRHH(TypedDict):
    mensajes:          Annotated[list, add_messages]
    tarea_original:    str
    siguiente_agente:  str
    contexto_reunido:  str
    fuentes:           list
    analisis:          str
    respuesta_final:   str


# ═══════════════════════════════════════════════════════════
#  AGENTES
# ═══════════════════════════════════════════════════════════
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


def crear_agente_investigador(retriever):
    """Closure para inyectar el retriever de Chroma ya construido."""
    def agente_investigador(estado: EstadoRRHH) -> dict:
        docs = retriever.invoke(estado["tarea_original"])

        contexto = ""
        titulos = []
        for i, doc in enumerate(docs, 1):
            titulo = doc.metadata.get("titulo", "Documento")
            titulos.append(titulo)
            contexto += f"\n--- Fragmento {i} — {titulo} ---\n{doc.page_content}\n"

        fuentes = sorted(set(titulos))
        return {
            "contexto_reunido": contexto,
            "fuentes": fuentes,
            "mensajes": [AIMessage(content=f"[Investigador]: consultó {', '.join(fuentes)}")],
        }
    return agente_investigador


def agente_analista(estado: EstadoRRHH) -> dict:
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

Documentos internos recuperados:
{estado['contexto_reunido']}

Analiza esta información y extrae las conclusiones relevantes para responder
la pregunta."""),
    ])

    return {
        "analisis": respuesta.content,
        "mensajes": [AIMessage(content=f"[Analista]: {respuesta.content[:200]}...")],
    }


def agente_redactor(estado: EstadoRRHH) -> dict:
    llm = crear_llm(temperature=0.5)

    system = """Eres un asistente de RRHH que comunica hallazgos a líderes de
la empresa. Tu trabajo es:
1. Responder de forma directa y clara a la pregunta del usuario
2. Basarte únicamente en el análisis proporcionado
3. Citar los documentos fuente relevantes al final
4. Usar formato Markdown si ayuda a la claridad (listas, negritas)
Responde en español."""

    fuentes = estado.get("fuentes", [])
    respuesta = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"""Pregunta del usuario: {estado['tarea_original']}

Análisis del equipo de RRHH:
{estado['analisis']}

Documentos fuente disponibles: {', '.join(fuentes) if fuentes else 'N/A'}

Redacta la respuesta final para el usuario, citando las fuentes al final."""),
    ])

    return {
        "respuesta_final": respuesta.content,
        "mensajes": [AIMessage(content=respuesta.content)],
    }


def router_coordinador(estado: EstadoRRHH) -> Literal[
    "investigador", "analista", "redactor", "__end__"
]:
    sig = estado.get("siguiente_agente", "investigador")
    if sig == "fin":
        return "__end__"
    return sig


# ═══════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL VECTOR STORE Y DEL GRAFO
# ═══════════════════════════════════════════════════════════
def _construir_retriever():
    documentos = [
        Document(
            page_content=doc["contenido"],
            metadata={"titulo": doc["titulo"], "empresa": "TechnoDistrib S.A.S."},
        )
        for doc in DOCUMENTOS_RRHH
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("TAMANO_FRAGMENTO", "800")),
        chunk_overlap=int(os.getenv("SUPERPOSICION_FRAGMENTO", "100")),
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    k = int(os.getenv("K_FRAGMENTOS", "4"))
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever, len(chunks)


def _construir_grafo_rrhh(retriever):
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
#  SERVICIO — patrón singleton, igual que AgentService
#  Se construye UNA vez (vector store + grafo) al arrancar.
# ═══════════════════════════════════════════════════════════
class RRHHService:
    def __init__(self):
        print("🔧 RRHHService: construyendo base de conocimiento (RAG)...")
        self._retriever, self._total_chunks = _construir_retriever()
        self._grafo = _construir_grafo_rrhh(self._retriever)
        print(f"✅ RRHHService listo — {len(DOCUMENTOS_RRHH)} documentos, {self._total_chunks} fragmentos")

    def _estado_inicial(self, pregunta: str) -> dict:
        return {
            "mensajes":         [],
            "tarea_original":   pregunta,
            "siguiente_agente": "",
            "contexto_reunido": "",
            "fuentes":          [],
            "analisis":         "",
            "respuesta_final":  "",
        }

    def _lineas_para_nodo(self, nodo: str, estado: dict) -> list:
        """Traduce cada paso del grafo a las mismas líneas que ya se ven en la CLI."""
        if nodo == "coordinador":
            siguiente = estado.get("siguiente_agente", "").upper()
            return [f"🎯 [COORDINADOR] → {siguiente}"]
        if nodo == "investigador":
            return [
                "🔍 [INVESTIGADOR] Recopilando información...",
                f"📚 RAG listo: {len(DOCUMENTOS_RRHH)} documento(s), {self._total_chunks} fragmentos",
            ]
        if nodo == "analista":
            return ["📊 [ANALISTA] Analizando información..."]
        if nodo == "redactor":
            return ["✍️  [REDACTOR] Redactando respuesta final..."]
        return [f"[{nodo}]"]

    def responder(self, pregunta: str) -> dict:
        """Ejecuta el grafo multiagente completo para una pregunta de RRHH (sin progreso)."""
        try:
            resultado = self._grafo.invoke(self._estado_inicial(pregunta))
        except Exception as e:
            raise RuntimeError(f"Error en el multiagente RRHH: {e}")

        return {
            "respuesta": resultado["respuesta_final"],
            "fuentes":   resultado.get("fuentes", []),
        }

    def responder_stream(self, pregunta: str):
        """
        Generador que emite un evento por cada paso del grafo mientras corre,
        y termina con un evento 'final' con la respuesta completa. Pensado
        para transmitirse como Server-Sent Events desde el endpoint.
        """
        estado_actual = self._estado_inicial(pregunta)

        try:
            for actualizacion in self._grafo.stream(estado_actual, stream_mode="updates"):
                for nodo, salida in actualizacion.items():
                    estado_actual.update(salida)
                    for linea in self._lineas_para_nodo(nodo, estado_actual):
                        yield {"tipo": "paso", "linea": linea}
        except Exception as e:
            yield {"tipo": "error", "mensaje": f"Error en el multiagente RRHH: {e}"}
            return

        yield {
            "tipo": "final",
            "respuesta": estado_actual.get("respuesta_final", ""),
            "fuentes": estado_actual.get("fuentes", []),
        }

    @property
    def documentos_disponibles(self) -> list:
        return [doc["titulo"] for doc in DOCUMENTOS_RRHH]


# Instancia global — se crea una sola vez al arrancar el servidor.
# ✏️ No modifiques esta línea (patrón igual al de agente_service)
rrhh_service = RRHHService()
