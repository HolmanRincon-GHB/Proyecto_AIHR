"""
============================================================
  00_rag_documentos.py — RAG con documentos sintéticos
  Seminario: Agentes de IA · Sesión 4

  RAG = Retrieval Augmented Generation
  El agente consulta documentos reales antes de responder
  en lugar de inventar información.

  Este archivo genera 5 documentos sintéticos de TechnoDistrib
  y permite hacerle preguntas al agente sobre su contenido.

  Flujo:
    Documentos → Chunks → Embeddings → Vector Store
    Pregunta → Recuperar chunks relevantes → LLM responde

  Uso:
      python 00_rag_documentos.py

  Dependencias adicionales:
      pip install langchain-community chromadb sentence-transformers
============================================================
"""

import os, sys, textwrap
from dotenv import load_dotenv

# En algunas consolas de Windows la salida no es UTF-8 por defecto,
# lo que rompe los emojis usados en los mensajes. Forzamos UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from s4_llm_factory import crear_llm

# ══════════════════════════════════════════════════════════════
#  DOCUMENTOS SINTÉTICOS — TechnoDistrib S.A.S.
#  ✏️ MODIFICA AQUÍ: reemplaza con tus propios documentos
#  Para cargar PDFs reales usa:
#  from langchain_community.document_loaders import PyPDFLoader
#  loader = PyPDFLoader("mi_documento.pdf")
#  docs = loader.load()
# ══════════════════════════════════════════════════════════════

DOCUMENTOS_SINTETICOS = [
    {
        "titulo": "Manual de Política Comercial 2024",
        "contenido": """
        POLÍTICA COMERCIAL — TechnoDistrib S.A.S.
        Vigencia: Enero 2024 — Diciembre 2024

        1. DESCUENTOS POR VOLUMEN
        Los descuentos se aplican según el volumen mensual acumulado:
        - Compras entre $5.000.000 y $20.000.000 COP: descuento del 5%
        - Compras entre $20.000.001 y $50.000.000 COP: descuento del 10%
        - Compras superiores a $50.000.000 COP: descuento del 15%
        - Clientes Platinum con más de 2 años: descuento adicional del 3%

        2. PLAZOS DE PAGO
        - Contado: descuento adicional del 2%
        - 30 días: sin recargo
        - 60 días: recargo del 1.5% mensual
        - 90 días: requiere aprobación de crédito y garantía

        3. POLÍTICA DE DEVOLUCIONES
        - Productos defectuosos: devolución hasta 30 días con factura
        - Cambio de referencia: hasta 15 días, sin uso, empaque original
        - Equipos abiertos: no aplica devolución salvo defecto de fábrica
        - El cliente asume el costo de transporte en devoluciones por cambio

        4. GARANTÍAS
        - Computadores y portátiles: 12 meses contra defectos de fábrica
        - Dispositivos móviles: 6 meses
        - Periféricos y accesorios: 3 meses
        - Redes y comunicaciones: 12 meses
        """,
    },
    {
        "titulo": "Informe de Gestión RRHH — Primer Semestre 2024",
        "contenido": """
        INFORME DE GESTIÓN — RECURSOS HUMANOS
        TechnoDistrib S.A.S. · Primer Semestre 2024

        RESUMEN EJECUTIVO
        Al cierre del primer semestre, TechnoDistrib cuenta con 287 empleados
        distribuidos en 5 regiones del país. La tasa de rotación fue del 8.3%,
        ligeramente por encima del objetivo del 7% establecido para 2024.

        INDICADORES CLAVE
        - Total empleados: 287 (vs 271 en S1 2023, +5.9%)
        - Rotación voluntaria: 8.3% (objetivo: 7%)
        - Ausentismo promedio: 4.2 días por empleado
        - Satisfacción laboral (encuesta): 7.1/10
        - Capacitaciones realizadas: 34 programas, 892 participantes

        HALLAZGOS PRINCIPALES
        1. El departamento de Operaciones presenta el clima laboral más bajo
           con 6.1/10, asociado a carga de trabajo y turnos extendidos.
        2. Los vendedores senior muestran una brecha salarial promedio del
           18% respecto al mercado, generando riesgo de fuga de talento.
        3. Se identificaron 23 empleados con alto riesgo de renuncia en el
           siguiente trimestre, concentrados en Ventas y Logística.

        PLAN DE ACCIÓN S2 2024
        - Revisión salarial para vendedores senior: presupuesto $180M COP
        - Programa de bienestar para Operaciones: inicio agosto 2024
        - Plan de retención para los 23 empleados en riesgo identificados
        """,
    },
    {
        "titulo": "Protocolo de Atención al Cliente",
        "contenido": """
        PROTOCOLO DE ATENCIÓN AL CLIENTE
        TechnoDistrib S.A.S. — Versión 3.2 — Marzo 2024

        NIVELES DE ATENCIÓN

        NIVEL 1 — Soporte Básico (0-15 minutos)
        Atendido por: Asesores de servicio al cliente
        Casos: consultas de precio, disponibilidad, estado de pedido,
               facturación, información general de productos

        NIVEL 2 — Soporte Técnico (15-60 minutos)
        Atendido por: Técnicos certificados
        Casos: configuración de equipos, diagnóstico de fallas,
               instalación de software, compatibilidad de productos

        NIVEL 3 — Escalamiento Gerencial (mayor a 60 minutos)
        Atendido por: Gerente de cuenta o Director Comercial
        Casos: reclamaciones mayores a $10.000.000 COP,
               clientes Platinum insatisfechos, demandas o quejas formales

        TIEMPOS DE RESPUESTA COMPROMETIDOS
        - Chat en línea: respuesta inicial en menos de 2 minutos
        - Correo electrónico: respuesta en menos de 4 horas hábiles
        - Teléfono: atención inmediata, espera máxima 3 minutos
        - Visita técnica: programación en menos de 48 horas

        POLÍTICA DE COMPENSACIONES POR INCUMPLIMIENTO
        Si el tiempo de entrega supera lo prometido en más de 5 días:
        - Descuento del 5% en el siguiente pedido
        - Si supera 10 días: descuento del 10% más envío gratis
        - Si supera 15 días: devolución parcial del 20% del valor
        """,
    },
    {
        "titulo": "Plan Estratégico 2024-2026",
        "contenido": """
        PLAN ESTRATÉGICO — TechnoDistrib S.A.S. · Período 2024-2026

        VISIÓN
        Ser la distribuidora de tecnología líder en Colombia para 2026,
        con presencia en las 5 principales ciudades y canal digital
        representando el 35% de las ventas totales.

        PILARES ESTRATÉGICOS

        PILAR 1: CRECIMIENTO DIGITAL
        Meta: aumentar el canal e-commerce del 8% actual al 35% para 2026
        - Lanzamiento plataforma B2B en línea: Q3 2024
        - Integración con marketplaces Mercado Libre y Amazon: Q4 2024
        - App móvil para clientes corporativos: Q1 2025

        PILAR 2: EXPANSIÓN GEOGRÁFICA
        Meta: abrir operaciones en Pereira y Cúcuta para 2025
        Inversión estimada: $2.800.000.000 COP por ciudad

        PILAR 3: DIVERSIFICACIÓN DE PORTAFOLIO
        - Incorporar línea de ciberseguridad: Q2 2024
        - Servicios administrados MSP: Q3 2024
        - Soluciones cloud AWS, Azure, GCP: Q1 2025

        METAS FINANCIERAS
        - 2024: crecimiento ingresos 18% vs 2023
        - 2025: crecimiento 22%, margen EBITDA 12%
        - 2026: ingresos $85.000.000.000 COP
        """,
    },
    {
        "titulo": "Catálogo de Productos — Línea Empresarial 2024",
        "contenido": """
        CATÁLOGO LÍNEA EMPRESARIAL — TechnoDistrib S.A.S. · Julio 2024

        COMPUTADORES PORTÁTILES EMPRESARIALES

        Dell Latitude 5540
        - Procesador: Intel Core i7-1365U
        - RAM: 16 GB DDR5 ampliable a 64 GB
        - Almacenamiento: SSD 512 GB NVMe
        - Precio lista: $4.850.000 COP
        - Garantía: 3 años ProSupport on-site

        HP EliteBook 840 G10
        - Procesador: Intel Core i5-1345U
        - RAM: 8 GB DDR5 ampliable a 32 GB
        - Almacenamiento: SSD 256 GB
        - Precio lista: $3.920.000 COP
        - Garantía: 3 años HP Care Pack

        INFRAESTRUCTURA DE RED

        Switch Cisco Catalyst 1300-24T
        - 24 puertos Gigabit Ethernet
        - 4 puertos SFP uplink 1G
        - Precio lista: $1.850.000 COP

        Access Point Ubiquiti UniFi U6 Pro
        - WiFi 6, cobertura hasta 300 dispositivos
        - Precio lista: $780.000 COP

        CONDICIONES COMERCIALES LÍNEA EMPRESARIAL
        - Pedido mínimo: 3 unidades por referencia
        - Tiempo de entrega: 2 a 5 días hábiles
        - Financiación disponible: 12 a 36 cuotas
        """,
    },
]


# ══════════════════════════════════════════════════════════════
#  PASO 1 — Crear documentos LangChain
# ══════════════════════════════════════════════════════════════

def crear_documentos() -> list:
    documentos = []
    for doc in DOCUMENTOS_SINTETICOS:
        documentos.append(Document(
            page_content=doc["contenido"],
            metadata={
                "titulo":  doc["titulo"],
                "empresa": "TechnoDistrib S.A.S.",
            }
        ))
    print(f"  📄 Documentos creados: {len(documentos)}")
    return documentos


# ══════════════════════════════════════════════════════════════
#  PASO 2 — Dividir en chunks
#  ✏️ MODIFICA AQUÍ: ajusta chunk_size y chunk_overlap
# ══════════════════════════════════════════════════════════════

def dividir_en_chunks(documentos: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documentos)
    total_chars = sum(len(c.page_content) for c in chunks)
    promedio = total_chars // len(chunks) if chunks else 0
    print(f"  ✂️  Chunks generados: {len(chunks)} (~{promedio} chars/chunk)")
    return chunks


# ══════════════════════════════════════════════════════════════
#  PASO 3 — Embeddings y Vector Store
#  ✏️ MODIFICA AQUÍ: cambia el modelo de embeddings
# ══════════════════════════════════════════════════════════════

def crear_vector_store(chunks: list):
    print("  🔢 Generando embeddings (30-60 segundos primera vez)...")
    embeddings = HuggingFaceEmbeddings(
        # ✏️ Modelo multilingüe — bueno para español
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    print(f"  ✅ Vector store: {vector_store._collection.count()} vectores")
    return vector_store


# ══════════════════════════════════════════════════════════════
#  PASO 4 — Chain RAG con LCEL
# ══════════════════════════════════════════════════════════════

def construir_chain_rag(vector_store):
    """
    Chain RAG con LCEL (operador |):
    Pregunta → Retriever → Contexto → LLM → Respuesta
    """
    llm = crear_llm(temperature=0.2)

    # ✏️ k = número de chunks a recuperar (más k = más contexto, más tokens)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # ✏️ MODIFICA AQUÍ: ajusta las instrucciones del sistema
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente experto de TechnoDistrib S.A.S.
Responde ÚNICAMENTE basándote en el contexto de los documentos internos proporcionados.
Si la información no está en el contexto, dilo claramente.
Cita el documento fuente cuando sea posible.
Responde en español de forma clara y profesional."""),
        ("human", """Contexto de los documentos internos:
{contexto}

Pregunta: {pregunta}

Responde basándote exclusivamente en el contexto anterior."""),
    ])

    def formatear_contexto(docs) -> str:
        resultado = ""
        for i, doc in enumerate(docs, 1):
            titulo = doc.metadata.get("titulo", "Documento")
            resultado += f"\n--- Fragmento {i} — {titulo} ---\n"
            resultado += doc.page_content + "\n"
        return resultado

    # ── Chain RAG con LCEL ────────────────────────────────────
    chain = (
        {
            "contexto": retriever | formatear_contexto,
            "pregunta": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ══════════════════════════════════════════════════════════════
#  INTERFAZ INTERACTIVA
# ══════════════════════════════════════════════════════════════

def mostrar_fuentes(retriever, pregunta: str):
    docs = retriever.invoke(pregunta)
    print(f"\n  📎 Fragmentos recuperados:")
    for i, doc in enumerate(docs, 1):
        titulo = doc.metadata.get("titulo", "?")
        print(f"  [{i}] {titulo[:55]} — {len(doc.page_content)} chars")


def main():
    print("\n" + "═"*55)
    print("  RAG — Consulta de Documentos Internos")
    print("  TechnoDistrib S.A.S.")
    print("═"*55)

    print("\n  Construyendo pipeline RAG...")
    documentos    = crear_documentos()
    chunks        = dividir_en_chunks(documentos)
    vector_store  = crear_vector_store(chunks)
    chain, retriever = construir_chain_rag(vector_store)

    print("\n  ✅ Sistema RAG listo\n")
    print("  Documentos cargados:")
    for doc in DOCUMENTOS_SINTETICOS:
        print(f"    📄 {doc['titulo']}")

    print("\n  Preguntas de ejemplo:")
    print("    • '¿Cuál es el descuento para compras de $30 millones?'")
    print("    • '¿Qué garantía tiene un portátil Dell?'")
    print("    • '¿Cuántos empleados tiene la empresa?'")
    print("    • '¿Cuál es la meta de ventas para 2026?'")
    print("    • '¿Cuánto tiempo tengo para devolver un producto?'")
    print("\n  Comandos: /fuentes | salir")
    print("─"*55)

    while True:
        try:
            pregunta = input("\n❓ Pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            break
        if pregunta.lower() == "/fuentes":
            demo = "¿Qué descuentos ofrece la empresa?"
            mostrar_fuentes(retriever, demo)
            continue

        mostrar_fuentes(retriever, pregunta)

        try:
            print(f"\n  🤖 Consultando documentos...\n")
            respuesta = chain.invoke(pregunta)
            print("  Respuesta:\n")
            for linea in respuesta.split("\n"):
                if linea.strip():
                    print(textwrap.fill(
                        linea, width=68,
                        initial_indent="  ",
                        subsequent_indent="  "
                    ))
                else:
                    print()
        except Exception as e:
            print(f"\n  ❌ Error: {e}")

    print("\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()
