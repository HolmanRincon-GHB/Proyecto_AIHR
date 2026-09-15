"""
============================================================
  02_grafo_condicional.py — Edges condicionales
  Seminario: Agentes de IA · Sesión 4

  Los edges condicionales son la clave del poder de LangGraph.
  En lugar de seguir siempre el mismo camino, el grafo
  DECIDE qué nodo ejecutar a continuación basándose
  en el contenido del State.

  Este grafo clasifica la pregunta del usuario y la
  enruta al especialista correcto:

      START → clasificar → [tecnico | general | no_sé]
                               ↓          ↓        ↓
                           respuesta  respuesta  disculpa
                               ↓          ↓        ↓
                              END        END      END

  ✏️ El router (clasificador) es lo que debes personalizar
     para tu caso de uso.

  Uso:
      python 02_grafo_condicional.py
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from s4_llm_factory import crear_llm

load_dotenv()


class Estado(TypedDict):
    pregunta:   str
    categoria:  str   # "tecnico" | "general" | "desconocido"
    respuesta:  str


# ═══════════════════════════════════════════════════════════
#  NODO CLASIFICADOR — decide el camino
#  ✏️ MODIFICA AQUÍ: cambia las categorías y la lógica
#     de clasificación para tu dominio
# ═══════════════════════════════════════════════════════════

def nodo_clasificar(estado: Estado) -> dict:
    """
    Clasifica la pregunta en una categoría.
    Esta clasificación determina qué nodo se ejecuta después.
    """
    llm = crear_llm(temperature=0.0)  # temperatura 0 para clasificación determinista

    # ✏️ MODIFICA AQUÍ: ajusta las categorías y sus descripciones
    prompt_clasificacion = """Clasifica la siguiente pregunta en UNA de estas categorías:
- tecnico: preguntas sobre programación, software, hardware, tecnología, IA
- general: preguntas de conocimiento general, cultura, historia, ciencias
- desconocido: preguntas que no puedes clasificar con certeza

Responde SOLO con la categoría en minúsculas, sin explicación.
Ejemplos:
  "¿Cómo instalo Python?" → tecnico
  "¿Cuál es la capital de Francia?" → general
  "asdfgh" → desconocido"""

    respuesta = llm.invoke([
        SystemMessage(content=prompt_clasificacion),
        HumanMessage(content=estado["pregunta"]),
    ])

    categoria = respuesta.content.strip().lower()
    # Normaliza por si el LLM devuelve algo inesperado
    if categoria not in ["tecnico", "general", "desconocido"]:
        categoria = "general"

    print(f"  [Clasificador] Categoría detectada: '{categoria}'")
    return {"categoria": categoria}


# ═══════════════════════════════════════════════════════════
#  NODOS ESPECIALIZADOS — uno por categoría
#  ✏️ MODIFICA AQUÍ: personaliza el system prompt de cada
#     especialista para tu dominio
# ═══════════════════════════════════════════════════════════

def nodo_experto_tecnico(estado: Estado) -> dict:
    """Responde preguntas técnicas con profundidad."""
    print("  [Experto Técnico] Generando respuesta técnica...")
    llm = crear_llm(temperature=0.3)
    # ✏️ Cambia este system prompt
    resp = llm.invoke([
        SystemMessage(content="Eres un ingeniero senior. Responde de forma técnica y precisa en español. Incluye ejemplos de código si es relevante."),
        HumanMessage(content=estado["pregunta"]),
    ])
    return {"respuesta": f"🔧 [Experto Técnico]\n{resp.content}"}


def nodo_experto_general(estado: Estado) -> dict:
    """Responde preguntas de conocimiento general."""
    print("  [Experto General] Generando respuesta general...")
    llm = crear_llm(temperature=0.7)
    # ✏️ Cambia este system prompt
    resp = llm.invoke([
        SystemMessage(content="Eres un asistente educativo. Explica de forma clara y accesible en español."),
        HumanMessage(content=estado["pregunta"]),
    ])
    return {"respuesta": f"📚 [Experto General]\n{resp.content}"}


def nodo_no_sabe(estado: Estado) -> dict:
    """Responde cuando no puede clasificar la pregunta."""
    print("  [No Sé] Generando disculpa...")
    return {"respuesta": "🤷 No pude clasificar tu pregunta. ¿Puedes reformularla?"}


# ═══════════════════════════════════════════════════════════
#  FUNCIÓN ROUTER — el "cerebro" de los edges condicionales
#  Debe devolver el nombre del nodo al que ir
#
#  ✏️ MODIFICA AQUÍ: agrega más rutas según tus categorías
# ═══════════════════════════════════════════════════════════

def router(estado: Estado) -> Literal["experto_tecnico", "experto_general", "no_sabe"]:
    """
    Lee el State y devuelve el nombre del nodo siguiente.
    LangGraph usa este valor para decidir qué edge seguir.
    """
    cat = estado.get("categoria", "desconocido")
    if cat == "tecnico":
        return "experto_tecnico"
    elif cat == "general":
        return "experto_general"
    else:
        return "no_sabe"


def construir_grafo():
    grafo = StateGraph(Estado)

    # Nodos
    grafo.add_node("clasificar",      nodo_clasificar)
    grafo.add_node("experto_tecnico", nodo_experto_tecnico)
    grafo.add_node("experto_general", nodo_experto_general)
    grafo.add_node("no_sabe",         nodo_no_sabe)

    # Edge fijo: siempre empieza clasificando
    grafo.add_edge(START, "clasificar")

    # ✏️ Edge condicional: después de clasificar, el router decide
    grafo.add_conditional_edges(
        "clasificar",   # nodo de origen
        router,         # función que decide el destino
        {               # mapa: valor retornado → nodo destino
            "experto_tecnico": "experto_tecnico",
            "experto_general": "experto_general",
            "no_sabe":         "no_sabe",
        }
    )

    # Todos los especialistas terminan en END
    grafo.add_edge("experto_tecnico", END)
    grafo.add_edge("experto_general", END)
    grafo.add_edge("no_sabe",         END)

    return grafo.compile()


def main():
    app = construir_grafo()

    print("\n" + "═" * 52)
    print("  Grafo Condicional — Router Inteligente")
    print("  Clasifica tu pregunta y la envía al experto")
    print("═" * 52)
    print("  Prueba:")
    print("    • '¿Cómo funciona un árbol binario?'")
    print("    • '¿Quién pintó La Mona Lisa?'")
    print("    • 'xkjdhfksjd'")
    print("  Para salir: salir\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not entrada or entrada.lower() == "salir":
            break

        print()
        try:
            resultado = app.invoke({"pregunta": entrada, "categoria": "", "respuesta": ""})
            print(f"\n{resultado['respuesta']}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")

    print("\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()
