"""
============================================================
  04_visualizar_grafo.py — Exporta imagen del grafo
  Seminario: Agentes de IA · Sesión 4

  Genera una imagen PNG del grafo de ejecución para
  documentación y presentaciones.

  Requiere: pip install grandalf
  (o simplemente muestra el grafo en texto si no está)

  Uso:
      python 04_visualizar_grafo.py
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ── Reimporta los grafos de los archivos anteriores ──────
# Para no duplicar código, reconstruimos versiones simplificadas

class EstadoSimple(TypedDict):
    mensajes: Annotated[list, add_messages]
    siguiente: str

class EstadoCondicional(TypedDict):
    pregunta:  str
    categoria: str
    respuesta: str

class EstadoMulti(TypedDict):
    mensajes:         Annotated[list, add_messages]
    tarea_original:   str
    siguiente_agente: str
    contexto_reunido: str
    analisis:         str
    respuesta_final:  str


def nodo_dummy(estado):
    return {}

def router_dummy(estado) -> Literal["a", "b"]:
    return "a"

def router_multi(estado) -> Literal["investigador","analista","redactor","__end__"]:
    return "investigador"


def grafo_minimo():
    g = StateGraph(EstadoSimple)
    g.add_node("preparar",  nodo_dummy)
    g.add_node("responder", nodo_dummy)
    g.add_edge(START, "preparar")
    g.add_edge("preparar", "responder")
    g.add_edge("responder", END)
    return g.compile()


def grafo_condicional():
    g = StateGraph(EstadoCondicional)
    g.add_node("clasificar",      nodo_dummy)
    g.add_node("experto_tecnico", nodo_dummy)
    g.add_node("experto_general", nodo_dummy)
    g.add_node("no_sabe",         nodo_dummy)
    g.add_edge(START, "clasificar")
    g.add_conditional_edges("clasificar", router_dummy,
        {"a": "experto_tecnico", "b": "experto_general"})
    g.add_edge("experto_tecnico", END)
    g.add_edge("experto_general", END)
    g.add_edge("no_sabe",         END)
    return g.compile()


def grafo_multiagente():
    g = StateGraph(EstadoMulti)
    g.add_node("coordinador",  nodo_dummy)
    g.add_node("investigador", nodo_dummy)
    g.add_node("analista",     nodo_dummy)
    g.add_node("redactor",     nodo_dummy)
    g.add_edge(START, "coordinador")
    g.add_conditional_edges("coordinador", router_multi,
        {"investigador":"investigador","analista":"analista",
         "redactor":"redactor","__end__":END})
    g.add_edge("investigador", "coordinador")
    g.add_edge("analista",     "coordinador")
    g.add_edge("redactor",     "coordinador")
    return g.compile()


def intentar_png(app, nombre):
    """Intenta exportar PNG — requiere grandalf o pygraphviz."""
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open(nombre, "wb") as f:
            f.write(png_data)
        print(f"  ✅ PNG guardado: {nombre}")
        return True
    except Exception as e:
        print(f"  ⚠️  No se pudo generar PNG: {e}")
        return False


def mostrar_mermaid(app, titulo):
    """Muestra el diagrama en formato Mermaid (texto)."""
    print(f"\n  📊 Diagrama Mermaid — {titulo}:")
    print("  " + "─" * 40)
    try:
        mermaid = app.get_graph().draw_mermaid()
        for linea in mermaid.split("\n"):
            print(f"  {linea}")
    except Exception as e:
        print(f"  Error: {e}")
    print("  " + "─" * 40)
    print("  💡 Pega este texto en https://mermaid.live para ver el diagrama")


def main():
    print("\n" + "═" * 52)
    print("  Visualización de Grafos LangGraph")
    print("═" * 52)

    grafos = [
        (grafo_minimo(),      "Grafo Mínimo",       "grafo_minimo.png"),
        (grafo_condicional(), "Grafo Condicional",  "grafo_condicional.png"),
        (grafo_multiagente(), "Sistema Multiagente","grafo_multiagente.png"),
    ]

    for app, titulo, archivo in grafos:
        print(f"\n  [{titulo}]")
        exito = intentar_png(app, archivo)
        if not exito:
            mostrar_mermaid(app, titulo)

    print("\n" + "═" * 52)
    print("  Tip: instala 'grandalf' para generar PNG automático:")
    print("  pip install grandalf")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()
