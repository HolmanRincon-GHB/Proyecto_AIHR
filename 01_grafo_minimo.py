"""
============================================================
  01_grafo_minimo.py — El grafo más simple posible
  Seminario: Agentes de IA · Sesión 4

  Objetivo: entender los 4 conceptos fundamentales de
  LangGraph antes de construir algo complejo.

  Conceptos que aprenderás aquí:
    • State     → el diccionario compartido entre nodos
    • Node      → función Python que transforma el State
    • Edge      → conexión entre nodos
    • Graph     → el contenedor que orquesta todo

  Este grafo tiene 2 nodos y 1 edge:
      START → nodo_saludo → nodo_respuesta → END

  Uso:
      python 01_grafo_minimo.py
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from s4_llm_factory import crear_llm

load_dotenv()

# ═══════════════════════════════════════════════════════════
#  PASO 1 — Definir el State
#  El State es el diccionario que viaja entre todos los nodos.
#  Cada nodo puede leerlo y modificarlo.
#
#  ✏️ MODIFICA AQUÍ: agrega o quita campos según lo que
#     necesites pasar entre nodos en tu grafo.
# ═══════════════════════════════════════════════════════════
class MiEstado(TypedDict):
    mensajes: list          # historial de mensajes
    entrada_usuario: str    # el texto que escribió el usuario
    respuesta_final: str    # lo que devuelve el sistema


# ═══════════════════════════════════════════════════════════
#  PASO 2 — Definir los Nodos
#  Cada nodo es una función Python que:
#    - Recibe el State actual
#    - Hace algo (llama al LLM, procesa datos, etc.)
#    - Devuelve un dict con los campos a actualizar
#
#  ✏️ MODIFICA AQUÍ: cambia lo que hace cada nodo.
# ═══════════════════════════════════════════════════════════

def nodo_preparar(estado: MiEstado) -> dict:
    """
    Nodo 1: prepara el contexto antes de llamar al LLM.
    En este ejemplo solo formatea el mensaje.
    En producción podría: validar input, buscar contexto, etc.
    """
    print(f"  [Nodo 1 - Preparar] Procesando: '{estado['entrada_usuario']}'")
    
    # Agrega el mensaje del usuario al historial
    mensajes_actualizados = estado["mensajes"] + [
        HumanMessage(content=estado["entrada_usuario"])
    ]
    return {"mensajes": mensajes_actualizados}


def nodo_responder(estado: MiEstado) -> dict:
    """
    Nodo 2: llama al LLM y genera la respuesta.
    Recibe el State ya preparado por el nodo anterior.
    """
    print(f"  [Nodo 2 - Responder] Llamando al LLM...")
    
    llm = crear_llm()
    
    # ✏️ MODIFICA AQUÍ: cambia el system prompt del nodo
    from langchain_core.messages import SystemMessage
    mensajes_con_sistema = [
        SystemMessage(content="Eres un asistente conciso. Responde en español en máximo 2 oraciones.")
    ] + estado["mensajes"]
    
    respuesta = llm.invoke(mensajes_con_sistema)
    
    mensajes_actualizados = estado["mensajes"] + [respuesta]
    return {
        "mensajes":       mensajes_actualizados,
        "respuesta_final": respuesta.content,
    }


# ═══════════════════════════════════════════════════════════
#  PASO 3 — Construir el Grafo
#  Conectamos nodos con edges (aristas)
# ═══════════════════════════════════════════════════════════

def construir_grafo():
    # Crea el grafo con el esquema de estado
    grafo = StateGraph(MiEstado)

    # Agrega los nodos
    grafo.add_node("preparar",  nodo_preparar)
    grafo.add_node("responder", nodo_responder)

    # ✏️ MODIFICA AQUÍ: cambia las conexiones entre nodos
    # Conecta: START → preparar → responder → END
    grafo.add_edge(START,       "preparar")
    grafo.add_edge("preparar",  "responder")
    grafo.add_edge("responder", END)

    # Compila el grafo (lo convierte en un Runnable)
    return grafo.compile()


def main():
    app = construir_grafo()

    print("\n" + "═" * 52)
    print("  Grafo Mínimo LangGraph")
    print("  START → preparar → responder → END")
    print("═" * 52)
    print("  Escribe algo para ver el flujo. Salir: salir\n")

    while True:
        try:
            entrada = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not entrada or entrada.lower() == "salir":
            break

        # El estado inicial que entra al grafo
        estado_inicial = {
            "mensajes":        [],
            "entrada_usuario": entrada,
            "respuesta_final": "",
        }

        print()
        try:
            # .invoke() ejecuta el grafo completo
            resultado = app.invoke(estado_inicial)
            print(f"\n  Respuesta final: {resultado['respuesta_final']}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")

    print("\n  👋 Hasta luego.\n")


if __name__ == "__main__":
    main()
