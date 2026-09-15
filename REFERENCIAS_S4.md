# Referencias y Recursos — Sesión 4
## Agentes de IA · Seminario · Semana 2, Sesión 4

---

## 🔧 Documentación oficial

| Recurso | URL | Para qué sirve |
|---|---|---|
| LangGraph Quickstart | https://docs.langchain.com/oss/python/langgraph/quickstart | Tutorial oficial paso a paso |
| LangGraph Concepts | https://docs.langchain.com/oss/python/langgraph/persistence | State, Nodes, Edges, Checkpointers |
| StateGraph | https://reference.langchain.com/python/langgraph/graph/state/StateGraph| API completa del grafo |
| add_messages reducer | https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers | Cómo acumular mensajes |
| MemorySaver | https://reference.langchain.com/javascript/langchain-langgraph/index/MemorySaver | Persistencia entre invocaciones |
| Mermaid Live Editor | https://mermaid.live | Visualiza diagramas Mermaid en el navegador |

---

## 📄 Lectura recomendada

### Multi-agent Systems
> **A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges** — Li et al., 2024
> Link: https://link.springer.com/article/10.1007/s44336-024-00009-2
> **Qué leer**: Figura 1 (arquitecturas de sistemas multiagente) y la Tabla 1.

### Por qué grafos en lugar de chains
> **LangGraph: Building Stateful, Multi-Actor Applications** — Blog LangChain
> Link: https://blog.langchain.dev/langgraph/
> **Qué leer**: Todo el artículo — es corto y explica la motivación detrás de LangGraph.

---

## 💡 Comandos rápidos

```bash
# Instalar LangGraph
pip install langgraph

# Ejecutar en orden
python 01_grafo_minimo.py          # State + Node + Edge
python 02_grafo_condicional.py     # Router inteligente
python 03_multiagente.py           # 4 agentes coordinados
python 04_visualizar_grafo.py      # Exporta diagrama
python 05_agente_completo_s4.py    # Sistema completo ← BASE S5

# Ver el diagrama Mermaid en el navegador
# 1. Corre: python 04_visualizar_grafo.py
# 2. Copia el texto Mermaid
# 3. Pega en: https://mermaid.live
```

---

## 🧠 Glosario — Sesión 4

| Término | Definición simple |
|---|---|
| **State** | Diccionario compartido que viaja entre todos los nodos del grafo |
| **Node** | Función Python que recibe el State, lo transforma y devuelve cambios |
| **Edge** | Conexión entre nodos — puede ser fija o condicional |
| **Conditional Edge** | Edge cuyo destino se decide en tiempo de ejecución según el State |
| **Router** | Función que lee el State y devuelve el nombre del nodo siguiente |
| **StateGraph** | Clase de LangGraph para construir grafos tipados |
| **add_messages** | Reducer que acumula mensajes en lugar de sobreescribirlos |
| **MemorySaver** | Checkpointer que persiste el State entre invocaciones del grafo |
| **thread_id** | Identificador de sesión para el MemorySaver — como session_id en S3 |
| **compile()** | Transforma el grafo en un Runnable ejecutable |
| **Checkpointer** | Sistema que guarda snapshots del State para recuperación |

---

## 🗂️ LangChain vs LangGraph

| | LangChain Agents | LangGraph |
|---|---|---|
| Flujo | Secuencial con tools | Grafo con ciclos |
| Control | LLM decide todo | Tú defines el flujo |
| Múltiples agentes | Difícil | Nativo |
| Memoria entre llamadas | Manual | MemorySaver automático |
| Debugging | Difícil | Muy visual (Mermaid) |
| Ideal para | Agentes simples | Sistemas complejos |

---

*Próxima sesión: Sesión 5 — Backend REST con FastAPI*
