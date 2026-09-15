"""s6_llm_factory.py — Helper compartido Sesión 6"""
import os
from dotenv import load_dotenv
load_dotenv()

def crear_llm(temperature=None, max_tokens=None, modelo=None):
    """
    ✏️ El parámetro `modelo` permite usar un modelo distinto al de .env
    para un nodo puntual. Es útil porque no todos los modelos de Groq
    manejan igual de bien cada tarea:
      - create_agent (herramientas)  -> más confiable con GROQ_MODEL normal
      - with_structured_output       -> más confiable con GROQ_MODEL_ESTRUCTURADO
    """
    proveedor   = os.getenv("LLM_PROVIDER", "groq")
    temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens  = max_tokens  if max_tokens  is not None else int(os.getenv("MAX_TOKENS", "1024"))
    if proveedor == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
                        model=modelo or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                        temperature=temperature, max_tokens=max_tokens)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=modelo or os.getenv("OLLAMA_MODEL", "llama3.2"),
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                          temperature=temperature)
