"""
Funciones auxiliares del asistente:
- Decisión de usar RAG vs. herramientas externas.
- Construcción de contexto RAG (con citación numerada).
- Construcción de la lista de mensajes para OpenAI.
- Llamada a herramientas (function calling).
- Procesado de la respuesta del LLM (incluye tool‑calls y citas de fuentes).
"""

import json
from typing import Dict, Any, List, Optional

from openai import OpenAI
from .memory import ConversationMemory, summarize_history, count_tokens
from .rag import retrieve
from .tools import TOOLS
from .assistant_logging import logger, DEFAULT_MODEL, MAX_TOKENS_CONTEXT


# ----------------------------------------------------------------------
# 1️⃣  Decisión: ¿usar RAG o herramienta?
# ----------------------------------------------------------------------
def should_use_rag(query: str) -> bool:
    """
    Decide si la consulta debe resolverse vía RAG o mediante una herramienta.
    """
    weather_kw = {"tiempo", "clima", "lluvia", "temperatura", "pronóstico"}
    transport_kw = {"guagua", "bus", "autobús", "transporte", "ruta", "horario"}
    food_kw = {"restaurante", "comida", "oferta", "menú", "cocina"}

    low = query.lower()
    if any(w in low for w in weather_kw):
        return False
    if any(w in low for w in transport_kw):
        return False
    if any(w in low for w in food_kw):
        return False
    return True


# ----------------------------------------------------------------------
# 2️⃣  Construcción de contexto RAG con citación numerada
# ----------------------------------------------------------------------
def build_rag_context(query: str, rag_bundle: Dict[str, Any], k: int = 4) -> str:
    """
    Recupera los *k* fragmentos más relevantes del pipeline RAG y los
    devuelve con numeración de fuente. Además, guarda en rag_bundle
    la lista de citas para poder añadirlas luego a la respuesta final.
    """
    results = retrieve(
        query=query,
        chunks=rag_bundle["chunks"],
        index=rag_bundle["index"],
        metadata=rag_bundle["metadata"],
        k=k,
    )

    if not results:
        logger.warning("RAG no devolvió fragmentos para la consulta.")
        rag_bundle.pop("last_citations", None)
        return ""

    # Guardamos las citas para usarlas después en la respuesta final
    rag_bundle["last_citations"] = [
        f"[{i+1}] {r['source']}" for i, r in enumerate(results)
    ]

    parts = [f"[{i+1}] {r['chunk'].strip()}" for i, r in enumerate(results)]
    return "\n".join(parts)


# ----------------------------------------------------------------------
# 3️⃣  Construir la lista de mensajes para OpenAI
# ----------------------------------------------------------------------
def build_messages(
    user_query: str,
    rag_bundle: Dict[str, Any],
    memory: ConversationMemory,
    use_rag: bool,
    client: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """
    Construye los mensajes en el orden esperado por la API.
    Si use_rag=True, añade contexto de la guía con numeración y
    el modelo sabe que debe citar esos números.
    """
    system_msg = {
        "role": "system",
        "content": (
            "Eres un asistente turístico especializado en Tenerife. "
            "Responde con información clara y cita la fuente usando los números entre corchetes. "
            "Si no hay información suficiente en el contexto, dilo explícitamente."
        ),
    }

    msgs: List[Dict[str, str]] = [system_msg]

    # Añadir contexto RAG
    if use_rag:
        rag_context = build_rag_context(user_query, rag_bundle)
        if rag_context:
            msgs.append(
                {
                    "role": "system",
                    "content": (
                        "Información de la guía turística (usa y cita los números entre corchetes):\n"
                        f"{rag_context}"
                    ),
                }
            )

    # Historial
    msgs.extend(memory.get_history())

    # Mensaje del usuario
    msgs.append({"role": "user", "content": user_query})

    # Verificar límite de tokens
    total_tokens = count_tokens(msgs, DEFAULT_MODEL)
    if total_tokens > MAX_TOKENS_CONTEXT:
        logger.info(
            f"Contexto supera {MAX_TOKENS_CONTEXT} tokens ({total_tokens}). "
            "Se crea resumen del historial."
        )

        summary = summarize_history(
            memory.get_history(),
            client=memory.client,
            model=DEFAULT_MODEL,
        )

        msgs = [
            system_msg,
            {"role": "assistant", "content": f"Resumen de la conversación previa:\n{summary}"},
            {"role": "user", "content": user_query},
        ]

    return msgs


# ----------------------------------------------------------------------
# 4️⃣  Ejecutar herramienta externa
# ----------------------------------------------------------------------
def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta la herramienta registrada en ``src.tools``.
    """
    logger.info(f"Llamada a herramienta: {tool_name} args={arguments}")

    if tool_name not in TOOLS:
        err = f"Herramienta desconocida: {tool_name}"
        logger.error(err)
        return {"error": err}

    try:
        return TOOLS[tool_name](**arguments)
    except Exception as exc:
        logger.exception(f"Error ejecutando {tool_name}")
        return {"error": f"Error al ejecutar {tool_name}: {str(exc)}"}


# ----------------------------------------------------------------------
# 5️⃣  Procesar la respuesta del LLM
# ----------------------------------------------------------------------
def process_llm_response(
    response,
    original_query: str,
    use_rag: bool,
    memory: ConversationMemory,
    rag_bundle: Dict[str, Any],
    client: OpenAI,
) -> str:
    """
    Procesa la respuesta del modelo, incluyendo tool-calls y, si procede,
    añade una sección de 'Fuentes:' con las citas generadas por el RAG.
    """
    msg = response.choices[0].message

    # Tool call
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        tool_name = tc.function.name

        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
            logger.error("No se pudieron decodificar los argumentos de la herramienta")

        tool_result = call_tool(tool_name, args)

        memory.add_message("user", original_query)
        memory.add_message(
            "assistant",
            f"[Llamada a herramienta: {tool_name}]\nArgs: {json.dumps(args)}",
        )
        memory.add_message(
            "assistant",
            f"[Resultado herramienta]: {json.dumps(tool_result)}",
        )

        followup_msgs = build_messages(
            user_query=f"Usa el siguiente resultado para responder al usuario: {json.dumps(tool_result)}",
            rag_bundle=rag_bundle,
            memory=memory,
            use_rag=use_rag,
        )

        second_resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=followup_msgs,
            tools=[],
            max_tokens=2048,
        )

        final = second_resp.choices[0].message.content or ""
        memory.add_message("assistant", final)
        return final

    # Respuesta directa
    final = msg.content or ""

    # Añadir sección de fuentes si se usó RAG y hay citas disponibles
    if use_rag and "last_citations" in rag_bundle:
        citations = rag_bundle.get("last_citations") or []
        if citations:
            final += "\n\n**Fuentes:**\n" + "\n".join(citations)

    memory.add_message("user", original_query)
    memory.add_message("assistant", final)
    return final
