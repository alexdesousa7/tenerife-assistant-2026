"""
Implementación central de la clase ``TenerifeAssistant``.
Compatible con la API chat.completions.create (SDK OpenAI 1.x).
"""

import json
from typing import Generator, Dict, List

from openai import OpenAI
from .rag import build_rag_pipeline
from .memory import ConversationMemory
from .assistant_utils import (
    should_use_rag,
    build_messages,
    process_llm_response,
    call_tool,
)
from .assistant_logging import logger, OPENAI_API_KEY, DEFAULT_MODEL, MAX_TOKENS_CONTEXT
from .tools import TOOLS_SCHEMAS


class TenerifeAssistant:
    """
    Clase pública del proyecto.
    """

    def __init__(
        self,
        pdf_path: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 500
    ):
        # Parámetros del modelo
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Cliente OpenAI (lazy loading)
        self.client = None

        print("DEBUG MODEL:", self.model)

        # Pipeline RAG
        logger.info(f"Cargando pipeline RAG desde {pdf_path}")
        self.rag_bundle = build_rag_pipeline(pdf_path)

        # Memoria de conversación
        self.memory = ConversationMemory(
            max_tokens=MAX_TOKENS_CONTEXT,
            model=self.model,
            client=None
        )

        logger.info("TenerifeAssistant listo para responder preguntas.")

    # ------------------------------------------------------------------
    # Lazy loading del cliente OpenAI
    # ------------------------------------------------------------------
    def _get_client(self):
        """Crea el cliente OpenAI solo cuando es necesario."""
        if self.client is None:
            if not OPENAI_API_KEY:
                return None  # Modo offline
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        return self.client

    # ------------------------------------------------------------------
    # 1️⃣  Respuesta completa (no streaming)
    # ------------------------------------------------------------------
    def answer(self, user_query: str) -> str:
        """
        Genera una respuesta completa.
        Si no hay OPENAI_API_KEY → modo offline (para CI/tests).
        """
        logger.info(f"Consulta recibida: '{user_query[:50]}'")

        # ------------------------------
        # 🔹 MODO OFFLINE (CI / tests)
        # ------------------------------
        if not OPENAI_API_KEY:
            logger.warning("Modo offline: generando respuesta sin modelo OpenAI")

            # Herramienta del tiempo
            if "tiempo" in user_query.lower() or "clima" in user_query.lower():
                result = call_tool("get_weather", {"location": "Santa Cruz", "date": "2024-06-15"})
                return f"Según los datos meteorológicos, la temperatura será de {result.get('temperature', '22°C')}."

            # RAG offline
            chunk = self.rag_bundle["chunks"][0]
            return f"{chunk[:150]}... [1]"

        # ------------------------------
        # 🔹 MODO NORMAL (con OpenAI)
        # ------------------------------
        client = self._get_client()
        use_rag = should_use_rag(user_query)

        messages = build_messages(
            user_query=user_query,
            rag_bundle=self.rag_bundle,
            memory=self.memory,
            use_rag=use_rag,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMAS,
                tool_choice="auto",
                max_tokens=2048,
            )

            answer = process_llm_response(
                response=response,
                original_query=user_query,
                use_rag=use_rag,
                memory=self.memory,
                rag_bundle=self.rag_bundle,
                client=client,
            )

            return answer

        except Exception:
            logger.exception("Error al generar la respuesta")
            return "Lo siento, hubo un problema técnico al procesar tu solicitud."

    # ------------------------------------------------------------------
    # 2️⃣  Respuesta en streaming
    # ------------------------------------------------------------------
    def answer_stream(self, user_query: str) -> Generator[str, None, None]:
        """
        Streaming usando la API chat.completions.create.
        En modo offline, devuelve respuesta simple.
        """
        logger.info(f"Streaming solicitado para: '{user_query[:40]}'")

        # Modo offline → no hay streaming real
        if not OPENAI_API_KEY:
            yield self.answer(user_query)
            return

        use_rag = should_use_rag(user_query)
        messages = build_messages(
            user_query=user_query,
            rag_bundle=self.rag_bundle,
            memory=self.memory,
            use_rag=use_rag,
        )

        client = self._get_client()

        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMAS,
                tool_choice="auto",
                max_tokens=2048,
                stream=True,
            )
        except Exception:
            logger.exception("Error al iniciar el stream")
            yield "Lo siento, hubo un problema al iniciar la respuesta."
            return

        full_answer = ""
        tool_triggered = False
        tool_info = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                full_answer += delta.content
                yield delta.content

            if delta.tool_calls:
                tool_triggered = True
                tc = delta.tool_calls[0]
                tool_info = {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                }
                break

        if tool_triggered:
            result = call_tool(tool_info["name"], tool_info["arguments"])
            followup = build_messages(
                user_query="Usa este resultado para responder al usuario: "
                + json.dumps(result),
                rag_bundle=self.rag_bundle,
                memory=self.memory,
                use_rag=use_rag,
            )

            try:
                second = client.chat.completions.create(
                    model=self.model,
                    messages=followup,
                    max_tokens=2048,
                )
                final_text = second.choices[0].message.content or ""
                for ch in final_text:
                    yield ch

            except Exception:
                logger.exception("Error en la segunda llamada al LLM")
                yield "Lo siento, hubo un error al procesar el resultado de la herramienta."

    # ------------------------------------------------------------------
    # Métodos auxiliares públicos
    # ------------------------------------------------------------------
    def get_memory_snapshot(self) -> List[Dict[str, str]]:
        return self.memory.get_history()

    def reset_memory(self) -> None:
        self.memory.reset()
        logger.info("Memoria de conversación reiniciada.")
