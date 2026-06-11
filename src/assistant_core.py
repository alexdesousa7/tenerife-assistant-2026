"""
Implementación central de la clase ``TenerifeAssistant``.
Compatible con la API chat.completions.create (SDK OpenAI 1.x).
"""

import json
import time
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

    Ejemplo:
        >>> from src.assistant import TenerifeAssistant
        >>> a = TenerifeAssistant("data/TENERIFE.pdf")
        >>> a.answer("¿Qué playas hay en Adeje?")
    """

    def __init__(
        self,
        pdf_path: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 500
    ):
        # Guardamos parámetros del modelo
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Cliente OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        print("DEBUG MODEL:", self.model)

        # Pipeline RAG
        logger.info(f"Cargando pipeline RAG desde {pdf_path}")
        self.rag_bundle = build_rag_pipeline(pdf_path)

        # Memoria de conversación
        self.memory = ConversationMemory(
            max_tokens=MAX_TOKENS_CONTEXT,
            model=self.model,
            client=self.client   # ← NECESARIO
        )

        logger.info("TenerifeAssistant listo para responder preguntas.")

    # ------------------------------------------------------------------
    # 1️⃣  Respuesta completa (no streaming)
    # ------------------------------------------------------------------
    def answer(self, user_query: str) -> str:
        """
        Genera una respuesta completa a la pregunta del usuario.
        """
        start = time.time()
        logger.info(f"Consulta recibida: '{user_query[:50]}'")

        use_rag = should_use_rag(user_query)

        messages = build_messages(
            user_query=user_query,
            rag_bundle=self.rag_bundle,
            memory=self.memory,
            use_rag=use_rag,
        )

        try:
            response = self.client.chat.completions.create(
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
                client=self.client,
            )

        except Exception:
            logger.exception("Error al generar la respuesta")
            answer = (
                "Lo siento, hubo un problema técnico al procesar tu solicitud. "
                "Inténtalo de nuevo más tarde."
            )
            self.memory.add_message("assistant", answer)

        logger.info(f"Respuesta generada en {time.time() - start:.2f}s")
        return answer

    # ------------------------------------------------------------------
    # 2️⃣  Respuesta en streaming (API chat.completions)
    # ------------------------------------------------------------------
    def answer_stream(self, user_query: str) -> Generator[str, None, None]:
        """
        Streaming usando la API chat.completions.create,
        compatible con gpt‑4o-mini y el SDK OpenAI 1.x.
        """
        logger.info(f"Streaming solicitado para: '{user_query[:40]}'")

        use_rag = should_use_rag(user_query)

        messages = build_messages(
            user_query=user_query,
            rag_bundle=self.rag_bundle,
            memory=self.memory,
            use_rag=use_rag,
        )

        # 1️⃣ Primera llamada (streaming)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMAS,
                tool_choice="auto",
                max_tokens=2048,
                stream=True,
            )
        except Exception:
            logger.exception("Error al iniciar el stream")
            yield (
                "Lo siento, hubo un problema al iniciar la respuesta. "
                "Inténtalo de nuevo más tarde."
            )
            return

        full_answer = ""
        tool_triggered = False
        tool_info = {}

        # 2️⃣ Consumimos el stream
        for chunk in stream:
            delta = chunk.choices[0].delta

            # Texto normal
            if delta.content:
                full_answer += delta.content
                yield delta.content

            # Tool call detectada
            if delta.tool_calls:
                tool_triggered = True
                tc = delta.tool_calls[0]
                tool_info = {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                }
                break

        # 3️⃣ Si hubo tool call → segunda pasada
        if tool_triggered:
            result = call_tool(tool_info["name"], tool_info["arguments"])

            # Guardamos en memoria
            self.memory.add_message("user", user_query)

            tool_call_msg = (
                "[Llamada a herramienta: "
                + tool_info["name"]
                + "] Args: "
                + json.dumps(tool_info["arguments"])
            )
            self.memory.add_message("assistant", tool_call_msg)

            tool_result_msg = "[Resultado herramienta]: " + json.dumps(result)
            self.memory.add_message("assistant", tool_result_msg)

            # Segunda llamada al modelo
            followup = build_messages(
                user_query="Usa este resultado para responder al usuario: "
                + json.dumps(result),
                rag_bundle=self.rag_bundle,
                memory=self.memory,
                use_rag=use_rag,
            )

            try:
                second = self.client.chat.completions.create(
                    model=self.model,
                    messages=followup,
                    max_tokens=2048,
                )
                final_text = second.choices[0].message.content or ""
                for ch in final_text:
                    yield ch
                self.memory.add_message("assistant", final_text)

            except Exception:
                logger.exception("Error en la segunda llamada al LLM")
                err_msg = (
                    "Lo siento, hubo un error al procesar el resultado de la herramienta."
                )
                yield err_msg
                self.memory.add_message("assistant", err_msg)

    # ------------------------------------------------------------------
    # Métodos auxiliares públicos
    # ------------------------------------------------------------------
    def get_memory_snapshot(self) -> List[Dict[str, str]]:
        """Devuelve una copia del historial de conversación."""
        return self.memory.get_history()

    def reset_memory(self) -> None:
        """Vacía el historial de conversación."""
        self.memory.reset()
        logger.info("Memoria de conversación reiniciada.")
