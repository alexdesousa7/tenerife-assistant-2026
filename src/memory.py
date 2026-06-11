"""
memory.py — Gestión de memoria conversacional para el Asistente de Tenerife
---------------------------------------------------------------------------

Este módulo implementa:
- Historial de conversación multiturno
- Límite de tokens para evitar desbordes
- Recorte automático del historial (por pares, para mantener coherencia)
- Función opcional de resumen (para conversaciones largas)
- Logging de operaciones críticas
"""

import logging
import tiktoken
from typing import List, Dict

logger = logging.getLogger(__name__)


def count_tokens(messages: List[Dict], model: str = "gpt-4.1-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Modelo '{model}' no reconocido por tiktoken. Usando cl100k_base.")
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        total += 4
        total += len(enc.encode(msg.get("content", "")))
        total += len(enc.encode(msg.get("role", "")))

    total += 2
    return total


class ConversationMemory:
    def __init__(self, max_tokens: int = 3000, model: str = "gpt-4.1-mini", client=None):
        self.history: List[Dict] = []
        self.max_tokens = max_tokens
        self.model = model
        self.client = client
        logger.info(f"ConversationMemory inicializada: max_tokens={max_tokens}, model={model}")


    def add_message(self, role: str, content: str):
        if not content:
            logger.warning(f"Se intentó añadir un mensaje vacío con rol '{role}'. Ignorando.")
            return

        self.history.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self):
        while count_tokens(self.history, self.model) > self.max_tokens and len(self.history) > 2:
            removed = []

            for i, msg in enumerate(self.history):
                if msg["role"] != "system":
                    removed_msg = self.history.pop(i)
                    removed.append(removed_msg["role"])
                    break

            if count_tokens(self.history, self.model) > self.max_tokens and len(self.history) > 1:
                for i, msg in enumerate(self.history):
                    if msg["role"] != "system":
                        removed_msg = self.history.pop(i)
                        removed.append(removed_msg["role"])
                        break

            if removed:
                logger.info(f"Historial recortado (eliminados roles: {removed}).")
            else:
                logger.warning("No se pueden recortar más mensajes sin perder instrucciones de sistema.")
                break

    def get_history(self) -> List[Dict]:
        return self.history.copy()

    def reset(self):
        self.history = []
        logger.info("Memoria conversacional reiniciada.")

    def status(self) -> Dict:
        return {
            "messages_count": len(self.history),
            "tokens_used": count_tokens(self.history, self.model),
            "max_tokens": self.max_tokens,
            "model": self.model,
        }


def summarize_history(history: List[Dict], client, model: str = "gpt-4.1-mini") -> str:
    if not history:
        raise ValueError("No se puede resumir un historial vacío.")

    text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Resume esta conversación manteniendo SOLO la información importante: "
                        "lugares mencionados, preferencias del usuario y decisiones tomadas."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error al generar resumen: {e}")
        recent = history[-2:] if len(history) >= 2 else history
        return " | ".join([f"{m['role']}: {m['content'][:100]}" for m in recent])
