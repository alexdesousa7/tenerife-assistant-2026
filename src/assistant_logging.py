"""
Configuración de logging y constantes globales usadas por el asistente.
Mantener todo aquí evita repetir código en varios módulos y permite
cambiar rápidamente nivel, formato o variables de entorno.
"""

import logging
import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Configuración del logger del módulo
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Carga de variables de entorno
# ----------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning(
        "OPENAI_API_KEY no está configurada. "
        "El asistente no podrá llamar al modelo."
    )

# Modelo por defecto (puede sobrescribirse con la variable de entorno)
DEFAULT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Tokens que reservamos al modelo para que la respuesta no lo agote
MAX_TOKENS_CONTEXT: int = int(os.getenv("MAX_TOKENS_CONTEXT", "2500"))

# ----------------------------------------------------------------------
# Configuración de logging (nivel INFO por defecto, se puede sobreescribir
# con la variable de entorno LOG_LEVEL)
# ----------------------------------------------------------------------
log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Logger propio del asistente (se reutiliza en todos los sub‑módulos)
logger = logging.getLogger("TenerifeAssistant")
