"""
Fachada pública del asistente.

Se mantiene el mismo import que usaban los notebooks y la UI:

    from src.assistant import TenerifeAssistant
"""

# Exportamos la clase desde el módulo core
from .assistant_core import TenerifeAssistant

__all__ = ["TenerifeAssistant"]
