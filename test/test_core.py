import pytest
from src.assistant import TenerifeAssistant

@pytest.fixture(scope="module")
def assistant():
    return TenerifeAssistant("data/TENERIFE.pdf")

def test_rag_response(assistant):
    resp = assistant.answer("¿Qué playas tranquilas hay cerca de La Laguna?")
    assert "[1]" in resp or "[2]" in resp   # debe citar una fuente

def test_weather_tool(assistant):
    resp = assistant.answer("¿Qué tiempo hará en Santa Cruz el 2024‑06‑15?")
    assert "°C" in resp or "temperatura" in resp.lower()
