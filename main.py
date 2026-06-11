import sys
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ----------------------------------------------------------------------
# Ajustar PYTHONPATH para permitir "import src..."
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.assistant import TenerifeAssistant
from src.memory import ConversationMemory

# ----------------------------------------------------------------------
# Configuración de Streamlit
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Turístico Tenerife 2026",
    page_icon="🌋",
    layout="centered",
)

# ----------------------------------------------------------------------
# Cabecera
# ----------------------------------------------------------------------
st.title("🏖️🌋 Asistente Turístico de Tenerife 2026 🏝️👙🌞")
st.markdown("*Descubre la isla de Tenerife con inteligencia artificial*")

st.divider()

# ----------------------------------------------------------------------
# Inicializar estado de la sesión
# ----------------------------------------------------------------------
if "assistant" not in st.session_state:
    with st.spinner("Cargando el asistente..."):
        try:
            st.session_state.assistant = TenerifeAssistant("data/TENERIFE.pdf")
        except Exception as e:
            st.error(f"Error al cargar el asistente: {e}")
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    st.markdown(f"**Modelo:** `{st.session_state.assistant.model}`")

    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.assistant.memory = ConversationMemory(
            max_tokens=3000,
            model=st.session_state.assistant.model,
            client=st.session_state.assistant.client
        )
        st.rerun()

    st.markdown("---")
    st.markdown("**Herramientas disponibles:**")
    st.markdown("- 🌤️  `get_weather` (tiempo real)")
    st.markdown("- 🚌  `get_bus_stops` (mock)")
    st.markdown("- 🍽️  `get_restaurant_offers` (mock)")
    st.markdown("- 📷  `get_webcams` (webcams en directo)")
    st.markdown("- 🚴  `get_bike_rentals` (alquiler de bicicletas)")
    # st.markdown("- 🚌  `get_transport_info` (rutas TITSA/Moovit)")  # opcional

# ----------------------------------------------------------------------
# Mostrar historial del chat
# ----------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------------------------------------------
# Prompt del usuario
# ----------------------------------------------------------------------
prompt = st.chat_input("¿Qué quieres saber sobre Tenerife?")

# ----------------------------------------------------------------------
# Procesar respuesta del asistente
# ----------------------------------------------------------------------
if prompt:
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Detectar si la pregunta requiere herramientas
    needs_tool = any(keyword in prompt.lower() for keyword in [
        # Clima
        "clima", "tiempo", "temperatura",

        # Transporte
        "guagua", "parada", "transporte", "bus", "titsa", "moovit",

        # Restaurantes
        "restaurante", "comida", "oferta", "menú",

        # Webcams
        "webcam", "cámara", "camara", "playa en directo",

        # Bicicletas
        "bici", "bicicleta", "alquiler de bicicletas",
    ])

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # Si usa herramienta → mensaje fijo
            if needs_tool:
                placeholder.markdown("🔧 *Consultando herramientas…*")
            else:
                # Si NO usa herramienta → animación giratoria
                spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                for frame in spinner_frames * 2:
                    placeholder.markdown(f"{frame} *Pensando…*")
                    time.sleep(0.08)

            # Obtener respuesta completa
            final_text = st.session_state.assistant.answer(prompt)

            # Streaming manual del texto final
            full_response = ""
            for token in final_text.split():
                full_response += token + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)

            placeholder.markdown(full_response)

        except Exception as e:
            full_response = (
                "Lo siento, hubo un error al procesar tu consulta.\n\n"
                f"*Detalles técnicos: {str(e)}*"
            )
            placeholder.markdown(full_response)

        # Guardar respuesta
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# ----------------------------------------------------------------------
# Pie de página
# ----------------------------------------------------------------------
st.divider()
