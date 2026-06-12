# 🏝️ Asistente Turístico de Tenerife 2026 🌋

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenAI-LLM-412991?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-0099CC?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Project-LLM-blue?style=for-the-badge" />
</p>

---

### Módulo: Large Language Models  
## **Descripción del proyecto**

Este proyecto implementa un **asistente turístico conversacional** basado en modelos de lenguaje (LLM), capaz de:

- Recuperar información desde una **guía turística en PDF** mediante RAG.  
- Mantener **diálogo multiturno** con memoria conversacional.  
- Invocar **herramientas externas (function calling)** para obtener datos en tiempo real.  
- Ejecutarse mediante una **interfaz web en Streamlit**.  
- Generar un **notebook reproducible** con todo el pipeline.

El asistente responde preguntas sobre Tenerife combinando:

- **RAG (Retrieval Augmented Generation)**  
- **Memoria conversacional**  
- **Herramientas externas**  
- **LLM comercial (OpenAI)**  
- **Interfaz Streamlit**

---

# **Arquitectura del sistema**

```
┌──────────────────────────┐
│        Streamlit         │ ← Interfaz web
└──────────────┬───────────┘
               │
┌──────────────▼──────────────┐
│     TenerifeAssistant       │ ← Núcleo del asistente
│     (assistant_core.py)     │
└──────────────┬──────────────┘
               │
     ┌─────────┴──────────┐
     │                    │
┌────▼──────┐      ┌──────▼──────┐
│   RAG     │      │  Tools FC   │
│ (FAISS)   │      │ FunctionCall│
└───────────┘      └─────────────┘
```

---

# Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/asistente-tenerife.git
cd asistente-tenerife
```

---

# Instalación automática (Makefile) — Recomendado

Este proyecto incluye un **Makefile** que automatiza:

- creación del entorno virtual  
- instalación de dependencias  
- registro del kernel de Jupyter  
- preparación del proyecto  

## Ejecutar instalación automática:

```bash
make setup
```

Esto ejecutará:

- `make venv` → crea `.venv/`  
- `make install` → instala dependencias  
- registra el kernel `tenerife-assistant`  
- deja todo listo para ejecutar el proyecto  

---

# Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

WEATHER_API_KEY=tu_api_key_openweather
BUS_API_URL=
RESTAURANT_API_URL=
```

**Dónde obtener las claves:**

- OpenAI API Key → [https://platform.openai.com/](https://platform.openai.com/)  
- OpenWeather API Key → [https://openweathermap.org/](https://openweathermap.org/)

El resto de herramientas no usan API Keys actualmente.

---

# Instalación manual (alternativa)

## 1. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecutar el asistente en local (Streamlit)

```bash
streamlit run main.py
```

Esto abrirá la interfaz web:

- Chat estilo ChatGPT  
- Respuestas con RAG  
- Llamadas a herramientas  
- Streaming de tokens  
- Botón para limpiar memoria  

---

# Ejecutar tests

```bash
pytest -q
```

---

# Herramientas externas (Function Calling)

El asistente incluye **5 herramientas**:

| Tool | Descripción |
|------|-------------|
| `get_weather` | Predicción del tiempo real |
| `get_bus_stops` | Paradas de guagua (mock) |
| `get_restaurant_offers` | Ofertas gastronómicas (mock) |
| `get_webcams` | Webcams en directo |
| `get_bike_rentals` | Alquiler de bicicletas |
| `get_transport_info` | Rutas TITSA/Moovit (opcional) |

Todas están definidas en:

```
src/tools.py
```

Y registradas en:

```
TOOLS
TOOLS_SCHEMAS
```

---

# RAG (Retrieval Augmented Generation)

El pipeline RAG:

- Carga el PDF `data/TENERIFE.pdf`
- Divide en chunks
- Genera embeddings
- Construye un índice FAISS
- Recupera fragmentos relevantes
- Añade citaciones numéricas a las respuestas

Código en:

```
src/rag.py
```

---

# Memoria conversacional

Implementada en:

```
src/memory.py
```

Características:

- Guarda historial de turnos  
- Resume automáticamente si supera el límite de tokens  
- Se puede resetear desde Streamlit  

---

# Modelos de IA utilizados

### **OpenAI GPT‑4.1-mini**  
Para generación de texto y function calling.

### **OpenAI text-embedding-3-small**  
Para embeddings del RAG.

Ambos configurables desde `.env`.

---

# Gestión de API Keys

Las claves se cargan mediante:

```
python-dotenv
```

Nunca se incluyen en el repositorio.

---

# Cómo funciona el asistente

1. El usuario escribe una pregunta.  
2. El sistema decide si usar:
   - RAG  
   - Tools  
   - O ambos  
3. Si usa tools → ejecuta la función y pasa el resultado al LLM.  
4. Si usa RAG → recupera fragmentos y cita fuentes.  
5. El modelo genera la respuesta final.  
6. Streamlit la muestra con streaming.  

---

# Estructura del proyecto

```
tenerife-assistant-2026/
│
├── Makefile                         # Automatización: setup, instalación, limpieza, etc.
├── README.md                        # Documentación principal del proyecto
├── asistente_tenerife_2026.ipynb    # Notebook final del proyecto
├── asistente_tenerife_test01.ipynb  # Notebook de pruebas 1
├── asistente_tenerife_test02.ipynb  # Notebook de pruebas 2
├── env.template                     # Plantilla para crear el archivo .env
├── main.py                          # Aplicación Streamlit (interfaz del asistente)
├── requirements.txt                 # Dependencias del proyecto
│
├── actividad/
│   └── Enunciado Entrega Final.pdf  # Documento oficial del ejercicio
│
├── data/
│   └── TENERIFE.pdf                 # Guía turística usada para el RAG
│
├── src/
│   ├── assistant.py                 # Clase principal del asistente (RAG + Tools + Memoria)
│   ├── assistant_core.py            # Lógica central de interacción con el LLM
│   ├── assistant_logging.py         # Logging y trazabilidad de llamadas y errores
│   ├── assistant_utils.py           # Utilidades auxiliares del asistente
│   ├── memory.py                    # Implementación de la memoria conversacional
│   ├── rag.py                       # Pipeline RAG: embeddings, FAISS, recuperación
│   └── tools.py                     # Definición de herramientas externas (function calling)
│
└── tests/
    ├── conftest.py                  # Fixtures para PyTest
    └── test_core.py                 # Tests unitarios del asistente
```

---

# Resumen

### Fortalezas
- RAG real sobre PDF turístico  
- Function calling con varias herramientas  
- Memoria conversacional  
- Interfaz Streamlit  
- Notebook reproducible  

### Limitaciones
- Chunking básico  
- Sin reranking  
- Evaluación limitada  

---