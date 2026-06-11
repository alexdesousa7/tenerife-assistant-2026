# Variables

VENV        := .venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
BLACK       := $(VENV)/bin/black
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
STREAMLIT   := $(VENV)/bin/streamlit

KERNEL_NAME := tenerife-assistant
NOTEBOOK    := asistente_tenerife_2026.ipynb

# Crear entorno virtual

venv:
    python3 -m venv $(VENV)

# Instalar dependencias

install: venv
    $(PIP) install --upgrade pip
    $(PIP) install -r requirements.txt

# Setup completo (entorno + kernel Jupyter)

setup: install
    @echo "Registrando kernel de Jupyter..."
    @$(PYTHON) -m ipykernel install --user --name $(KERNEL_NAME) --display-name "Tenerife Assistant"
    @echo "Entorno listo. Abre el notebook '$(NOTEBOOK)' usando el kernel '$(KERNEL_NAME)'."

# Generar lockfile sencillo

freeze: install
    $(PIP) freeze > requirements.lock.txt
    @echo "requirements.lock.txt creado"

# Ejecutar Jupyter Notebook

notebook:
    $(PYTHON) -m jupyter notebook $(NOTEBOOK)

# Ejecutar aplicación Streamlit

app:
    $(STREAMLIT) run main.py

# Asistente en modo consola

run:
    @echo "Iniciando asistente en consola (escribe 'salir' para terminar)"
    $(PYTHON) -c "\
import sys; \
from src.assistant_core import TenerifeAssistant; \
assistant = TenerifeAssistant('data/TENERIFE.pdf'); \
while True: \
    q = input('\n>>> Pregunta: '); \
    if q.strip().lower() in {'salir', 'exit', 'quit'}: \
        sys.exit(); \
    print('\nRespuesta:', assistant.answer(q))"

# Tests

test: install
    $(PYTEST) -v tests

# Linter

lint: install
    $(RUFF) check src

# Formateo de código

format: install
    $(BLACK) src

# Limpiar cachés

clean:
    rm -rf __pycache__ */__pycache__ .ipynb_checkpoints .streamlit

# Eliminar entorno virtual

distclean: clean
    rm -rf $(VENV)
    @echo "Entorno virtual eliminado. Usa 'make setup' para recrearlo."

# Ayuda

help:
    @echo "Objetivos disponibles:"
    @echo "  make setup       → crea entorno + instala deps + registra kernel Jupyter"
    @echo "  make venv        → crea el entorno virtual"
    @echo "  make install     → instala dependencias"
    @echo "  make notebook    → abre el notebook principal ($(NOTEBOOK))"
    @echo "  make app         → lanza la app Streamlit (main.py)"
    @echo "  make run         → demo interactiva en consola"
    @echo "  make test        → ejecuta tests"
    @echo "  make lint        → ejecuta ruff"
    @echo "  make format      → formatea código"
    @echo "  make clean       → limpia cachés"
    @echo "  make distclean   → elimina el entorno virtual"
