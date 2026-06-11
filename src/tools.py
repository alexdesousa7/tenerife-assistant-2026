"""
tools.py — Herramientas externas para el Asistente Turístico de Tenerife
------------------------------------------------------------------------

Este módulo define las herramientas que el asistente puede llamar mediante
Function Calling de OpenAI.

Incluye:
- get_weather: API real (OpenWeatherMap)
- get_bus_stops: API real + fallback mock
- get_restaurant_offers: API real + fallback mock
- Manejo robusto de errores
- Logging integrado
"""

import os
import json
import logging
import argparse
import requests
import datetime

from typing import Dict, Any, Optional
from dataclasses import dataclass
from requests import RequestException

# Logging
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_TIMEOUT = 10
WIND_SPEED_CONVERSION = 3.6  # m/s → km/h


# ----------------------------------------------------------------------
# MODELO DE RESPUESTA METEOROLÓGICA
# ----------------------------------------------------------------------
@dataclass
class WeatherResponse:
    location: str
    date: str
    temperature: float
    condition: str
    humidity: int
    wind_kmh: float
    error: Optional[str] = None


# ----------------------------------------------------------------------
# 1. HERRAMIENTA REAL: Predicción del tiempo
# ----------------------------------------------------------------------
def get_weather(location: str, date: str = None) -> Dict[str, Any]:

    # 1. Normalizar ubicaciones comunes
    NORMALIZED_LOCATIONS = {
        "tenerife": "Santa Cruz de Tenerife",
        "norte de tenerife": "Puerto de la Cruz",
        "sur de tenerife": "Adeje",
    }

    loc_key = location.lower().strip()
    if loc_key in NORMALIZED_LOCATIONS:
        location = NORMALIZED_LOCATIONS[loc_key]

    # 2. Si no viene fecha → usar hoy
    if date is None:
        date = datetime.date.today().isoformat()

    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return {
            "error": "Falta WEATHER_API_KEY en .env",
            "location": location,
            "date": date,
        }

    try:
        logger.info(f"Consultando tiempo para {location}")

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{location},ES",
            "appid": api_key,
            "units": "metric",
            "lang": "es",
        }

        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

        # 3. Fallback técnico si la API devuelve 404
        if response.status_code == 404 and location == "Santa Cruz de Tenerife":
            logger.info("Fallback a Icod el Alto para OpenWeatherMap")
            location = "Icod el Alto"
            params["q"] = f"{location},ES"
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)

        response.raise_for_status()
        data = response.json()

        return {
            "location": location,
            "date": date,
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_kmh": round(data["wind"]["speed"] * WIND_SPEED_CONVERSION, 1),
        }

    except RequestException as e:
        logger.error(f"Error en API de OpenWeatherMap: {e}")
        return {
            "error": f"No se pudo obtener el tiempo real: {str(e)}",
            "location": location,
            "date": date,
        }

    except Exception as e:
        logger.error(f"Error inesperado en get_weather: {e}")
        return {
            "error": f"Error inesperado: {str(e)}",
            "location": location,
            "date": date,
        }


# ----------------------------------------------------------------------
# SCHEMA DE WEATHER  ← IMPORTANTE
# ----------------------------------------------------------------------
weather_schema = {
    "name": "get_weather",
    "description": "Obtiene la predicción del tiempo usando OpenWeatherMap",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["location"],
    },
}


# ----------------------------------------------------------------------
# 2. HERRAMIENTA: Paradas de guagua
# ----------------------------------------------------------------------
def get_bus_stops(location: str) -> Dict[str, Any]:
    api_url = os.getenv("BUS_API_URL")

    if api_url:
        try:
            logger.info(f"Consultando paradas de guagua para {location}")

            response = requests.get(
                api_url,
                params={"city": location},
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "location": location,
                "stops": data.get("stops", []),
                "source": "API",
            }

        except RequestException as e:
            logger.error(f"Error en API de BUS: {e}")
            return _get_mock_bus_stops(location)
        except Exception as e:
            logger.error(f"Error inesperado en API de BUS: {e}")
            return {"error": str(e), "location": location, "stops": []}

    return _get_mock_bus_stops(location)


def _get_mock_bus_stops(location: str) -> Dict[str, Any]:
    mock_data = {
        "Santa Cruz": [
            "Intercambiador Santa Cruz",
            "Plaza de España",
            "Rambla de Pulido",
            "Parque García Sanabria",
            "Calle Castillo",
        ],
        "La Laguna": [
            "Aguere",
            "Padre Anchieta",
            "La Trinidad",
            "San Agustín",
            "Campamento Alto",
        ],
        "Adeje": [
            "Costa Adeje",
            "Torviscas",
            "Fañabé",
            "Playa Paraiso",
            "Marazul",
        ],
        "Puerto de la Cruz": [
            "Plaza del Charco",
            "Calle San Telmo",
            "Avenida de Colón",
            "La Paz",
            "Martín Alonso Pinzón",
        ],
    }

    return {
        "location": location,
        "stops": mock_data.get(location, ["Parada central", "Parada secundaria"]),
        "source": "MOCK",
    }


bus_schema = {
    "name": "get_bus_stops",
    "description": "Obtiene paradas de guagua cercanas",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


# ----------------------------------------------------------------------
# 3. HERRAMIENTA: Ofertas gastronómicas
# ----------------------------------------------------------------------
def get_restaurant_offers(location: str) -> Dict[str, Any]:
    api_url = os.getenv("RESTAURANT_API_URL")

    if api_url:
        try:
            logger.info(f"Consultando ofertas gastronómicas para {location}")

            response = requests.get(
                api_url,
                params={"city": location},
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "location": location,
                "offers": data.get("offers", []),
                "source": "API",
            }

        except RequestException as e:
            logger.error(f"Error en API de RESTAURANT: {e}")
            return _get_mock_restaurant_offers(location)
        except Exception as e:
            logger.error(f"Error inesperado en API de RESTAURANT: {e}")
            return {"error": str(e), "location": location, "offers": []}

    return _get_mock_restaurant_offers(location)


def _get_mock_restaurant_offers(location: str) -> Dict[str, Any]:
    mock_offers = {
        "Santa Cruz": [
            {"nombre": "Guachinche El Abuelo", "oferta": "Menú €12", "descripcion": "Guachinche canario"},
            {"nombre": "La Hierbita", "oferta": "10% descuento", "descripcion": "Cocina tradicional"},
        ],
        "Adeje": [
            {"nombre": "El Gomero", "oferta": "Postre gratis", "descripcion": "Cocina canaria"},
            {"nombre": "La Cueva", "oferta": "2x1 en tapas", "descripcion": "Ambiente acogedor"},
        ],
    }

    return {
        "location": location,
        "offers": mock_offers.get(location, [{"nombre": "Restaurante local", "oferta": "Sin ofertas"}]),
        "source": "MOCK",
    }


restaurant_schema = {
    "name": "get_restaurant_offers",
    "description": "Devuelve ofertas gastronómicas",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


# ----------------------------------------------------------------------
# 4. REGISTRO DE HERRAMIENTAS
# ----------------------------------------------------------------------
TOOLS = {
    "get_weather": get_weather,
    "get_bus_stops": get_bus_stops,
    "get_restaurant_offers": get_restaurant_offers,
}

TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": weather_schema,
    },
    {
        "type": "function",
        "function": bus_schema,
    },
    {
        "type": "function",
        "function": restaurant_schema,
    },
]


# ----------------------------------------------------------------------
# 5. CLI PARA PRUEBAS
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probar herramientas del asistente turístico")

    parser.add_argument("tool", choices=["weather", "bus", "restaurant"])
    parser.add_argument("--params", type=str)

    args = parser.parse_args()

    if args.params:
        params = json.loads(args.params)
    else:
        params = {"location": "Santa Cruz", "date": "2024-06-15"}

    if args.tool == "weather":
        print(get_weather(**params))
    elif args.tool == "bus":
        print(get_bus_stops(**params))
    elif args.tool == "restaurant":
        print(get_restaurant_offers(**params))
