"""
tools.py — Herramientas externas para el Asistente Turístico de Tenerife
------------------------------------------------------------------------
Este módulo define las herramientas que el asistente puede llamar mediante
Function Calling de OpenAI.

Incluye:
- get_weather: API real (OpenWeatherMap)
- get_bus_stops: API real + fallback mock
- get_restaurant_offers: API real + fallback mock
- get_webcams: webcams oficiales de Tenerife
- get_bike_rentals: alquiler de bicicletas
- get_transport_info: rutas aproximadas TITSA/Moovit (opcional)
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

    NORMALIZED_LOCATIONS = {
        "tenerife": "Santa Cruz de Tenerife",
        "norte de tenerife": "Puerto de la Cruz",
        "sur de tenerife": "Adeje",
    }

    loc_key = location.lower().strip()
    if loc_key in NORMALIZED_LOCATIONS:
        location = NORMALIZED_LOCATIONS[loc_key]

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
# 4. NUEVAS TOOLS
# ----------------------------------------------------------------------

# ⭐ WEBCAMS
def get_webcams(location: str) -> Dict[str, Any]:
    location = location.lower()
    webcams = []

    if any(x in location for x in ["arona", "cristianos", "vistas", "américas"]):
        webcams.append({
            "name": "Playa de Las Vistas (Arona)",
            "url": "https://canariaslife.com/camaras-de-tenerife/arona/"
        })

    if "santa cruz" in location:
        webcams.append({
            "name": "Santa Cruz de Tenerife (Skyline)",
            "url": "https://www.skylinewebcams.com/es/webcam/espana/canarias/santa-cruz-de-tenerife.html"
        })

    if not webcams:
        webcams.append({
            "name": "Webcams de Tenerife",
            "url": "https://canariaslife.com/camaras-de-tenerife/"
        })

    return {"location": location, "webcams": webcams}


webcam_schema = {
    "name": "get_webcams",
    "description": "Devuelve enlaces a webcams en directo según la zona indicada",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


# ⭐ BICICLETAS
def get_bike_rentals(location: str) -> Dict[str, Any]:
    return {
        "location": location,
        "rentals": [
            {
                "name": "BikePoint Tenerife",
                "url": "https://bikepointtenerife.com/es/"
            },
            {
                "name": "Cycling in Tenerife",
                "url": "https://www.cyclingintenerife.com/es/alquiler-de-bicicletas-en-tenerife/"
            }
        ]
    }


bike_schema = {
    "name": "get_bike_rentals",
    "description": "Devuelve tiendas de alquiler de bicicletas en Tenerife",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


# ⭐ TRANSPORTE (opcional)
def get_transport_info(origin: str, destination: str) -> Dict[str, Any]:
    routes = [
        {
            "line": "TITSA 015",
            "from": origin,
            "to": destination,
            "duration": "35 min",
            "frequency": "Cada 20 min",
            "url": "https://www.titsa.com/"
        },
        {
            "line": "Moovit",
            "from": origin,
            "to": destination,
            "duration": "30–45 min",
            "frequency": "Variable",
            "url": "https://moovitapp.com/"
        }
    ]

    return {"origin": origin, "destination": destination, "routes": routes}


transport_schema = {
    "name": "get_transport_info",
    "description": "Devuelve rutas aproximadas de transporte público entre dos puntos",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["origin", "destination"],
    },
}


# ----------------------------------------------------------------------
# 5. REGISTRO DE HERRAMIENTAS
# ----------------------------------------------------------------------
TOOLS = {
    "get_weather": get_weather,
    "get_bus_stops": get_bus_stops,
    "get_restaurant_offers": get_restaurant_offers,
    "get_webcams": get_webcams,
    "get_bike_rentals": get_bike_rentals,
    # "get_transport_info": get_transport_info,  # activar si quieres
}


TOOLS_SCHEMAS = [
    {"type": "function", "function": weather_schema},
    {"type": "function", "function": bus_schema},
    {"type": "function", "function": restaurant_schema},
    {"type": "function", "function": webcam_schema},
    {"type": "function", "function": bike_schema},
    # {"type": "function", "function": transport_schema},  # activar si quieres
]


# ----------------------------------------------------------------------
# 6. CLI PARA PRUEBAS
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probar herramientas del asistente turístico")

    parser.add_argument("tool", choices=["weather", "bus", "restaurant", "webcams", "bikes"])
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
    elif args.tool == "webcams":
        print(get_webcams(**params))
    elif args.tool == "bikes":
        print(get_bike_rentals(**params))
