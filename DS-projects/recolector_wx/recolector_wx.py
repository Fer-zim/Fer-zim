import os
import requests
import pandas as pd
from datetime import datetime

CSV_FILE = "datos_clima_bariloche.csv"

# Coordenadas (Bariloche)
LATITUD = -41.1335
LONGITUD = -71.3103

def obtener_clima(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "precipitation",
            "rain",
            "snowfall",
            "visibility",
            "weather_code",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("current", {})
    except Exception as e:
        print(f"Error consultando la API: {e}")
        return None

def registrar():
    actual = obtener_clima(LATITUD, LONGITUD)
    if not actual:
        print("No se obtuvieron datos climáticos en este intento.")
        return

    # Fila consolidada
    fila = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_local": actual.get("time"),
        "temperatura_c": actual.get("temperature_2m"),
        "precipitacion_mm": actual.get("precipitation"),
        "lluvia_mm": actual.get("rain"),
        "nieve_cm": actual.get("snowfall"),
        "visibilidad_m": actual.get("visibility"),
        "viento_kmh": actual.get("wind_speed_10m"),
        "codigo_wmo": actual.get("weather_code")
    }

    df_nuevo = pd.DataFrame([fila])

    # Si no existe el CSV escribe encabezado; si existe, solo añade la fila
    existe = os.path.exists(CSV_FILE)
    df_nuevo.to_csv(CSV_FILE, mode="a", index=False, header=not existe)
    print(f"Registro guardado con éxito: {fila['timestamp_local']} | Temp: {fila['temperatura_c']} °C")

if __name__ == "__main__":
    registrar()
