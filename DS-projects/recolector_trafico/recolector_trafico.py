import requests
import os
import time
import pandas as pd
from datetime import datetime

CSV_FILE = "datos_trafico_bariloche.csv"
API_KEY = os.environ.get("TOMTOM_API_KEY", "YQXoHyXYemCxSoIkyS6XC2RuoXSVpNrz")

tramos = {
    "Bustillo (km 1 a 8)":{
        "doble_sentido": True,
        "extremo_a": (-41.1165, -71.4050),
        "extremo_b": (-41.1338, -71.3255 ),
        "sentido_a_b": "hacia el centro",
        "sentido_b_a": "hacia km 8"
    },
        "Pioneros (km 1 a 8)":{
        "doble_sentido": True,
        "extremo_a": (-41.12113475534983, -71.40542824129189),
        "extremo_b": (-41.13974750818536, -71.31530250494299),
        "sentido_a_b": "hacia el centro",
        "sentido_b_a": "hacia km 8"
        },

    "Onelli":{
        "doble_sentido": False,
        "extremo_a": (-41.1359282250399, -71.29720119285051),
        "extremo_b": (-41.14821053828844, -71.29939594714604),
        "sentido_a_b": "hacia el alto",
        #"sentido_b_a": "hacia el alto"
        },

    "Elordi":{
        "doble_sentido": False,
        "extremo_a": (-41.14923089128537, -71.29828585418639),
        "extremo_b": (-41.13610797379817, -71.29577531918925),
        "sentido_a_b": "hacia el centro",
        #"sentido_b_a": "hacia el alto"
        },

    "Av 12 de octubre":{
        "doble_sentido": True,
        "extremo_a": (-41.13265235565798, -71.31026993677759),
        "extremo_b": (-41.13280126536771, -71.28451954369174),
        "sentido_a_b": "hacia E",
        "sentido_b_a": "hacia W (hasta rotonda Ñireco)"
        },

    "Esandi":{
        "doble_sentido": True,
        "extremo_a": (-41.13287753811387, -71.28396257664328),
        "extremo_b": (-41.15835867135046, -71.26219019574278),
        "sentido_a_b": "hacia ruta",
        "sentido_b_a": "hacia rotonda ñireco)"
        },

    "2do sombra":{
        "doble_sentido": True,
        "extremo_a": (-41.13498418610558, -71.2886214438174),
        "extremo_b": (-41.133918040226575, -71.28545289403938),
        "sentido_a_b": "hacia abajo",
        "sentido_b_a": "hacia arriba)"
        },    

    "Camino aeropuerto":{
        "doble_sentido": True,
        "extremo_a": (-41.11991519119234, -71.22121032390473),
        "extremo_b": (-41.144211196413174, -71.16959927447431),
        "sentido_a_b": "hacia aeropuerto",
        "sentido_b_a": "hacia ciudad)"
        }, 


}

def consultar_tramo(nombre:str, sentido:str, origen:tuple, destino:tuple, api_key:str) -> dict | None:
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origen[0]},{origen[1]}:{destino[0]},{destino[1]}/json"
    params = {"key": api_key, "traffic": "true", "departAt": "now"}


    try:
        resp = requests.get(url, params=params, timeout=10)
        time.sleep(1)  # Pausa obligatoria para respetar el limite de TomTom
        resp.raise_for_status()
        summary = resp.json()["routes"][0]["summary"]

        tiempo_real_seg = summary.get("travelTimeInSeconds", 0)
        tiempo_base_seg = summary.get("noTrafficTravelTimeInSeconds", tiempo_real_seg)
        demora_seg = summary.get("trafficDelayInSeconds", 0)
        distancia_m = summary.get("lengthInMeters", 0)

        #convertiones
        distancia_km = distancia_m/1000.0
        tiempo_real_min = tiempo_real_seg/60.0
        tiempo_base_min = tiempo_base_seg/60.0
        demora_min = demora_seg/60.0

        velocidad_kmh = (distancia_km/(tiempo_real_min/60)) if tiempo_real_min > 0 else 0.0
        indice_congestion = (tiempo_real_seg/tiempo_base_seg) if tiempo_base_seg > 0 else 1.0

        return{
            "timestamp_utc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "arteria": nombre,
            "sentido": sentido,
            "distancia_km": round(distancia_km, 2),
            "tiempo_real_min": round(tiempo_real_min, 1),
            "tiempo_base_min": round(tiempo_base_min, 1),
            "demora_min": round(demora_min, 1),
            "velocidad_kmh": round(velocidad_kmh, 1),
            "indice_congestion": round(indice_congestion, 2)
        }
    except Exception as e:
        print(f"Error consultando {nombre}: {e}")
        return None


registros = []


for nombre, cfg in tramos.items():
    #Consulta sentido A -> B (aplica para todas las arterias)
    d_ab = consultar_tramo(nombre, cfg["sentido_a_b"], cfg["extremo_a"], cfg["extremo_b"], API_KEY)
    if d_ab:
        registros.append(d_ab)
    
    #Sentido B->A - Solo si la calle es doble sentido
    if cfg.get("doble_sentido", False):
        d_ba = consultar_tramo(nombre, cfg["sentido_b_a"], cfg["extremo_b"], cfg["extremo_a"], API_KEY)
        if d_ba:
            registros.append(d_ba)
    


if registros:
    df_nuevo = pd.DataFrame(registros)

    columnas = [
        "timestamp_utc",
        "arteria",
        "sentido",
        "distancia_km",
        "tiempo_real_min",
        "tiempo_base_min",
        "demora_min",
        "velocidad_kmh",
        "indice_congestion"
    ]
    df_nuevo = df_nuevo[columnas]
    CSV_FILE = "datos_trafico_bariloche.csv"
    archivo_existe = os.path.exists(CSV_FILE)
    df_nuevo.to_csv(CSV_FILE, mode="a", index=False, header=not archivo_existe)
    print("registros guardados")
