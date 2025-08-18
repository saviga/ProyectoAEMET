import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv


load_dotenv()
AEMET_API_KEY = os.getenv("AEMET_API_KEY")
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"

def extractorAEMET(municipio_id: str = "28079"): 
    if not AEMET_API_KEY:
        print("Error: AEMET_API_KEY no encontrada en las variables de entorno.")
        return
    endpoint_url = f"{AEMET_BASE_URL}/prediccion/especifica/municipio/diaria/{municipio_id}"
    print(f"Realizando solicitud inicial a {endpoint_url} para municipio {municipio_id}...")
    try:
        response_inicial = requests.get(
            endpoint_url,
            params={'api_key': AEMET_API_KEY},
            headers={'Cache-Control': 'no-cache'} #RECOMENDACION IA Puede ser útil para evitar cachés antiguos
        )
        response_inicial.raise_for_status() 
        data_inicial = response_inicial.json()
        if response_inicial.status_code == 200:
            print(f"Respuesta inicial recibida con éxito. Código de estado: {response_inicial.status_code}")
            print(f"Estado de la respuesta de AEMET: {data_inicial.get('estado')}")
            print(f"Descripción: {data_inicial.get('descripcion')}")
        else:
            print(f"Error en la solicitud inicial. Código de estado: {response_inicial.status_code}")
            print(f"Detalles del error: {data_inicial.get('descripcion')}")
            return
        datos_url = data_inicial.get('datos')
        if not datos_url:
            print("Error: No se encontró la URL de los datos reales en la respuesta inicial.")
            return
        print(f"Obteniendo datos reales desde: {datos_url}")
        time.sleep(0.5) 
        response_datos = requests.get(datos_url)
        response_datos.raise_for_status()
        all_data = response_datos.json()
        print("Datos recopilados con éxito.")
        if isinstance(all_data, list) and len(all_data) > 0:
            df = pd.DataFrame([all_data[0]]) 
        else:
            print("La respuesta de datos no tiene el formato esperado (lista no vacía).")
            return
        df.to_json(f'../../datos_aemet_municipio_{municipio_id}.json', orient='records', indent=4)
        print(f"Proceso terminado. Datos guardados en 'datos_aemet_municipio_{municipio_id}.json'.")
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión o HTTP: {e}")
    except ValueError as e:
        print(f"Error al parsear la respuesta JSON: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


# PARA EJECUTAR EXTRACTOR
# Si es es un .py acordarse de From Extractor import extractorAEMET

extractorAEMET("28079")