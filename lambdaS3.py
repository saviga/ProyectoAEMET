import requests
import os
import json
import time
import boto3
from datetime import datetime, timedelta

# Cargar las variables de entorno desde el archivo .env
AEMET_API_KEY = os.environ.get('AEMET_API_KEY')
AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"

# Configuración del bucket de S3
S3_BUCKET_NAME = "<nombre_bucket_S3>"
S3_REGION = "<region_S3>"
S3_OUTPUT_DIRECTORY = "<nombre_directorio_datos_obtenidos>"
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_to_s3(file_path, s3_key):
    try:
        s3.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        print(f"Archivo subido exitosamente a S3: s3://{S3_BUCKET_NAME}/{s3_key}")
    except Exception as e:
        print(f"Error al subir el archivo a S3: {e}")


def extractorAEMET_Hoy():
    if not AEMET_API_KEY:
        print("Error: AEMET_API_KEY no encontrada. Por favor, asegúrate de que esté configurada.")
        return []
    
    # Fecha de hoy menos 5 días
    fecha_actual = datetime.now() - timedelta(days=5)
    fecha_real = fecha_actual.strftime("%Y-%m-%d")
    fecha_ini_api_format = f"{fecha_real}T00:00:00UTC"
    fecha_fin_api_format = f"{fecha_real}T23:59:59UTC"
    endpoint_url = (
        f"{AEMET_BASE_URL}/valores/climatologicos/diarios/datos/"
        f"fechaini/{fecha_ini_api_format}/fechafin/{fecha_fin_api_format}/todasestaciones"
    )
    print(f"  Solicitando datos para el día de hoy: {fecha_real}")
    
    try:
        response_inicial = requests.get(
            endpoint_url,
            params={'api_key': AEMET_API_KEY},
            headers={'Cache-Control': 'no-cache'},
            timeout=60
        )
        response_inicial.raise_for_status()
        data_inicial = response_inicial.json()
        if response_inicial.status_code != 200 or 'datos' not in data_inicial:
            print(f"  Error en la solicitud inicial o no hay datos disponibles. Código: {response_inicial.status_code}. Detalles: {data_inicial.get('descripcion', 'N/A')}")
            return []
        datos_url = data_inicial.get('datos')
        if not datos_url:
            print(f"  Error: No se encontró la URL de los datos reales en la respuesta inicial para {fecha_real}.")
            print(f"  Respuesta inicial de AEMET: {data_inicial}")
            return []
        time.sleep(1.0)
        response_datos = requests.get(datos_url)
        response_datos.raise_for_status()
        batch_data = response_datos.json()

        if isinstance(batch_data, list) and batch_data:
            print(f"  Datos recibidos para el día de hoy: {len(batch_data)} registros.")
            
            # Guardar los datos en un archivo JSON
            output_filename = f"{fecha_real}.json"
            
            # Subir el archivo a S3
            s3_key = f"{S3_OUTPUT_DIRECTORY}/{output_filename}"
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=json.dumps(batch_data, indent=2, ensure_ascii=False),
                ContentType='application/json'
            )

            

            return batch_data
        else:
            print(f"  La respuesta de datos para el día de hoy no contiene una lista o está vacía.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"  Error de conexión o HTTP para el día de hoy: {e}")
        return []
    except ValueError as e:
        print(f"  Error al parsear la respuesta JSON para el día de hoy: {e}")
        return []
    except Exception as e:
        print(f"  Ocurrió un error inesperado durante la extracción de los datos: {e}")
        return []

def lambda_handler(event, context):
    extractorAEMET_Hoy()