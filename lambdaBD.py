import json
import boto3
import psycopg
import pandas as pd
import numpy as np
import os
import unicodedata

def lambda_handler(event, context):

    def limpiar_y_procesar_dataframe(df):
        """
        Función centralizada para limpiar, normalizar, convertir tipos de datos
        e interpolar los datos del DataFrame.
        """
        # --- 1. Definición de columnas ---
        numeric_cols = [
            'altitud', 'tmed', 'prec', 'tmin', 'tmax',
            'hrMax', 'hrMin', 'hrMedia'
        ]
        text_cols = [
            'indicativo', 'nombre', 'provincia', 'horatmin', 'horatmax',
            'horaHrMax', 'horaHrMin'
        ]
        columnas_a_eliminar = [
            'presMax', 'horaPresMax', 'presMin', 'horaPresMin',
            'sol', 'dir', 'velmedia', 'racha', 'horaracha'
        ]

        # 2. Eliminar columnas innecesarias
        df.drop(columns=columnas_a_eliminar, inplace=True, errors='ignore')

        # 3. Procesar la columna 'fecha'
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce').dt.date

        # 4. Limpiar columnas numéricas
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = df[col].replace(['', ' ', 'null', 'None', '-'], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        

        # 5. Normalizar texto
        def normalizar_texto(valor):
            if pd.isnull(valor):
                return None
            return unicodedata.normalize('NFKD', str(valor)).encode('ASCII', 'ignore').decode('utf-8').strip()

        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(normalizar_texto)

        return df  # Aquí termina la función interna

    # --- A partir de aquí está el handler principal ---
    # 1. Obtiene bucket y key del evento S3
    bucket = event['Records'][0]['s3']['bucket']['name']
    file_name = event['Records'][0]['s3']['object']['key'].split("/")[-1]
    key = f"data/aemet-diarios/{file_name}"


    # 2. Lee archivo JSON desde S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read().decode('utf-8')
    raw_data = json.loads(file_content)

    # 3. Convierte a DataFrame
    df = pd.DataFrame(raw_data)

    # 4. Limpieza
    df_procesado = limpiar_y_procesar_dataframe(df)
    processed_data = df_procesado.to_dict(orient='records')

    # 5. Inserción en base de datos
    try:
        with psycopg.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get('RDS_PASS'),
            port=os.environ.get("DB_PORT")
        ) as conn:
            with conn.cursor() as cur:
                for row in processed_data:
                    cur.execute("""
                        INSERT INTO datos_clima (
                            indicativo, nombre, provincia, altitud, fecha, tmed, prec, tmin,
                            horatmin, tmax, horatmax, hrMax, horaHrMax, hrMin, horaHrMin, hrMedia
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (fecha, indicativo) DO UPDATE SET
                            nombre = EXCLUDED.nombre,
                            provincia = EXCLUDED.provincia,
                            altitud = EXCLUDED.altitud,
                            tmed = EXCLUDED.tmed,
                            prec = EXCLUDED.prec,
                            tmin = EXCLUDED.tmin,
                            horatmin = EXCLUDED.horatmin,
                            tmax = EXCLUDED.tmax,
                            horatmax = EXCLUDED.horatmax,
                            hrMax = EXCLUDED.hrMax,
                            horaHrMax = EXCLUDED.horaHrMax,
                            hrMin = EXCLUDED.hrMin,
                            horaHrMin = EXCLUDED.horaHrMin,
                            hrMedia = EXCLUDED.hrMedia;
                    """, (
                        row.get("indicativo"), row.get("nombre"), row.get("provincia"),
                        row.get("altitud"), row.get("fecha"), row.get("tmed"),
                        row.get("prec"), row.get("tmin"), row.get("horatmin"),
                        row.get("tmax"), row.get("horatmax"),
                        row.get("hrMax"), row.get("horaHrMax"),
                        row.get("hrMin"), row.get("horaHrMin"),
                        row.get("hrMedia")
                    ))
            conn.commit()

        return {
            'statusCode': 200,
            'body': json.dumps(f"{len(processed_data)} registros del archivo {key} procesados correctamente.")
        }
    except Exception as e:
        print(f"Error al procesar el archivo {key}: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error en el procesamiento: {e}")
        }