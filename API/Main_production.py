from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import tempfile
import unicodedata

# Importaciones de lógica de negocio
from model_production import predecir_temperatura
from qa_production import GeminiAssistant

# Importaciones de carga de modelos PyTorch
import torch
from joblib import load

# Importar arquitectura del modelo desde archivo separado
from model_architecture import ProductionLSTMTransformerModel

# --- Configuración de rutas locales en EC2 ---
MODEL_PATH = "./production_weather_model.pth"  # Ruta local en EC2
SCALER_X_PATH = "./scaler_X_production.joblib"
SCALER_Y_PATH = "./scaler_y_production.joblib"
LABEL_ENCODER_PATH = "./label_encoder_global_all_stations.joblib"

# --- Variables Globales ---
# Modelos para el pronóstico
model = None
label_encoder = None
scaler_X = None
scaler_y = None

# Instancia del asistente de preguntas
assistant = GeminiAssistant()

# --- Función de Lifespan para cargar recursos al inicio ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Función de lifespan para FastAPI.
    Se ejecuta al iniciar la aplicación para cargar los modelos desde EC2.
    """
    global model, label_encoder, scaler_X, scaler_y
    print("Iniciando carga de modelo PyTorch y scalers desde archivos locales...")

    try:
        # Configurar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {device}")

        # Verificar que los archivos existen
        required_files = {
            'model': MODEL_PATH,
            'scaler_X': SCALER_X_PATH,
            'scaler_y': SCALER_Y_PATH,
            'label_encoder': LABEL_ENCODER_PATH
        }

        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Archivo requerido no encontrado: {file_path}")
            print(f"✓ Archivo encontrado: {file_path}")

        # Cargar el modelo PyTorch
        print("Cargando checkpoint PyTorch...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Extraer configuración del modelo
        model_config = checkpoint['config']
        print("Configuración del modelo extraída del checkpoint.")
        
        # Crear el modelo con la configuración guardada
        model = ProductionLSTMTransformerModel(
            d_input=model_config['n_features'],
            lstm_hidden_size=model_config['LSTM_HIDDEN_SIZE'],
            lstm_layers=model_config['LSTM_LAYERS'],
            n_transformer_layers=model_config['NUM_LAYERS'],
            d_model=model_config['D_MODEL'],
            n_heads=model_config['NUM_HEAD'],
            dff=model_config['DFF'],
            max_len=model_config['T'],
            dropout_rate=model_config['DROPOUT_RATE'],
        ).to(device)
        
        # Cargar los pesos del modelo
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Modo evaluación para inferencia
        print("Modelo PyTorch cargado y configurado exitosamente.")

        # Cargar scalers y label encoder
        label_encoder = load(LABEL_ENCODER_PATH)
        print("LabelEncoder cargado exitosamente.")
        
        scaler_X = load(SCALER_X_PATH)
        print("Scaler_X cargado exitosamente.")
        
        scaler_y = load(SCALER_Y_PATH)
        print("Scaler_y cargado exitosamente.")

        print("Todos los recursos de ML cargados exitosamente. API lista para usar!")
        
    except Exception as e:
        print(f" ERROR FATAL: No se pudieron cargar los recursos. Error: {e}")
        print(f"   Verifica que los archivos del modelo estén en el directorio actual:")
        print(f"   - {MODEL_PATH}")
        print(f"   - {SCALER_X_PATH}")
        print(f"   - {SCALER_Y_PATH}")
        print(f"   - {LABEL_ENCODER_PATH}")
        raise RuntimeError(f"Fallo al cargar modelos: {e}")

    yield  # La aplicación está lista
    
    print("Cerrando la aplicación y liberando recursos.")


# --- Inicialización de la App FastAPI ---
app = FastAPI(title="API de Predicción y Preguntas", lifespan=lifespan)

# --- Modelos Pydantic para las solicitudes ---
class ForecastRequest(BaseModel):
    provincia: str
    dias_a_predecir: int

class AskRequest(BaseModel):
    pregunta: str


# --- Endpoints de la API ---
@app.get("/")
async def root():
    """Endpoint raíz - información básica de la API."""
    return {
        "mensaje": "API de Predicción Meteorológica y Consultas IA",
        "version": "2.0",
        "endpoints": {
            "prediccion": "/predecir",
            "conversacion": "/conversacion", 
            "legacy": "/forecast",
            "qa": "/ask"
        },
        "documentacion": "/docs",
        "estado": "Operativo"
    }

@app.post("/ask")
async def ask(request: AskRequest):
    """Endpoint para hacer preguntas en lenguaje natural."""
    try:
        pregunta = request.pregunta
        print(f"Recibida pregunta: '{pregunta}'")

        # --- PRE-PROCESAMIENTO: Quitar tildes, caracteres especiales y pasar a mayúsculas ---
        pregunta_procesada = unicodedata.normalize('NFKD', pregunta).encode('ascii', 'ignore').decode('utf-8').upper()
        
        
        # 1. Generar SQL desde la pregunta procesada
        sql = assistant.generar_sql_desde_pregunta(pregunta_procesada)

        # 2. Ejecutar la consulta
        datos = assistant.ejecutar_sql(sql)

        # 3. Verificar errores en los datos
        if isinstance(datos, dict) and "error" in datos:
            # Devuelve el error específico para facilitar la depuración
            raise HTTPException(status_code=400, detail=f"Error al ejecutar SQL: {datos['error']}")

        # 4. Generar respuesta en lenguaje natural
        respuesta = assistant.responder_pregunta(pregunta_procesada, datos)
        return {"respuesta": respuesta, "sql_generada": sql}
    except Exception as e:
        # Captura cualquier otro error inesperado
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno: {e}")


@app.post("/predecir")
async def predecir(request: ForecastRequest):
    """Endpoint para generar predicciones de temperatura."""
    try:
        print(f"Solicitud de predicción para provincia: {request.provincia}, días: {request.dias_a_predecir}")
        
        # Llamar a la función de predicción con datos reales
        resultado_prediccion = predecir_temperatura(
            provincia=request.provincia,
            dias_a_predecir=request.dias_a_predecir,
            scaler_X=scaler_X, 
            scaler_y=scaler_y, 
            label_encoder=label_encoder,
            model=model
        )

        return resultado_prediccion
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")

@app.post("/conversacion")
async def conversacion(request: AskRequest):
    """Endpoint para consultas conversacionales usando Gemini AI."""
    try:
        print(f"Consulta conversacional: {request.pregunta}")
        
        # Usar el asistente Gemini para responder
        respuesta = assistant.generar_respuesta_desde_pregunta(request.pregunta)
        
        return {"respuesta": respuesta}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en consulta conversacional: {e}")

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    """Endpoint legacy para compatibilidad (redirige a /predecir)."""
    return await predecir(request)
    




