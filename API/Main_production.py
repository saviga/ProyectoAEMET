from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import tempfile
import unicodedata

# Importaciones de lógica de negocio
from model_production import generar_forecast 
from qa_production import GeminiAssistant

# Importaciones de carga de modelos PyTorch
import torch
import torch.nn as nn
import math
from joblib import load

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

# --- Definición del Modelo PyTorch (arquitectura del notebook) ---
class AdvancedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.mha(x, x, x, need_weights=False)
        return self.norm(x + self.dropout(attn_output))

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttentionLayer(d_model, n_heads, dropout_rate)
        self.feed_forward = FeedForwardLayer(d_model, dff, dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class ProductionLSTMTransformerModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        n_transformer_layers: int,
        d_model: int,
        n_heads: int,
        dff: int,
        max_len: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(d_input, lstm_hidden_size)
        
        # LSTM Stack
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=lstm_hidden_size if i == 0 else lstm_hidden_size * 2,
                hidden_size=lstm_hidden_size,
                batch_first=True,
                dropout=dropout_rate if i < lstm_layers - 1 else 0,
                bidirectional=True
            ) for i in range(lstm_layers)
        ])
        
        # Projection to transformer dimension
        self.lstm_to_transformer = nn.Linear(lstm_hidden_size * 2, d_model)
        
        # Transformer Stack
        self.pos_encoding = AdvancedPositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dff, dropout_rate)
            for _ in range(n_transformer_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, 1)
        )
        
        # Skip connection
        self.skip_connection = nn.Linear(d_input, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Skip connection
        skip = self.skip_connection(x[:, -1, :])
        
        # Input projection
        x = self.input_projection(x)
        
        # LSTM processing
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Project to transformer dimension
        x = self.lstm_to_transformer(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output processing
        x = self.output_norm(x)
        x = x[:, -7:, :].mean(dim=1)  # prediccion = 7 según el notebook
        
        # Main prediction
        main_output = self.output_layers(x)
        
        # Combine with skip connection
        output = main_output + 0.1 * skip
        
        return output

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
    ubicacion: str
    dias: int

class AskRequest(BaseModel):
    pregunta: str


# --- Endpoints de la API ---
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


@app.post("/forecast")
async def forecast(request: ForecastRequest):
    """Endpoint para generar un pronóstico del tiempo."""
    try:
        print(f"Solicitud de pronóstico para ubicación: {request.ubicacion}, días: {request.dias}")
        
        # --- PRE-PROCESAMIENTO: Quitar tildes, caracteres especiales y pasar a mayúsculas ---
        ubicacion_procesada = unicodedata.normalize('NFKD', request.ubicacion).encode('ascii', 'ignore').decode('utf-8').upper()
        

        # Llamar a la función de pronóstico con la ubicación procesada
        forecast_results = generar_forecast(
            ubicacion=ubicacion_procesada,
            dias_a_predecir=request.dias,
            model=model, 
            scaler_X=scaler_X, 
            scaler_y=scaler_y, 
            label_encoder=label_encoder
        )

        if isinstance(forecast_results, dict) and "error" in forecast_results:
            raise HTTPException(status_code=400, detail=forecast_results["error"])

        return {"ubicacion": ubicacion_procesada, "pronostico": forecast_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno en el pronóstico: {e}")
    




