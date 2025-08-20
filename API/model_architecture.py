"""
Arquitectura del modelo LSTM-Transformer para predicción meteorológica.
Contiene todas las clases del modelo neural para evitar duplicación de código.
"""

import torch
import torch.nn as nn
import math

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
    """
    Modelo híbrido LSTM-Transformer para predicción meteorológica.
    Combina LSTM bidireccional con Transformer para capturar patrones temporales complejos.
    """
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
        
        # Transformer dimension
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

        # Transformer dimension
        x = self.lstm_to_transformer(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Output processing
        x = self.output_norm(x)
        x = x[:, -7:, :].mean(dim=1)  # Usar últimos 7 días

        # Prediction
        main_output = self.output_layers(x)
        
        # Combine with skip connection
        output = main_output + 0.1 * skip
        
        return output
