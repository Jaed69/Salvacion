# config/settings.py
# Configuración principal del proyecto LSP Esperanza

from pathlib import Path

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" 
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

# Configuración de datos
SEQUENCE_LENGTH = 50
MAX_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
FEATURES_PER_FRAME = MAX_HANDS * LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 126

# Configuración de modelo
MODEL_CONFIG = {
    'bidirectional_dynamic': {
        'name': 'sign_model_bidirectional_dynamic.h5',
        'sequence_input_shape': (SEQUENCE_LENGTH, FEATURES_PER_FRAME),
        'motion_features_count': 14,  # 6 básicas + 8 dinámicas avanzadas
        'prediction_threshold': 0.8
    }
}

# Configuración de señas
STATIC_SIGNS = {
    'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
}

DYNAMIC_SIGNS = {
    'J', 'Z', 'HOLA', 'GRACIAS', 'POR FAVOR'
}

# Configuración de cámara
CAMERA_CONFIG = {
    'width': 1280,
    'height': 720,
    'fps': 30,
    'device_id': 0
}

# Configuración de MediaPipe
MEDIAPIPE_CONFIG = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.8,
    'min_tracking_confidence': 0.7
}

# Configuración de movimiento
MOVEMENT_CONFIG = {
    'buffer_size': 20,
    'stability_buffer_size': 15,
    'movement_threshold': 0.02,
    'prediction_timeout': 2.0,
    'no_hands_timeout': 3.0
}

# Configuración de augmentación de datos
AUGMENTATION_CONFIG = {
    'max_augmentations_per_sign': 50,
    'quality_threshold': 0.5,
    'noise_factor': 0.1,
    'time_warp_factor': 0.2,
    'rotation_range': 15,  # grados
    'scale_range': (0.8, 1.2)
}

# Configuración de UI
UI_COLORS = {
    'primary': (64, 128, 255),
    'success': (46, 204, 113),
    'warning': (255, 193, 7),
    'danger': (231, 76, 60),
    'dark': (52, 73, 94),
    'light': (236, 240, 241),
    'background': (44, 62, 80),
    'text': (255, 255, 255),
    'accent': (155, 89, 182),
    'static': (52, 152, 219),
    'dynamic': (230, 126, 34)
}

# Configuración de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': REPORTS_DIR / 'lsp_esperanza.log'
}

# Asegurar que los directorios existen
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)
