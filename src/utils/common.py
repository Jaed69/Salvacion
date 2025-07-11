"""
Utilidades comunes para el proyecto LSP Esperanza
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio config al path si no está
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import LOGGING_CONFIG, PROJECT_ROOT, REPORTS_DIR
except ImportError:
    # Fallback si no se puede importar la configuración
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    REPORTS_DIR = PROJECT_ROOT / "reports"
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': REPORTS_DIR / 'lsp_esperanza.log'
    }

def setup_logging(name: str = "LSP_Esperanza") -> logging.Logger:
    """
    Configura el sistema de logging para el proyecto
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    # Crear directorio de reportes si no existe
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Configurar el logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # Evitar duplicar handlers
    if not logger.handlers:
        # Handler para archivo
        file_handler = logging.FileHandler(LOGGING_CONFIG['file'])
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def ensure_directories():
    """
    Asegura que todos los directorios necesarios existan
    """
    directories = [
        PROJECT_ROOT / "data" / "sequences",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "docs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def save_report(data: dict, report_type: str = "session") -> str:
    """
    Guarda un reporte en formato JSON con timestamp
    
    Args:
        data: Datos del reporte
        report_type: Tipo de reporte (session, training, augmentation, etc.)
        
    Returns:
        Ruta del archivo guardado
    """
    ensure_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report_type}_report_{timestamp}.json"
    filepath = REPORTS_DIR / filename
    
    # Agregar metadata
    data['timestamp'] = datetime.now().isoformat()
    data['report_type'] = report_type
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)

def load_latest_report(report_type: str = "session") -> dict:
    """
    Carga el reporte más reciente de un tipo específico
    
    Args:
        report_type: Tipo de reporte a buscar
        
    Returns:
        Datos del reporte o diccionario vacío si no se encuentra
    """
    if not REPORTS_DIR.exists():
        return {}
    
    # Buscar archivos del tipo especificado
    pattern = f"{report_type}_report_*.json"
    report_files = list(REPORTS_DIR.glob(pattern))
    
    if not report_files:
        return {}
    
    # Ordenar por fecha de modificación y tomar el más reciente
    latest_file = max(report_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def validate_model_files() -> dict:
    """
    Valida que los archivos del modelo existan y sean accesibles
    
    Returns:
        Diccionario con el estado de validación
    """
    models_dir = PROJECT_ROOT / "models"
    
    required_files = {
        'model': models_dir / "sign_model_bidirectional_dynamic.h5",
        'labels': models_dir / "label_encoder.npy"
    }
    
    validation_result = {
        'valid': True,
        'missing_files': [],
        'existing_files': [],
        'models_dir_exists': models_dir.exists()
    }
    
    for file_type, filepath in required_files.items():
        if filepath.exists():
            validation_result['existing_files'].append({
                'type': file_type,
                'path': str(filepath),
                'size_mb': filepath.stat().st_size / (1024 * 1024)
            })
        else:
            validation_result['missing_files'].append({
                'type': file_type,
                'path': str(filepath)
            })
            validation_result['valid'] = False
    
    return validation_result

def print_system_info():
    """
    Imprime información del sistema y configuración
    """
    import platform
    import cv2
    import numpy as np
    
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = "No instalado"
    
    try:
        import mediapipe as mp
        mp_version = mp.__version__
    except ImportError:
        mp_version = "No instalado"
    
    print("🔍 INFORMACIÓN DEL SISTEMA LSP ESPERANZA")
    print("=" * 60)
    print(f"🖥️  SO: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"🧠 TensorFlow: {tf_version}")
    print(f"👁️  OpenCV: {cv2.__version__}")
    print(f"✋ MediaPipe: {mp_version}")
    print(f"🔢 NumPy: {np.__version__}")
    print("=" * 60)
    print(f"📁 Directorio del proyecto: {PROJECT_ROOT}")
    print(f"📊 Directorio de reportes: {REPORTS_DIR}")
    print("=" * 60)
    
    # Validar archivos del modelo
    validation = validate_model_files()
    if validation['valid']:
        print("✅ Todos los archivos del modelo están disponibles")
        for file_info in validation['existing_files']:
            print(f"   {file_info['type']}: {file_info['size_mb']:.2f} MB")
    else:
        print("⚠️  Archivos del modelo faltantes:")
        for file_info in validation['missing_files']:
            print(f"   ❌ {file_info['type']}: {file_info['path']}")

def get_project_stats() -> dict:
    """
    Obtiene estadísticas del proyecto
    
    Returns:
        Diccionario con estadísticas del proyecto
    """
    stats = {
        'data_sequences': 0,
        'models_count': 0,
        'reports_count': 0,
        'signs_available': []
    }
    
    # Contar secuencias de datos
    sequences_dir = PROJECT_ROOT / "data" / "sequences"
    if sequences_dir.exists():
        for sign_dir in sequences_dir.iterdir():
            if sign_dir.is_dir():
                sequence_files = list(sign_dir.glob("*.npy"))
                if sequence_files:
                    stats['signs_available'].append({
                        'sign': sign_dir.name,
                        'samples': len(sequence_files)
                    })
                    stats['data_sequences'] += len(sequence_files)
    
    # Contar modelos
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        stats['models_count'] = len(list(models_dir.glob("*.h5")))
    
    # Contar reportes
    if REPORTS_DIR.exists():
        stats['reports_count'] = len(list(REPORTS_DIR.glob("*.json")))
    
    return stats

if __name__ == "__main__":
    """Test de las utilidades"""
    print_system_info()
    
    print("\n📊 ESTADÍSTICAS DEL PROYECTO")
    print("=" * 60)
    stats = get_project_stats()
    print(f"📝 Secuencias de datos: {stats['data_sequences']}")
    print(f"🤖 Modelos: {stats['models_count']}")
    print(f"📋 Reportes: {stats['reports_count']}")
    print(f"✋ Señas disponibles: {len(stats['signs_available'])}")
    
    for sign_info in stats['signs_available']:
        print(f"   {sign_info['sign']}: {sign_info['samples']} muestras")
    
    # Test de logging
    logger = setup_logging("TEST")
    logger.info("Test de logging completado")
    
    print("\n✅ Test de utilidades completado")
