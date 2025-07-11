#!/usr/bin/env python3
"""
Script principal para ejecutar el traductor de lenguaje de señas en tiempo real
Proyecto LSP Esperanza
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from translation.real_time_translator import BidirectionalRealTimeTranslator
from config.settings import MODEL_CONFIG, MODELS_DIR

def main():
    parser = argparse.ArgumentParser(
        description='Traductor bidireccional de lenguaje de señas en tiempo real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                    # Ejecutar con configuración por defecto
  python main.py --threshold 0.9                  # Usar umbral de confianza más alto
  python main.py --model models/mi_modelo.h5       # Usar modelo personalizado
        """
    )
    
    parser.add_argument(
        '--model', 
        default=str(MODELS_DIR / MODEL_CONFIG['bidirectional_dynamic']['name']),
        help='Ruta al modelo bidireccional (default: %(default)s)'
    )
    
    parser.add_argument(
        '--labels', 
        default=str(MODELS_DIR / 'label_encoder.npy'),
        help='Ruta al archivo de etiquetas (default: %(default)s)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=MODEL_CONFIG['bidirectional_dynamic']['prediction_threshold'],
        help='Umbral de confianza para predicciones (default: %(default)s)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada de debug'
    )
    
    args = parser.parse_args()
    
    print("🚀 Iniciando LSP Esperanza - Traductor de Lenguaje de Señas")
    print("=" * 60)
    print(f"📁 Modelo: {args.model}")
    print(f"🏷️  Etiquetas: {args.labels}")
    print(f"🎯 Umbral: {args.threshold}")
    print("=" * 60)
    
    try:
        # Crear y configurar el traductor
        translator = BidirectionalRealTimeTranslator(
            model_path=args.model,
            signs_path=args.labels
        )
        
        # Configurar umbral
        translator.prediction_threshold = args.threshold
        
        # Ejecutar traductor
        translator.run()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Soluciones posibles:")
        print("   1. Entrena el modelo: python scripts/train_model.py")
        print("   2. Verifica las rutas de archivos")
        return 1
        
    except KeyboardInterrupt:
        print("\n👋 Traductor detenido por el usuario")
        return 0
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
