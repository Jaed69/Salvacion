#!/usr/bin/env python3
"""
Script principal para ejecutar el traductor de lenguaje de señas en tiempo real
Proyecto LSP Esperanza - Optimizado para señales estáticas
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from translation.real_time_translator import BidirectionalRealTimeTranslator
from translation.static_real_time_translator import StaticRealTimeTranslator
from config.settings import MODEL_CONFIG, MODELS_DIR

def main():
    parser = argparse.ArgumentParser(
        description='Traductor de lenguaje de señas en tiempo real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                    # Ejecutar modo estático (recomendado)
  python main.py --mode static                      # Modo estático explícito
  python main.py --mode dynamic                     # Modo dinámico (experimental)
  python main.py --threshold 0.9                   # Usar umbral de confianza más alto
  python main.py --model models/mi_modelo.keras     # Usar modelo personalizado
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['static', 'dynamic'],
        default='static',
        help='Modo de traducción: static (recomendado) o dynamic (experimental)'
    )
    
    parser.add_argument(
        '--model', 
        default=None,
        help='Ruta al modelo (se auto-detecta según el modo si no se especifica)'
    )
    
    parser.add_argument(
        '--labels', 
        default=None,
        help='Ruta al archivo de etiquetas (se auto-detecta según el modo si no se especifica)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=None,
        help='Umbral de confianza para predicciones (se ajusta automáticamente según el modo)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada de debug'
    )
    
    args = parser.parse_args()
    
    # Configuración automática según el modo
    if args.mode == 'static':
        # Configuración para modo estático
        model_path = args.model or str(MODELS_DIR / 'sign_model_static.keras')
        labels_path = args.labels or str(MODELS_DIR / 'label_encoder_static.npy')
        threshold = args.threshold or 0.85
        
        print("🎯 Iniciando LSP Esperanza - Traductor de Señales Estáticas")
        print("=" * 60)
        print("🌟 MODO ESTÁTICO - Optimizado para alta precisión")
        print("📋 Ventajas:")
        print("   • Alta precisión (>95%)")
        print("   • Baja latencia")
        print("   • Detección de estabilidad")
        print("   • Análisis geométrico avanzado")
        
    else:  # dynamic
        # Configuración para modo dinámico (experimental)
        model_path = args.model or str(MODELS_DIR / MODEL_CONFIG['bidirectional_dynamic']['name'])
        labels_path = args.labels or str(MODELS_DIR / 'label_encoder.npy')
        threshold = args.threshold or MODEL_CONFIG['bidirectional_dynamic']['prediction_threshold']
        
        print("⚡ Iniciando LSP Esperanza - Traductor de Señales Dinámicas")
        print("=" * 60)
        print("🧪 MODO DINÁMICO - Experimental")
        print("⚠️ Limitaciones conocidas:")
        print("   • Precisión limitada (~45%)")
        print("   • Sensible a variaciones de velocidad")
        print("   • Requiere secuencias perfectas")
    
    print(f"📁 Modelo: {model_path}")
    print(f"🏷️  Etiquetas: {labels_path}")
    print(f"🎯 Umbral: {threshold}")
    print("=" * 60)
    
    try:
        if args.mode == 'static':
            # Crear traductor estático
            translator = StaticRealTimeTranslator(
                model_path=model_path,
                labels_path=labels_path
            )
            translator.config['confidence_threshold'] = threshold
            
        else:  # dynamic
            # Crear traductor dinámico
            translator = BidirectionalRealTimeTranslator(
                model_path=model_path,
                signs_path=labels_path
            )
            translator.prediction_threshold = threshold
        
        # Ejecutar traductor
        translator.run()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Soluciones posibles:")
        if args.mode == 'static':
            print("   1. Entrena el modelo estático: python scripts/train_static_model.py")
        else:
            print("   1. Entrena el modelo dinámico: python scripts/train_model.py")
        print("   2. Verifica las rutas de archivos")
        print("   3. Usa el modo estático (recomendado): python main.py --mode static")
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
