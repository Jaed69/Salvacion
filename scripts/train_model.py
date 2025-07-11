#!/usr/bin/env python3
"""
Script para entrenar modelos de lenguaje de señas
Proyecto LSP Esperanza
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description='Entrenador de modelos de lenguaje de señas',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-type',
        choices=['basic', 'bidirectional_dynamic'],
        default='bidirectional_dynamic',
        help='Tipo de modelo a entrenar (default: %(default)s)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número de épocas de entrenamiento (default: %(default)s)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño de batch (default: %(default)s)'
    )
    
    parser.add_argument(
        '--data-path',
        default='data/sequences',
        help='Ruta a los datos de entrenamiento (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    print("🎯 Iniciando entrenamiento de modelo LSP Esperanza")
    print("=" * 50)
    print(f"🔧 Tipo de modelo: {args.model_type}")
    print(f"📊 Épocas: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📁 Datos: {args.data_path}")
    print("=" * 50)
    
    try:
        from models.trainer import EnhancedModelTrainer
        
        trainer = EnhancedModelTrainer(data_path=args.data_path)
        
        if args.model_type == 'bidirectional_dynamic':
            trainer.train_bidirectional_dynamic_model(
                epochs=args.epochs,
                batch_size=args.batch_size
            )
        else:
            print(f"❌ Tipo de modelo '{args.model_type}' no implementado aún")
            return 1
            
        print("✅ Entrenamiento completado exitosamente")
        return 0
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
