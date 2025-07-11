#!/usr/bin/env python3
"""
Script para recolectar datos de lenguaje de señas
Proyecto LSP Esperanza
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description='Recolector de datos de lenguaje de señas'
    )
    
    parser.add_argument(
        '--sign',
        required=True,
        help='Seña a recolectar (ejemplo: A, B, J, HOLA)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Número de muestras a recolectar (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/sequences',
        help='Directorio de salida (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    print("📊 Iniciando recolección de datos LSP Esperanza")
    print("=" * 50)
    print(f"✋ Seña: {args.sign}")
    print(f"📝 Muestras: {args.samples}")
    print(f"📁 Directorio: {args.output_dir}")
    print("=" * 50)
    
    try:
        from data_processing.collector import DataCollector
        
        collector = DataCollector()
        collector.collect_data_for_sign(
            sign=args.sign,
            num_samples=args.samples,
            output_dir=args.output_dir
        )
        
        print("✅ Recolección completada exitosamente")
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Recolección detenida por el usuario")
        return 0
        
    except Exception as e:
        print(f"❌ Error durante la recolección: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
