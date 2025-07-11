#!/usr/bin/env python3
"""
Script de prueba rápida para el traductor estático
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from translation.static_real_time_translator import StaticRealTimeTranslator
    
    print("🎯 LSP ESPERANZA - Prueba del Traductor Estático")
    print("="*50)
    print("🚀 Iniciando traductor...")
    print("\n📋 INSTRUCCIONES:")
    print("   - Presiona 'q' para salir")
    print("   - Mantén la seña estable por 2-3 segundos")
    print("   - Evita movimientos bruscos")
    print("\n✨ ¡Disfruta traduciendo!")
    print("-" * 50)
    
    # Crear e iniciar el traductor con configuración básica
    translator = StaticRealTimeTranslator(
        confidence_threshold=0.85,
        camera_index=0,
        show_landmarks=True
    )
    
    translator.run()
    
except KeyboardInterrupt:
    print("\n\n👋 Saliendo del traductor...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Sugerencias:")
    print("   - Verifica que el modelo estático esté entrenado")
    print("   - Ejecuta: python scripts/train_static_model.py")
    print("   - Asegúrate de que la cámara esté conectada")
