#!/usr/bin/env python3
"""
Script de prueba mejorado para el traductor estático en tiempo real
Configuración optimizada para detectar señales con movimiento natural
"""

import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from translation.static_real_time_translator import StaticRealTimeTranslator
    
    print("🎯 LSP ESPERANZA - Traductor Estático Optimizado para Tiempo Real")
    print("="*60)
    print("🚀 Configuración mejorada para detectar señales naturales")
    print("\n📋 INSTRUCCIONES MEJORADAS:")
    print("   - Presiona 'q' para salir")
    print("   - Realiza las señas de forma natural (no necesitas estar 100% inmóvil)")
    print("   - Mantén la seña por 1-2 segundos para mejor reconocimiento")
    print("   - Las predicciones aparecerán en tiempo real")
    print("   - Confianza mínima: 40% (más permisivo)")
    print("\n✨ ¡Traducción en tiempo real mejorada!")
    print("-" * 60)
    
    # Crear e iniciar el traductor con configuración optimizada
    translator = StaticRealTimeTranslator(
        confidence_threshold=0.75,  # Umbral moderado
        camera_index=0,
        show_landmarks=True
    )
    
    print("\n🔧 Configuración aplicada:")
    print(f"   • Umbral de confianza: {translator.config['confidence_threshold']:.2f}")
    print(f"   • Frames de estabilidad: {translator.config['stability_frames']}")
    print(f"   • Cooldown entre predicciones: {translator.config['prediction_cooldown']}")
    print(f"   • Tolerancia al movimiento: {translator.config['movement_tolerance']:.2f}")
    print(f"   • Predicción continua: {'Activada' if translator.config['continuous_prediction'] else 'Desactivada'}")
    print(f"   • Confianza mínima para mostrar: {translator.config['min_prediction_confidence']:.2f}")
    print("\n🎬 Iniciando traductor...")
    
    translator.run()
    
except KeyboardInterrupt:
    print("\n\n👋 Saliendo del traductor...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Sugerencias:")
    print("   - Verifica que el modelo estático esté entrenado")
    print("   - Asegúrate de que la cámara esté conectada")
    print("   - Ejecuta el entrenamiento si es necesario:")
    print("     python scripts/train_static_model.py")
