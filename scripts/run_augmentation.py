# run_augmentation.py
# Script ejecutable simplificado para data augmentation

import os
import sys

def main():
    """Ejecuta el motor de data augmentation con el entorno correcto"""
    print("🚀 EJECUTOR DE DATA AUGMENTATION")
    print("🎯 Generando datos para cumplir plan de mejora")
    print("=" * 50)
    
    # Comando para ejecutar con el entorno conda correcto
    conda_command = r"C:/ProgramData/miniconda3/Scripts/conda.exe run -p C:\Users\twofi\.conda\envs\LS --no-capture-output python"
    
    print("📋 Opciones disponibles:")
    print("1. Ejecutar demo con datos sintéticos")
    print("2. Ejecutar augmentation en dataset real")
    print("3. Solo análisis del dataset actual")
    
    try:
        choice = input("\nSelecciona una opción (1-3): ").strip()
        
        if choice == "1":
            print("\n🎬 Ejecutando demostración...")
            os.system(f'{conda_command} demo_augmentation.py')
            
        elif choice == "2":
            print("\n🏭 Ejecutando augmentation en dataset real...")
            os.system(f'{conda_command} data_augmentation_engine.py')
            
        elif choice == "3":
            print("\n📊 Análisis del dataset...")
            analysis_code = '''
import json
from data_augmentation_engine import SignDataAugmentationEngine

try:
    augmenter = SignDataAugmentationEngine()
    current_counts, deficits = augmenter.analyze_current_dataset()
    
    print("\\n📊 Estado actual del dataset:")
    for sign, count in current_counts.items():
        sign_type = augmenter.classify_sign_type(sign)
        print(f"   {sign} ({sign_type}): {count} secuencias")
    
    if deficits:
        print("\\n📋 Déficits según plan de mejora:")
        for sign, info in deficits.items():
            print(f"   {sign}: {info['current']}/{info['target']} (falta {info['deficit']})")
        
        total_needed = sum(info['deficit'] for info in deficits.values())
        print(f"\\n🎯 Total a generar: {total_needed} secuencias")
    else:
        print("\\n✅ ¡Dataset completo según el plan!")
        
except Exception as e:
    print(f"❌ Error: {e}")
'''
            
            # Escribir código temporal y ejecutar
            with open('temp_analysis.py', 'w', encoding='utf-8') as f:
                f.write(analysis_code)
            
            os.system(f'{conda_command} temp_analysis.py')
            
            # Limpiar archivo temporal
            if os.path.exists('temp_analysis.py'):
                os.remove('temp_analysis.py')
        
        else:
            print("❌ Opción no válida")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
