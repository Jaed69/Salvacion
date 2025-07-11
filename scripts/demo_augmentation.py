# demo_augmentation.py
# Demostración del motor de data augmentation

import numpy as np
import os
import json
from data_augmentation_engine import SignDataAugmentationEngine
import matplotlib.pyplot as plt

def create_demo_data():
    """Crea datos de demostración para probar el sistema"""
    print("🛠️  Creando datos de demostración...")
    
    # Crear estructura de directorios
    base_path = "data/sequences"
    os.makedirs(base_path, exist_ok=True)
    
    # Crear datos sintéticos para algunas señas
    demo_signs = {
        'J': 'dynamic',    # Seña dinámica prioritaria
        'Z': 'dynamic',    # Seña dinámica prioritaria
        'A': 'static',     # Seña estática 
        'HOLA': 'phrase'   # Frase
    }
    
    for sign, sign_type in demo_signs.items():
        sign_path = os.path.join(base_path, sign)
        os.makedirs(sign_path, exist_ok=True)
        
        # Generar algunas secuencias base (simuladas)
        for i in range(5):  # Solo 5 secuencias base por seña
            if sign_type == 'static':
                # Seña estática: poco movimiento
                base_position = np.random.rand(126) * 0.1
                sequence = np.tile(base_position, (60, 1))
                sequence += np.random.normal(0, 0.001, sequence.shape)
                
            elif sign_type == 'dynamic':
                # Seña dinámica: movimiento claro
                sequence = np.zeros((60, 126))
                t = np.linspace(0, 2*np.pi, 60)
                
                # Simular movimiento sinusoidal en algunas coordenadas
                for coord in range(0, 126, 9):  # Cada 3er landmark
                    sequence[:, coord] = 0.1 * np.sin(t + np.random.rand())
                    sequence[:, coord+1] = 0.1 * np.cos(t + np.random.rand())
                
            else:  # phrase
                # Frase: combinación de movimientos
                sequence = np.zeros((60, 126))
                t = np.linspace(0, 4*np.pi, 60)
                
                # Movimiento más complejo
                for coord in range(0, 126, 6):
                    sequence[:, coord] = 0.05 * np.sin(t) + 0.03 * np.sin(3*t)
                    sequence[:, coord+1] = 0.05 * np.cos(t) + 0.03 * np.cos(2*t)
            
            # Añadir ruido realista
            sequence += np.random.normal(0, 0.002, sequence.shape)
            
            # Guardar secuencia
            filename = f"demo_{i+1}_q85_RH.npy"
            np.save(os.path.join(sign_path, filename), sequence)
    
    print(f"✅ Datos de demostración creados en {base_path}")

def analyze_augmentation_results(augmenter):
    """Analiza los resultados del augmentation"""
    print("\n📊 ANÁLISIS DE RESULTADOS DE AUGMENTATION")
    print("=" * 50)
    
    current_counts, remaining_deficits = augmenter.analyze_current_dataset()
    
    # Mostrar estadísticas finales
    print(f"\n📈 Estadísticas finales del dataset:")
    total_sequences = sum(current_counts.values())
    print(f"   Total de secuencias: {total_sequences}")
    
    for sign, count in current_counts.items():
        sign_type = augmenter.classify_sign_type(sign)
        print(f"   {sign} ({sign_type}): {count} secuencias")
    
    # Verificar cumplimiento del plan
    if not remaining_deficits:
        print("\n🎉 ¡PLAN DE MEJORA COMPLETADO!")
        print("   Todos los objetivos han sido alcanzados.")
    else:
        print(f"\n⚠️  Déficits restantes:")
        for sign, info in remaining_deficits.items():
            print(f"   {sign}: {info['current']}/{info['target']} "
                  f"(falta {info['deficit']})")

def visualize_augmentation_comparison(original_file, augmented_files, sign_name):
    """Visualiza comparación entre secuencia original y aumentadas"""
    print(f"\n📊 Visualizando augmentation para {sign_name}...")
    
    try:
        # Cargar secuencia original
        original = np.load(original_file)
        
        # Crear figura
        fig, axes = plt.subplots(2, len(augmented_files) + 1, figsize=(15, 8))
        if len(augmented_files) == 0:
            return
        
        # Plot original
        axes[0, 0].plot(original[:, 0], label='X coord')
        axes[0, 0].plot(original[:, 1], label='Y coord')
        axes[0, 0].set_title('Original')
        axes[0, 0].legend()
        
        axes[1, 0].plot(np.linalg.norm(np.diff(original, axis=0), axis=1))
        axes[1, 0].set_title('Movement magnitude')
        
        # Plot augmented
        for i, aug_file in enumerate(augmented_files[:4]):  # Max 4 ejemplos
            augmented = np.load(aug_file)
            
            axes[0, i+1].plot(augmented[:, 0], label='X coord')
            axes[0, i+1].plot(augmented[:, 1], label='Y coord')
            axes[0, i+1].set_title(f'Augmented {i+1}')
            axes[0, i+1].legend()
            
            axes[1, i+1].plot(np.linalg.norm(np.diff(augmented, axis=0), axis=1))
            axes[1, i+1].set_title(f'Movement {i+1}')
        
        plt.tight_layout()
        plt.savefig(f'augmentation_comparison_{sign_name}.png', dpi=150)
        plt.close()
        
        print(f"   💾 Comparación guardada: augmentation_comparison_{sign_name}.png")
        
    except Exception as e:
        print(f"   ⚠️  Error en visualización: {e}")

def main():
    """Función principal de demostración"""
    print("🚀 DEMOSTRACIÓN DEL MOTOR DE DATA AUGMENTATION")
    print("🎯 Sistema inteligente para alcanzar objetivos del plan de mejora")
    print("=" * 60)
    
    # Verificar si existen datos, si no, crearlos
    if not os.path.exists("data/sequences") or len(os.listdir("data/sequences")) == 0:
        print("\n📂 No se encontraron datos existentes.")
        create_demo = input("¿Crear datos de demostración? (s/n): ").lower() == 's'
        
        if create_demo:
            create_demo_data()
        else:
            print("❌ Necesitas datos para continuar.")
            return
    
    try:
        # Crear motor de augmentation
        print("\n🔧 Inicializando motor de augmentation...")
        augmenter = SignDataAugmentationEngine()
        
        # Mostrar análisis inicial
        print("\n📋 Análisis inicial del dataset:")
        current_counts, deficits = augmenter.analyze_current_dataset()
        
        if not deficits:
            print("✅ El dataset ya cumple con todos los objetivos del plan!")
            return
        
        # Preguntar si ejecutar augmentation
        print(f"\n🎯 Se necesita generar {sum(info['deficit'] for info in deficits.values())} secuencias")
        proceed = input("¿Proceder con el data augmentation? (s/n): ").lower() == 's'
        
        if proceed:
            # Ejecutar augmentation
            augmenter.augment_dataset_for_plan()
            
            # Análisis final
            analyze_augmentation_results(augmenter)
            
            # Demostrar visualización (si hay matplotlib)
            try:
                # Buscar ejemplos para visualizar
                for sign in ['J', 'Z', 'A']:
                    sign_path = f"data/sequences/{sign}"
                    if os.path.exists(sign_path):
                        files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                        original_files = [f for f in files if 'AUG' not in f]
                        augmented_files = [f for f in files if 'AUG' in f]
                        
                        if original_files and augmented_files:
                            original_file = os.path.join(sign_path, original_files[0])
                            aug_files = [os.path.join(sign_path, f) for f in augmented_files[:3]]
                            visualize_augmentation_comparison(original_file, aug_files, sign)
                            break
                            
            except ImportError:
                print("\n💡 Instala matplotlib para visualizaciones: pip install matplotlib")
            except Exception as e:
                print(f"\n⚠️  Error en visualización: {e}")
        
        print("\n✅ Demostración completada!")
        print("💡 El motor de augmentation puede generar datos de alta calidad")
        print("   para alcanzar los objetivos del plan de mejora sin recolección manual.")
        
    except FileNotFoundError:
        print("❌ Error: No se encontró plan_mejora_dataset.json")
        print("💡 Asegúrate de que el archivo existe en el directorio actual")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
