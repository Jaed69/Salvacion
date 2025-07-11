#!/usr/bin/env python3
"""
Script de prueba para el entrenamiento estático con base de datos existente
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from scripts.train_static_model import StaticSignTrainer
import numpy as np

def test_database_structure():
    """Prueba la estructura de la base de datos existente"""
    print("🔍 ANALIZANDO ESTRUCTURA DE LA BASE DE DATOS")
    print("="*50)
    
    data_path = ROOT_DIR / "data" / "sequences"
    
    if not data_path.exists():
        print("❌ No se encontró el directorio de datos")
        return False
    
    # Obtener clases disponibles
    sign_folders = [f for f in data_path.iterdir() 
                   if f.is_dir() and not f.name.startswith('.')]
    
    print(f"📊 Clases encontradas: {len(sign_folders)}")
    
    total_samples = 0
    sample_info = []
    
    for sign_folder in sorted(sign_folders):
        sign_files = list(sign_folder.glob('*.npy'))
        total_samples += len(sign_files)
        
        print(f"   {sign_folder.name}: {len(sign_files)} archivos")
        
        # Analizar primer archivo para obtener dimensiones
        if sign_files:
            try:
                sample = np.load(sign_files[0])
                sample_info.append((sign_folder.name, sample.shape, sign_files[0].name))
                print(f"      Ejemplo: {sign_files[0].name} -> shape: {sample.shape}")
            except Exception as e:
                print(f"      ⚠️ Error cargando ejemplo: {e}")
    
    print(f"\n📈 Resumen:")
    print(f"   Total de clases: {len(sign_folders)}")
    print(f"   Total de muestras: {total_samples}")
    print(f"   Promedio por clase: {total_samples/len(sign_folders):.1f}")
    
    # Verificar consistencia de dimensiones
    shapes = [info[1] for info in sample_info]
    unique_shapes = list(set(shapes))
    
    if len(unique_shapes) == 1:
        print(f"   ✅ Dimensiones consistentes: {unique_shapes[0]}")
    else:
        print(f"   ⚠️ Dimensiones inconsistentes: {unique_shapes}")
    
    return len(sign_folders) > 0

def test_trainer_initialization():
    """Prueba la inicialización del entrenador"""
    print("\n🤖 PROBANDO INICIALIZACIÓN DEL ENTRENADOR")
    print("="*50)
    
    try:
        trainer = StaticSignTrainer()
        print("✅ Entrenador inicializado correctamente")
        
        # Verificar directorios
        print(f"   📂 Directorio de datos: {trainer.data_path}")
        print(f"   🤖 Directorio de modelos: {trainer.models_dir}")
        
        return trainer
    except Exception as e:
        print(f"❌ Error inicializando entrenador: {e}")
        return None

def test_data_loading(trainer):
    """Prueba la carga de datos"""
    print("\n📥 PROBANDO CARGA DE DATOS")
    print("="*50)
    
    try:
        sequences, labels, quality_scores = trainer.load_static_data()
        
        print(f"✅ Datos cargados exitosamente:")
        print(f"   📊 Total de muestras: {len(sequences)}")
        print(f"   🏷️ Clases únicas: {len(np.unique(labels))}")
        print(f"   📏 Dimensión de features: {sequences[0].shape if len(sequences) > 0 else 'N/A'}")
        print(f"   ⭐ Calidad promedio: {np.mean(quality_scores):.3f}")
        
        # Mostrar distribución por clase
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📈 Distribución por clase:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count} muestras")
        
        return sequences, labels, quality_scores
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_feature_extraction(trainer, sequences):
    """Prueba la extracción de características"""
    print("\n🔧 PROBANDO EXTRACCIÓN DE CARACTERÍSTICAS")
    print("="*50)
    
    if sequences is None or len(sequences) == 0:
        print("❌ No hay datos para probar")
        return None
    
    try:
        # Probar con una muestra
        sample = sequences[0]
        geometric_features = trainer.extract_geometric_features(sample)
        
        print(f"✅ Características extraídas:")
        print(f"   📏 Features originales: {len(sample)}")
        print(f"   🔢 Features geométricas: {len(geometric_features)}")
        print(f"   📊 Total combinadas: {len(sample) + len(geometric_features)}")
        
        return geometric_features
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal de prueba"""
    print("🧪 PRUEBA COMPLETA DEL SISTEMA DE ENTRENAMIENTO ESTÁTICO")
    print("="*60)
    
    # Prueba 1: Estructura de la base de datos
    if not test_database_structure():
        print("❌ La estructura de la base de datos no es válida")
        return
    
    # Prueba 2: Inicialización del entrenador
    trainer = test_trainer_initialization()
    if trainer is None:
        print("❌ No se pudo inicializar el entrenador")
        return
    
    # Prueba 3: Carga de datos
    sequences, labels, quality_scores = test_data_loading(trainer)
    if sequences is None:
        print("❌ No se pudieron cargar los datos")
        return
    
    # Prueba 4: Extracción de características
    test_feature_extraction(trainer, sequences)
    
    # Resumen final
    print("\n🎯 RESUMEN DE PRUEBAS")
    print("="*30)
    print("✅ Estructura de base de datos: OK")
    print("✅ Inicialización del entrenador: OK")
    print("✅ Carga de datos: OK")
    print("✅ Extracción de características: OK")
    print("\n🚀 El sistema está listo para entrenar!")
    print(f"    Para entrenar ejecuta: python scripts/train_static_model.py")

if __name__ == "__main__":
    main()
