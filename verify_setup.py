#!/usr/bin/env python3
"""
Script de verificación del proyecto LSP Esperanza
Verifica que todos los componentes estén correctamente configurados
"""

import sys
import importlib
from pathlib import Path
import traceback

# Agregar src al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def check_import(module_name, description):
    """Verifica que un módulo pueda importarse"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description} - Advertencia: {e}")
        return False

def check_file_exists(file_path, description):
    """Verifica que un archivo exista"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✅ {description} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ {description} - No encontrado: {file_path}")
        return False

def main():
    print("🔍 VERIFICACIÓN DEL PROYECTO LSP ESPERANZA")
    print("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # ========== DEPENDENCIAS BÁSICAS ==========
    print("\n📦 VERIFICANDO DEPENDENCIAS BÁSICAS:")
    basic_deps = [
        ("numpy", "NumPy (computación numérica)"),
        ("cv2", "OpenCV (procesamiento de video)"),
        ("mediapipe", "MediaPipe (detección de landmarks)"),
        ("tensorflow", "TensorFlow (deep learning)"),
        ("sklearn", "Scikit-learn (machine learning)"),
        ("scipy", "SciPy (algoritmos científicos)"),
        ("matplotlib", "Matplotlib (visualización)"),
    ]
    
    for module, desc in basic_deps:
        if check_import(module, desc):
            success_count += 1
        total_checks += 1
    
    # ========== ESTRUCTURA DEL PROYECTO ==========
    print("\n📁 VERIFICANDO ESTRUCTURA DEL PROYECTO:")
    structure_checks = [
        (project_root / "src", "Directorio de código fuente"),
        (project_root / "config", "Directorio de configuración"),
        (project_root / "data", "Directorio de datos"),
        (project_root / "models", "Directorio de modelos"),
        (project_root / "scripts", "Directorio de scripts"),
        (project_root / "tests", "Directorio de tests"),
        (project_root / "docs", "Directorio de documentación"),
        (project_root / "reports", "Directorio de reportes"),
    ]
    
    for path, desc in structure_checks:
        if check_file_exists(path, desc):
            success_count += 1
        total_checks += 1
    
    # ========== ARCHIVOS PRINCIPALES ==========
    print("\n📄 VERIFICANDO ARCHIVOS PRINCIPALES:")
    main_files = [
        (project_root / "main.py", "Script principal"),
        (project_root / "requirements.txt", "Dependencias"),
        (project_root / "README.md", "Documentación principal"),
        (project_root / "config" / "settings.py", "Configuración"),
        (project_root / ".gitignore", "Configuración de Git"),
    ]
    
    for path, desc in main_files:
        if check_file_exists(path, desc):
            success_count += 1
        total_checks += 1
    
    # ========== MÓDULOS DEL PROYECTO ==========
    print("\n🐍 VERIFICANDO MÓDULOS DEL PROYECTO:")
    project_modules = [
        ("config.settings", "Configuración centralizada"),
        ("src.utils.common", "Utilidades comunes"),
        ("src.data_processing.augmentation_engine", "Motor de augmentación"),
        ("src.models.trainer", "Entrenador de modelos"),
        ("src.translation.real_time_translator", "Traductor en tiempo real"),
    ]
    
    for module, desc in project_modules:
        if check_import(module, desc):
            success_count += 1
        total_checks += 1
    
    # ========== ARCHIVOS DEL MODELO ==========
    print("\n🤖 VERIFICANDO MODELOS ENTRENADOS:")
    model_files = [
        (project_root / "models" / "sign_model_bidirectional_dynamic.h5", "Modelo bidireccional"),
        (project_root / "models" / "label_encoder.npy", "Codificador de etiquetas"),
    ]
    
    for path, desc in model_files:
        if check_file_exists(path, desc):
            success_count += 1
        total_checks += 1
    
    # ========== DATOS DE ENTRENAMIENTO ==========
    print("\n📊 VERIFICANDO DATOS DE ENTRENAMIENTO:")
    sequences_dir = project_root / "data" / "sequences"
    if sequences_dir.exists():
        sign_dirs = [d for d in sequences_dir.iterdir() if d.is_dir()]
        print(f"✅ Directorio de secuencias ({len(sign_dirs)} señas)")
        success_count += 1
        
        for sign_dir in sign_dirs:
            sequences = list(sign_dir.glob("*.npy"))
            if sequences:
                print(f"   📝 {sign_dir.name}: {len(sequences)} secuencias")
            else:
                print(f"   ⚠️  {sign_dir.name}: Sin secuencias")
    else:
        print("❌ Directorio de secuencias no encontrado")
    total_checks += 1
    
    # ========== SCRIPTS EJECUTABLES ==========
    print("\n🚀 VERIFICANDO SCRIPTS EJECUTABLES:")
    scripts = [
        (project_root / "scripts" / "collect_data.py", "Recolección de datos"),
        (project_root / "scripts" / "train_model.py", "Entrenamiento de modelos"),
        (project_root / "scripts" / "run_augmentation.py", "Augmentación de datos"),
    ]
    
    for path, desc in scripts:
        if check_file_exists(path, desc):
            success_count += 1
        total_checks += 1
    
    # ========== RESULTADO FINAL ==========
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE VERIFICACIÓN:")
    print(f"✅ Exitosos: {success_count}")
    print(f"❌ Fallidos: {total_checks - success_count}")
    print(f"📊 Total: {total_checks}")
    
    success_rate = (success_count / total_checks) * 100
    print(f"🎯 Tasa de éxito: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 ¡PROYECTO COMPLETAMENTE FUNCIONAL!")
        print("💡 Puedes ejecutar: python main.py")
    elif success_rate >= 75:
        print("\n⚠️  PROYECTO MAYORMENTE FUNCIONAL")
        print("💡 Algunos componentes opcionales fallan")
    elif success_rate >= 50:
        print("\n🔧 PROYECTO NECESITA CONFIGURACIÓN")
        print("💡 Ejecuta: python -m pip install -r requirements.txt")
    else:
        print("\n❌ PROYECTO NECESITA REPARACIÓN")
        print("💡 Verifica la instalación y configuración")
    
    # ========== PRUEBA RÁPIDA DE FUNCIONALIDAD ==========
    print("\n🧪 PRUEBA RÁPIDA DE FUNCIONALIDAD:")
    try:
        from src.utils.common import print_system_info, get_project_stats
        print("✅ Utilidades comunes funcionando")
        
        stats = get_project_stats()
        print(f"✅ Estadísticas: {stats['data_sequences']} secuencias, {stats['models_count']} modelos")
        
    except Exception as e:
        print(f"❌ Error en prueba de funcionalidad: {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 VERIFICACIÓN COMPLETADA")
    
    # Retornar código de salida basado en el éxito
    return 0 if success_rate >= 75 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
