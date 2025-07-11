#!/usr/bin/env python3
"""
Script mejorado para recolectar datos de lenguaje de señas por lotes
Proyecto LSP Esperanza
"""

import sys
import argparse
from pathlib import Path
import os
import time
from datetime import datetime

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def get_current_data_count(sign, data_dir="data/sequences"):
    """Obtiene la cantidad actual de datos para una seña específica"""
    sign_dir = Path(data_dir) / sign.upper()
    if not sign_dir.exists():
        return 0
    
    # Contar archivos .npy
    npy_files = list(sign_dir.glob("*.npy"))
    return len(npy_files)

def calculate_batch_info(current_count, batch_size=20):
    """Calcula información del lote actual"""
    current_batch = (current_count // batch_size) + 1
    samples_in_current_batch = current_count % batch_size
    samples_needed_for_next_batch = batch_size - samples_in_current_batch if samples_in_current_batch > 0 else batch_size
    
    return {
        'current_batch': current_batch,
        'samples_in_current_batch': samples_in_current_batch,
        'samples_needed_for_next_batch': samples_needed_for_next_batch,
        'total_samples': current_count
    }

def show_sign_menu():
    """Muestra menú de selección de señas"""
    print("\n🤚 SELECCIONAR SEÑA PARA RECOLECTAR")
    print("=" * 50)
    
    # Señas estáticas
    static_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    # Señas dinámicas
    dynamic_signs = ['J', 'Z', 'Ñ', 'RR']
    
    # Palabras/frases
    word_signs = ['HOLA', 'GRACIAS', 'POR_FAVOR', 'ADIOS', 'SI', 'NO']
    
    print("📝 SEÑAS ESTÁTICAS:")
    for i, sign in enumerate(static_signs, 1):
        current_count = get_current_data_count(sign)
        batch_info = calculate_batch_info(current_count)
        status = f"Lote {batch_info['current_batch']} ({batch_info['samples_in_current_batch']}/20)"
        print(f"  {i:2d}. {sign} - {current_count} muestras - {status}")
    
    print("\n🔄 SEÑAS DINÁMICAS:")
    for i, sign in enumerate(dynamic_signs, len(static_signs) + 1):
        current_count = get_current_data_count(sign)
        batch_info = calculate_batch_info(current_count)
        status = f"Lote {batch_info['current_batch']} ({batch_info['samples_in_current_batch']}/20)"
        print(f"  {i:2d}. {sign} - {current_count} muestras - {status}")
    
    print("\n💬 PALABRAS/FRASES:")
    for i, sign in enumerate(word_signs, len(static_signs) + len(dynamic_signs) + 1):
        current_count = get_current_data_count(sign)
        batch_info = calculate_batch_info(current_count)
        status = f"Lote {batch_info['current_batch']} ({batch_info['samples_in_current_batch']}/20)"
        print(f"  {i:2d}. {sign} - {current_count} muestras - {status}")
    
    print(f"\n  {len(static_signs) + len(dynamic_signs) + len(word_signs) + 1}. 🔧 ENTRENAR MODELO CON DATOS ACTUALES")
    print(f"  {len(static_signs) + len(dynamic_signs) + len(word_signs) + 2}. 📊 VER ESTADÍSTICAS DETALLADAS")
    print(f"  {len(static_signs) + len(dynamic_signs) + len(word_signs) + 3}. ❌ SALIR")
    
    all_signs = static_signs + dynamic_signs + word_signs
    return all_signs

def show_detailed_stats():
    """Muestra estadísticas detalladas de todas las señas"""
    print("\n📊 ESTADÍSTICAS DETALLADAS")
    print("=" * 70)
    
    data_dir = Path("data/sequences")
    if not data_dir.exists():
        print("❌ Directorio de datos no encontrado")
        return
    
    total_samples = 0
    total_batches = 0
    
    print(f"{'Seña':<12} {'Muestras':<10} {'Lote':<8} {'Progreso':<15} {'Estado':<15}")
    print("-" * 70)
    
    # Obtener todas las señas existentes
    for sign_dir in sorted(data_dir.iterdir()):
        if sign_dir.is_dir():
            sign_name = sign_dir.name
            current_count = get_current_data_count(sign_name)
            batch_info = calculate_batch_info(current_count)
            
            total_samples += current_count
            total_batches += batch_info['current_batch']
            
            # Estado del lote
            if batch_info['samples_in_current_batch'] == 0 and current_count > 0:
                status = "✅ Completo"
                progress = "20/20"
            elif current_count == 0:
                status = "⚪ Sin datos"
                progress = "0/20"
            else:
                status = "🔄 En progreso"
                progress = f"{batch_info['samples_in_current_batch']}/20"
            
            print(f"{sign_name:<12} {current_count:<10} {batch_info['current_batch']:<8} {progress:<15} {status:<15}")
    
    print("-" * 70)
    print(f"TOTAL: {total_samples} muestras en {total_batches} lotes")
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    incomplete_signs = []
    for sign_dir in data_dir.iterdir():
        if sign_dir.is_dir():
            current_count = get_current_data_count(sign_dir.name)
            batch_info = calculate_batch_info(current_count)
            if batch_info['samples_in_current_batch'] > 0:
                incomplete_signs.append((sign_dir.name, batch_info['samples_needed_for_next_batch']))
    
    if incomplete_signs:
        print("  📝 Señas con lotes incompletos:")
        for sign, needed in incomplete_signs:
            print(f"     {sign}: faltan {needed} muestras para completar lote")
    else:
        print("  ✅ Todos los lotes están completos")

def collect_batch_with_progress(collector, sign, batch_size=20):
    """Recolecta un lote con indicador de progreso"""
    current_count = get_current_data_count(sign)
    batch_info = calculate_batch_info(current_count, batch_size)
    
    print(f"\n📊 RECOLECCIÓN POR LOTES - SEÑA: {sign}")
    print("=" * 50)
    print(f"📈 Estado actual:")
    print(f"   Lote actual: {batch_info['current_batch']}")
    print(f"   Muestras en lote actual: {batch_info['samples_in_current_batch']}/20")
    print(f"   Total de muestras: {batch_info['total_samples']}")
    print(f"   Faltan para completar lote: {batch_info['samples_needed_for_next_batch']}")
    print("=" * 50)
    
    # Preguntar cuántas muestras recolectar
    print(f"\n💡 Opciones de recolección:")
    print(f"   1. Completar lote actual ({batch_info['samples_needed_for_next_batch']} muestras)")
    print(f"   2. Recolectar lote completo (20 muestras)")
    print(f"   3. Cantidad personalizada")
    print(f"   4. Volver al menú principal")
    
    while True:
        try:
            choice = input(f"\n🔢 Selecciona opción (1-4): ").strip()
            
            if choice == '1':
                samples_to_collect = batch_info['samples_needed_for_next_batch']
                break
            elif choice == '2':
                samples_to_collect = 20
                break
            elif choice == '3':
                samples_to_collect = int(input("📝 Cantidad de muestras: "))
                if samples_to_collect <= 0:
                    print("❌ La cantidad debe ser mayor a 0")
                    continue
                break
            elif choice == '4':
                return False
            else:
                print("❌ Opción inválida. Selecciona 1-4")
        except ValueError:
            print("❌ Ingresa un número válido")
    
    print(f"\n🚀 Iniciando recolección de {samples_to_collect} muestras para {sign}")
    print("⏳ Preparando cámara...")
    
    try:
        # Recolectar con progreso en tiempo real
        collector.collect_data_for_sign_with_progress(
            sign=sign,
            num_samples=samples_to_collect,
            output_dir="data/sequences"
        )
        
        # Mostrar estado actualizado
        new_count = get_current_data_count(sign)
        new_batch_info = calculate_batch_info(new_count)
        
        print(f"\n✅ RECOLECCIÓN COMPLETADA")
        print(f"📈 Estado actualizado:")
        print(f"   Total de muestras: {new_batch_info['total_samples']}")
        print(f"   Lote actual: {new_batch_info['current_batch']}")
        print(f"   Progreso del lote: {new_batch_info['samples_in_current_batch']}/20")
        
        if new_batch_info['samples_in_current_batch'] == 0:
            print(f"🎉 ¡Lote {new_batch_info['current_batch']} completado!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la recolección: {e}")
        return False

def train_model_option():
    """Opción para entrenar el modelo con los datos actuales"""
    print("\n🤖 ENTRENAMIENTO DE MODELO")
    print("=" * 50)
    
    # Verificar datos disponibles
    data_dir = Path("data/sequences")
    if not data_dir.exists():
        print("❌ No se encontraron datos de entrenamiento")
        return
    
    # Contar datos
    total_samples = 0
    signs_with_data = []
    
    for sign_dir in data_dir.iterdir():
        if sign_dir.is_dir():
            count = get_current_data_count(sign_dir.name)
            if count > 0:
                signs_with_data.append((sign_dir.name, count))
                total_samples += count
    
    if total_samples == 0:
        print("❌ No hay datos suficientes para entrenar")
        return
    
    print(f"📊 Datos disponibles:")
    print(f"   Total de muestras: {total_samples}")
    print(f"   Señas con datos: {len(signs_with_data)}")
    print(f"   Señas: {', '.join([f'{sign}({count})' for sign, count in signs_with_data])}")
    
    # Recomendación
    if total_samples < 60:
        print(f"\n⚠️  ADVERTENCIA: Solo {total_samples} muestras disponibles")
        print(f"   Recomendado: Al menos 60 muestras (3 señas x 20 muestras)")
        print(f"   Sugerencia: Recolecta más datos antes de entrenar")
    else:
        print(f"\n✅ Datos suficientes para entrenamiento")
    
    # Opciones de entrenamiento
    print(f"\n🎯 Opciones de entrenamiento:")
    print(f"   1. Entrenamiento rápido (50 epochs)")
    print(f"   2. Entrenamiento estándar (100 epochs)")
    print(f"   3. Entrenamiento intensivo (200 epochs)")
    print(f"   4. Volver al menú principal")
    
    while True:
        try:
            choice = input(f"\n🔢 Selecciona opción (1-4): ").strip()
            
            if choice == '1':
                epochs = 50
                break
            elif choice == '2':
                epochs = 100
                break
            elif choice == '3':
                epochs = 200
                break
            elif choice == '4':
                return
            else:
                print("❌ Opción inválida. Selecciona 1-4")
        except ValueError:
            print("❌ Ingresa un número válido")
    
    # Confirmar entrenamiento
    print(f"\n⚠️  CONFIRMAR ENTRENAMIENTO:")
    print(f"   Épocas: {epochs}")
    print(f"   Muestras: {total_samples}")
    print(f"   Tiempo estimado: {epochs//10}-{epochs//5} minutos")
    
    confirm = input(f"\n❓ ¿Continuar con el entrenamiento? (s/N): ").strip().lower()
    
    if confirm in ['s', 'si', 'yes', 'y']:
        print(f"\n🚀 Iniciando entrenamiento...")
        try:
            # Importar y ejecutar entrenador
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from models.trainer import EnhancedModelTrainer
            
            trainer = EnhancedModelTrainer(data_path="data/sequences")
            trainer.train_bidirectional_dynamic_model(
                epochs=epochs,
                batch_size=32
            )
            
            print("✅ Entrenamiento completado exitosamente")
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Entrenamiento cancelado")

def interactive_mode():
    """Modo interactivo con menú"""
    try:
        from data_processing.collector import DataCollector
    except ImportError:
        print("❌ Error: No se puede importar DataCollector")
        print("💡 Verifica que el proyecto esté correctamente configurado")
        return 1
    
    collector = DataCollector()
    
    print("🤚 RECOLECTOR DE DATOS LSP ESPERANZA - MODO INTERACTIVO")
    print("📦 Sistema de lotes de 20 muestras")
    print("🎯 Entrenamiento integrado")
    
    while True:
        all_signs = show_sign_menu()
        
        try:
            choice = input(f"\n🔢 Selecciona opción (1-{len(all_signs) + 3}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(all_signs):
                # Recolectar datos para la seña seleccionada
                selected_sign = all_signs[choice_num - 1]
                success = collect_batch_with_progress(collector, selected_sign)
                
                if success:
                    input("\n⏸️  Presiona Enter para continuar...")
                
            elif choice_num == len(all_signs) + 1:
                # Entrenar modelo
                train_model_option()
                input("\n⏸️  Presiona Enter para continuar...")
                
            elif choice_num == len(all_signs) + 2:
                # Ver estadísticas
                show_detailed_stats()
                input("\n⏸️  Presiona Enter para continuar...")
                
            elif choice_num == len(all_signs) + 3:
                # Salir
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida")
                
        except ValueError:
            print("❌ Ingresa un número válido")
        except KeyboardInterrupt:
            print("\n👋 Saliendo del programa...")
            break

def main():
    parser = argparse.ArgumentParser(
        description='Recolector mejorado de datos de lenguaje de señas por lotes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python collect_data.py                          # Modo interactivo con menú
  python collect_data.py --sign A                # Recolectar seña A (modo lote)
  python collect_data.py --sign J --samples 10   # Recolectar 10 muestras de J
  python collect_data.py --interactive            # Forzar modo interactivo
  python collect_data.py --stats                  # Ver estadísticas detalladas
        """
    )
    
    parser.add_argument(
        '--sign',
        help='Seña a recolectar (ejemplo: A, B, J, HOLA). Si no se especifica, usa modo interactivo'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        help='Número de muestras a recolectar. Por defecto completa el lote actual'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/sequences',
        help='Directorio de salida (default: %(default)s)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Forzar modo interactivo con menú'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Mostrar estadísticas detalladas y salir'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Tamaño de lote (default: %(default)s)'
    )
    
    args = parser.parse_args()
    
    # Mostrar estadísticas y salir
    if args.stats:
        show_detailed_stats()
        return 0
    
    # Modo interactivo si no se especifica seña o se fuerza
    if not args.sign or args.interactive:
        return interactive_mode()
    
    # Modo línea de comandos
    print("📊 RECOLECTOR DE DATOS LSP ESPERANZA - MODO LOTE")
    print("=" * 60)
    
    # Verificar información actual
    current_count = get_current_data_count(args.sign, args.output_dir)
    batch_info = calculate_batch_info(current_count, args.batch_size)
    
    print(f"✋ Seña: {args.sign.upper()}")
    print(f"� Estado actual:")
    print(f"   Total de muestras: {batch_info['total_samples']}")
    print(f"   Lote actual: {batch_info['current_batch']}")
    print(f"   Progreso del lote: {batch_info['samples_in_current_batch']}/{args.batch_size}")
    print(f"   Faltan para completar lote: {batch_info['samples_needed_for_next_batch']}")
    
    # Determinar cantidad a recolectar
    if args.samples:
        samples_to_collect = args.samples
        print(f"📝 Muestras a recolectar: {samples_to_collect} (especificado)")
    else:
        samples_to_collect = batch_info['samples_needed_for_next_batch']
        print(f"📝 Muestras a recolectar: {samples_to_collect} (completar lote)")
    
    print("=" * 60)
    
    try:
        from data_processing.collector import DataCollector
        
        collector = DataCollector()
        
        # Mostrar countdown antes de iniciar
        print(f"\n🚀 Iniciando recolección en:")
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        print("   ¡COMENZANDO!")
        
        # Recolectar con progreso
        collector.collect_data_for_sign_with_progress(
            sign=args.sign.upper(),
            num_samples=samples_to_collect,
            output_dir=args.output_dir
        )
        
        # Mostrar estado final
        final_count = get_current_data_count(args.sign, args.output_dir)
        final_batch_info = calculate_batch_info(final_count, args.batch_size)
        
        print(f"\n✅ RECOLECCIÓN COMPLETADA")
        print(f"📈 Estado final:")
        print(f"   Total de muestras: {final_batch_info['total_samples']}")
        print(f"   Lote actual: {final_batch_info['current_batch']}")
        print(f"   Progreso del lote: {final_batch_info['samples_in_current_batch']}/{args.batch_size}")
        
        if final_batch_info['samples_in_current_batch'] == 0:
            print(f"🎉 ¡Lote {final_batch_info['current_batch']} completado!")
            print(f"💡 Siguiente objetivo: Lote {final_batch_info['current_batch'] + 1}")
        
        # Sugerir próximos pasos
        print(f"\n💡 SUGERENCIAS:")
        if final_count >= 60:  # 3 lotes completos
            print(f"   ✅ Datos suficientes para entrenar modelo")
            print(f"   🎯 Comando: python scripts/train_model.py")
        else:
            print(f"   📊 Recolecta más datos para mejor entrenamiento")
            print(f"   🎯 Objetivo: {60 - final_count} muestras más para 3 lotes")
        
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
