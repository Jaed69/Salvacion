#!/usr/bin/env python3
# diagnose_j_problem.py
# Diagnóstico simple de por qué la J no se reconoce bien

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_j_data():
    """Análisis detallado de los datos de J vs otras letras"""
    
    print("🔍 DIAGNÓSTICO: ¿Por qué no se reconoce la letra J?")
    print("="*60)
    
    data_path = 'data/sequences'
    signs = ['A', 'B', 'J']  # Comparar con letras similares
    
    results = {}
    
    for sign in signs:
        sign_path = f'{data_path}/{sign}'
        if not os.path.exists(sign_path):
            print(f"❌ No se encontró directorio para {sign}")
            continue
            
        files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
        
        # Estadísticas básicas
        print(f"\n📊 ANÁLISIS DE LA LETRA {sign}")
        print("-" * 30)
        print(f"Número de archivos: {len(files)}")
        
        # Análisis de calidad por nombre de archivo
        quality_stats = {'exc': 0, 'bue': 0, 'reg': 0, 'other': 0}
        hand_stats = {'LH': 0, 'RH': 0, 'other': 0}
        
        movements = []
        variances = []
        trajectories = []
        sequence_lengths = []
        
        for file in files:
            # Extraer calidad del nombre
            if '_exc_' in file:
                quality_stats['exc'] += 1
            elif '_bue_' in file:
                quality_stats['bue'] += 1
            elif '_reg_' in file:
                quality_stats['reg'] += 1
            else:
                quality_stats['other'] += 1
            
            # Extraer mano dominante
            if '_LH.npy' in file:
                hand_stats['LH'] += 1
            elif '_RH.npy' in file:
                hand_stats['RH'] += 1
            else:
                hand_stats['other'] += 1
            
            # Cargar y analizar datos
            try:
                data = np.load(f'{sign_path}/{file}')
                sequence_lengths.append(len(data))
                
                # Calcular métricas de movimiento
                variance = np.var(data, axis=0).mean()
                variances.append(variance)
                
                if len(data) > 1:
                    movement = np.mean([np.mean(np.abs(data[i+1] - data[i])) for i in range(len(data)-1)])
                    movements.append(movement)
                
                # Trayectoria total
                start_pos = np.mean(data[:3], axis=0) if len(data) >= 3 else data[0]
                end_pos = np.mean(data[-3:], axis=0) if len(data) >= 3 else data[-1]
                trajectory = np.linalg.norm(end_pos - start_pos)
                trajectories.append(trajectory)
                
            except Exception as e:
                print(f"⚠️ Error cargando {file}: {e}")
        
        # Mostrar estadísticas
        print(f"Calidad de datos:")
        for quality, count in quality_stats.items():
            if count > 0:
                percentage = (count / len(files)) * 100
                print(f"  {quality}: {count} ({percentage:.1f}%)")
        
        print(f"Mano dominante:")
        for hand, count in hand_stats.items():
            if count > 0:
                percentage = (count / len(files)) * 100
                print(f"  {hand}: {count} ({percentage:.1f}%)")
        
        if movements:
            print(f"Características de movimiento:")
            print(f"  Varianza promedio: {np.mean(variances):.8f} ± {np.std(variances):.8f}")
            print(f"  Movimiento promedio: {np.mean(movements):.8f} ± {np.std(movements):.8f}")
            print(f"  Trayectoria promedio: {np.mean(trajectories):.6f} ± {np.std(trajectories):.6f}")
            print(f"  Longitud secuencia: {np.mean(sequence_lengths):.1f} ± {np.std(sequence_lengths):.1f}")
        
        results[sign] = {
            'files': len(files),
            'quality': quality_stats,
            'hands': hand_stats,
            'variance': np.mean(variances) if variances else 0,
            'movement': np.mean(movements) if movements else 0,
            'trajectory': np.mean(trajectories) if trajectories else 0,
            'length': np.mean(sequence_lengths) if sequence_lengths else 0
        }
    
    # Análisis comparativo
    print(f"\n🔍 ANÁLISIS COMPARATIVO")
    print("="*40)
    
    if 'J' in results:
        j_data = results['J']
        
        print(f"Problemas identificados con la letra J:")
        
        # 1. Cantidad de datos
        min_files = min([data['files'] for data in results.values()])
        if j_data['files'] == min_files:
            print(f"⚠️ J tiene pocos datos: {j_data['files']} archivos")
        
        # 2. Calidad de datos
        j_quality_ratio = j_data['quality']['exc'] / j_data['files']
        avg_quality_ratio = np.mean([
            data['quality']['exc'] / data['files'] 
            for sign, data in results.items() if sign != 'J'
        ])
        
        if j_quality_ratio < avg_quality_ratio:
            print(f"⚠️ J tiene menor calidad promedio: {j_quality_ratio:.2f} vs {avg_quality_ratio:.2f}")
        
        # 3. Características de movimiento extremas
        j_movement = j_data['movement']
        other_movements = [data['movement'] for sign, data in results.items() if sign != 'J']
        movement_ratio = j_movement / np.mean(other_movements) if other_movements else 1
        
        if movement_ratio > 3 or movement_ratio < 0.3:
            print(f"⚠️ J tiene movimiento extremo: {movement_ratio:.1f}x comparado con otras letras")
        
        # 4. Distribución de manos
        j_hand_balance = abs(j_data['hands']['LH'] - j_data['hands']['RH']) / j_data['files']
        if j_hand_balance > 0.6:
            print(f"⚠️ J tiene desbalance de manos: LH={j_data['hands']['LH']}, RH={j_data['hands']['RH']}")
        
        # 5. Longitud de secuencias
        j_length = j_data['length']
        other_lengths = [data['length'] for sign, data in results.items() if sign != 'J']
        length_diff = abs(j_length - np.mean(other_lengths)) / np.mean(other_lengths) if other_lengths else 0
        
        if length_diff > 0.2:
            print(f"⚠️ J tiene longitud de secuencia muy diferente: {j_length:.1f} vs {np.mean(other_lengths):.1f}")
    
    return results

def recommend_solutions(results):
    """Recomendar soluciones específicas basadas en el diagnóstico"""
    
    print(f"\n💡 RECOMENDACIONES PARA MEJORAR EL RECONOCIMIENTO DE J")
    print("="*60)
    
    if 'J' not in results:
        print("❌ No se pudo analizar la letra J")
        return
    
    j_data = results['J']
    solutions = []
    
    # Problema 1: Pocos datos
    if j_data['files'] < 15:
        solutions.append(
            "1. 📊 RECOLECTAR MÁS DATOS:\n"
            f"   - Actual: {j_data['files']} muestras\n"
            "   - Recomendado: mínimo 30-50 muestras\n"
            "   - Ejecutar: python scripts/collect_data.py --sign J --samples 20"
        )
    
    # Problema 2: Calidad baja
    exc_ratio = j_data['quality']['exc'] / j_data['files']
    if exc_ratio < 0.3:
        solutions.append(
            "2. 🎯 MEJORAR CALIDAD DE DATOS:\n"
            f"   - Solo {exc_ratio:.1%} de muestras son excelentes\n"
            "   - Recolectar muestras con mejor iluminación\n"
            "   - Hacer movimientos más lentos y precisos\n"
            "   - Mantener las manos dentro del marco de la cámara"
        )
    
    # Problema 3: Desbalance de manos
    hand_imbalance = abs(j_data['hands']['LH'] - j_data['hands']['RH']) / j_data['files']
    if hand_imbalance > 0.4:
        dominant_hand = 'LH' if j_data['hands']['LH'] > j_data['hands']['RH'] else 'RH'
        minority_hand = 'RH' if dominant_hand == 'LH' else 'LH'
        solutions.append(
            "3. ⚖️ BALANCEAR MANOS:\n"
            f"   - Desbalance actual: {hand_imbalance:.1%}\n"
            f"   - Recolectar más muestras con mano {minority_hand}\n"
            f"   - Actual {minority_hand}: {j_data['hands'][minority_hand]} muestras"
        )
    
    # Problema 4: Movimiento extremo
    other_movements = [data['movement'] for sign, data in results.items() if sign != 'J']
    if other_movements:
        movement_ratio = j_data['movement'] / np.mean(other_movements)
        if movement_ratio > 5:
            solutions.append(
                "4. 🐌 REDUCIR VELOCIDAD DE MOVIMIENTO:\n"
                f"   - J se mueve {movement_ratio:.1f}x más rápido que otras letras\n"
                "   - Hacer la seña J más lentamente\n"
                "   - Pausar ligeramente en cada parte del movimiento\n"
                "   - Considerar dividir J en sub-movimientos"
            )
        elif movement_ratio < 0.2:
            solutions.append(
                "4. 🚀 AUMENTAR MOVIMIENTO:\n"
                f"   - J se mueve {movement_ratio:.1f}x menos que otras letras\n"
                "   - Hacer movimientos más amplios y visibles\n"
                "   - Asegurar que la curva de la J sea bien definida"
            )
    
    # Problema 5: Arquitectura del modelo
    solutions.append(
        "5. 🏗️ AJUSTAR ARQUITECTURA DEL MODELO:\n"
        "   - Usar normalización específica para señas dinámicas\n"
        "   - Implementar atención temporal para capturar patrones de J\n"
        "   - Añadir características específicas de curvatura\n"
        "   - Usar data augmentation específico para J"
    )
    
    # Problema 6: Parámetros de entrenamiento
    solutions.append(
        "6. ⚙️ OPTIMIZAR ENTRENAMIENTO:\n"
        "   - Usar class_weight para dar más importancia a J\n"
        "   - Implementar focal loss para clases difíciles\n"
        "   - Aumentar epochs específicamente para J\n"
        "   - Usar learning rate más bajo para mejor convergencia"
    )
    
    # Mostrar soluciones
    for i, solution in enumerate(solutions, 1):
        print(f"{solution}\n")
    
    # Script de mejora rápida
    print("🚀 SCRIPT DE MEJORA RÁPIDA:")
    print("-" * 30)
    print("# 1. Recolectar más datos de J")
    print("python scripts/collect_data.py --sign J --samples 15 --quality high")
    print()
    print("# 2. Entrenar modelo con pesos balanceados")
    print("python scripts/train_model.py --model-type bidirectional_dynamic --balance-classes --focus-sign J")
    print()
    print("# 3. Probar reconocimiento")
    print("python main.py --test-sign J")

if __name__ == "__main__":
    # Ejecutar diagnóstico
    results = analyze_j_data()
    recommend_solutions(results)
