import numpy as np
import os

# Comparar J con otras letras dinámicas
letters = ['A', 'B', 'J']
print('Comparación entre letras dinámicas:')
print('====================================')

for letter in letters:
    if os.path.exists(f'data/sequences/{letter}'):
        files = os.listdir(f'data/sequences/{letter}')[:3]
        movements = []
        variances = []
        trajectories = []
        
        for file in files:
            data = np.load(f'data/sequences/{letter}/{file}')
            
            # Varianza temporal
            variance = np.var(data, axis=0).mean()
            variances.append(variance)
            
            # Movimiento promedio
            if len(data) > 1:
                movement = np.mean([np.mean(np.abs(data[i+1] - data[i])) for i in range(len(data)-1)])
                movements.append(movement)
            
            # Trayectoria
            start_pos = np.mean(data[:3], axis=0)
            end_pos = np.mean(data[-3:], axis=0)
            trajectory = np.linalg.norm(end_pos - start_pos)
            trajectories.append(trajectory)
        
        print(f'Letra {letter}:')
        print(f'  Varianza promedio: {np.mean(variances):.6f} ± {np.std(variances):.6f}')
        print(f'  Movimiento promedio: {np.mean(movements):.6f} ± {np.std(movements):.6f}')
        print(f'  Trayectoria promedio: {np.mean(trajectories):.6f} ± {np.std(trajectories):.6f}')
        print(f'  Número de archivos: {len(os.listdir(f"data/sequences/{letter}"))}')
        print()

# Análisis específico de calidad de datos J
print('Análisis de calidad de datos para J:')
print('=====================================')

j_files = os.listdir('data/sequences/J')
quality_stats = {'exc': 0, 'bue': 0, 'reg': 0}

for file in j_files:
    if '_exc_' in file:
        quality_stats['exc'] += 1
    elif '_bue_' in file:
        quality_stats['bue'] += 1
    elif '_reg_' in file:
        quality_stats['reg'] += 1

print(f"Distribución de calidad en J:")
print(f"  Excelente (exc): {quality_stats['exc']}")
print(f"  Bueno (bue): {quality_stats['bue']}")
print(f"  Regular (reg): {quality_stats['reg']}")

# Comparar con otras letras
for letter in ['A', 'B']:
    if os.path.exists(f'data/sequences/{letter}'):
        files = os.listdir(f'data/sequences/{letter}')
        quality_stats = {'exc': 0, 'bue': 0, 'reg': 0}
        
        for file in files:
            if '_exc_' in file:
                quality_stats['exc'] += 1
            elif '_bue_' in file:
                quality_stats['bue'] += 1
            elif '_reg_' in file:
                quality_stats['reg'] += 1
        
        print(f"Distribución de calidad en {letter}:")
        print(f"  Excelente (exc): {quality_stats['exc']}")
        print(f"  Bueno (bue): {quality_stats['bue']}")
        print(f"  Regular (reg): {quality_stats['reg']}")
