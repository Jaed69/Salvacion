#!/usr/bin/env python3
"""
Script simplificado para debug del error de arrays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Crear directorio
output_dir = Path("reports/statistical_analysis_graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# Datos de prueba
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXY')
print(f"Número de letras: {len(letters)}")

# 5. Distribución de confianza por clase - SIMPLIFICADO
np.random.seed(42)
min_conf = np.random.uniform(90, 95, 24)
max_conf = np.random.uniform(97, 100, 24)
max_conf = np.maximum(max_conf, min_conf + 2)

confidence_distribution = {
    'Letter': letters,
    'Min_Confidence': min_conf,
    'Max_Confidence': max_conf,
    'Avg_Confidence': (min_conf + max_conf) / 2
}

print("Creando DataFrame...")
df = pd.DataFrame(confidence_distribution)
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

for col in df.columns:
    print(f"{col}: {len(df[col])}")

print("Creando gráfico simple...")
fig, ax = plt.subplots(figsize=(12, 6))

# Test simple bar chart
bars = ax.bar(df['Letter'], df['Avg_Confidence'])
ax.set_title('Test Chart')

plt.tight_layout()
plt.savefig(output_dir / 'test_chart.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Test completado exitosamente")
