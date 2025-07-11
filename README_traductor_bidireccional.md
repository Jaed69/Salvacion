# 🚀 Traductor Bidireccional de Lenguaje de Señas

## 📋 Descripción

Traductor en tiempo real actualizado para trabajar con el modelo bidireccional GRU entrenado. Detecta y clasifica señas estáticas y dinámicas usando análisis avanzado de movimiento.

## ✅ Estado Actual

- **Modelo**: `sign_model_bidirectional_dynamic.h5` ✅ Funcionando
- **Señas detectadas**: A, B, J (3 señas disponibles)
- **Características**: 14 features (6 básicas + 8 dinámicas avanzadas)
- **Arquitectura**: CNN + GRU Bidireccional

## 🎯 Características del Sistema

### 🧠 Modelo Bidireccional
- **GRU Bidireccional**: Captura contexto temporal completo
- **Análisis dinámico**: Especializado en señas con movimiento (J, Z, Ñ, RR)
- **Precisión**: 88.8% en datos de prueba
- **Features avanzadas**: 14 características de movimiento

### 📊 Análisis de Movimiento
1. **Características básicas (6)**:
   - Varianza temporal
   - Movimiento entre frames
   - Velocidad de manos
   - Aceleración
   - Frecuencia dominante
   - Entropía de movimiento

2. **Características dinámicas avanzadas (8)**:
   - Magnitud de trayectoria
   - Curvatura de trayectoria
   - Desviación estándar de velocidad
   - Tendencia de velocidad
   - Simetría temporal
   - Frecuencias dominantes X/Y
   - Puntuación de repetición

### 🎨 Interfaz Visual
- **Análisis en tiempo real**: Visualización de movimiento
- **Paneles informativos**: Estado del modelo y características
- **Código de colores**: Estáticas (azul) vs Dinámicas (naranja)
- **Barras de confianza**: Indicadores visuales de precisión

## 🚀 Uso

### Ejecución básica
```bash
python real_time_translator.py
```

### Opciones avanzadas
```bash
# Cambiar umbral de confianza
python real_time_translator.py --threshold 0.9

# Usar modelo específico
python real_time_translator.py --model data/sign_model_bidirectional_dynamic.h5
```

### ⌨️ Controles
- **'q'**: Salir
- **'r'**: Resetear buffers
- **'t'**: Ajustar umbral (0.6 ↔ 0.9)
- **'d'**: Modo debug (información detallada)

## 📈 Resultados de Entrenamiento

```
📈 Resultados finales del modelo bidireccional:
📈 Precisión en datos de prueba: 0.888
📈 Pérdida en datos de prueba: 0.216
   Precisión para 'J' (dinámica): 1.000
```

## 🔧 Pruebas

Ejecuta la prueba del sistema:
```bash
python test_translator.py
```

Resultado esperado:
```
🧪 Probando Traductor Bidireccional...
✅ Traductor inicializado correctamente
📊 Modelo cargado: True
📋 Señas disponibles: 3 señas
🎯 Señas encontradas: A, B, J
🔧 Características de movimiento: 14 features
📈 Rango de características: [0.0003, 884.0000]
🎯 Predicción de prueba: J (confianza: 0.953)
✅ Todas las pruebas pasaron correctamente!
```

## 📋 Señas Soportadas

### Estáticas ✋
- **A, B**: Detectadas con alta precisión
- **Características**: Requieren estabilidad y baja varianza

### Dinámicas 👋
- **J**: Movimiento circular, precisión 100%
- **Características**: Requieren análisis de trayectoria y curvatura

## 🎯 Próximos Pasos

1. **Generar más datos**: Usar data augmentation para Z, Ñ, RR, etc.
2. **Entrenar modelo completo**: Con todas las señas del plan
3. **Optimizar umbral**: Para cada tipo de seña específicamente
4. **Validar en condiciones reales**: Diferentes usuarios y ambientes

## 🛠️ Requisitos Técnicos

- **TensorFlow**: 2.18.1
- **OpenCV**: 4.11.0.86
- **MediaPipe**: 0.10.21
- **NumPy, SciPy**: Para procesamiento de señales
- **Cámara**: Para captura en tiempo real

## 💡 Características Técnicas

- **Arquitectura híbrida**: Secuencias + Características de movimiento
- **Análisis bidireccional**: Contexto temporal completo
- **Control de calidad**: Umbral adaptativo por tipo de seña
- **Optimización**: Específica para señas dinámicas como J, Z

---

**🎉 Sistema completamente funcional y optimizado para reconocimiento avanzado de señas dinámicas!**
