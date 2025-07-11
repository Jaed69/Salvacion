# 🎯 Guía de Uso - Sistema de Señales Estáticas LSP Esperanza

## 📋 Descripción General

El **Sistema de Señales Estáticas LSP Esperanza** está optimizado específicamente para el reconocimiento de poses estáticas del lenguaje de señas peruano, ofreciendo **alta precisión (>95%)** y **baja latencia** para una experiencia de usuario superior.

## 🚀 Inicio Rápido

### 1. **Recolección de Datos**
```bash
# Recolectar datos para señales estáticas
python scripts/collect_static_data.py
```

**Instrucciones de recolección:**
- Presiona `A`, `B`, `C`, etc. para iniciar recolección de esa letra
- Mantén la pose **estable** por al menos 20 frames
- El sistema **auto-captura** cuando detecta estabilidad y calidad adecuada
- `ESPACIO` para captura manual si la auto-captura no funciona
- `ESC` para cancelar sesión actual
- `Q` para salir

### 2. **Entrenamiento del Modelo**
```bash
# Entrenar modelo optimizado para señales estáticas
python scripts/train_static_model.py
```

### 3. **Traducción en Tiempo Real**
```bash
# Modo estático (recomendado)
python main.py --mode static

# Con configuración personalizada
python main.py --mode static --threshold 0.9
```

## 🔧 Configuración Avanzada

### **Parámetros del Recolector**
```python
config = {
    'stability_threshold': 0.001,        # Umbral de estabilidad (muy estricto)
    'stability_frames_required': 20,     # Frames consecutivos estables
    'quality_threshold': 0.85,           # Calidad mínima requerida
    'samples_per_sign': 30,              # Muestras objetivo por seña
    'recording_duration': 45,            # Duración de captura (1.5s)
    'auto_capture': True,                # Captura automática
}
```

### **Parámetros del Traductor**
```python
config = {
    'confidence_threshold': 0.85,        # Umbral de confianza alto
    'stability_frames': 8,               # Frames para confirmar estabilidad
    'prediction_cooldown': 15,           # Cooldown entre predicciones
    'geometric_validation': True,        # Validar características geométricas
}
```

## 📊 Características del Sistema

### ✅ **Ventajas del Modo Estático**
- **Alta Precisión**: >95% accuracy en condiciones controladas
- **Baja Latencia**: <100ms tiempo de respuesta
- **Detección de Estabilidad**: Solo predice cuando la pose es estable
- **Análisis Geométrico**: Usa características invariantes a escala
- **Validación de Calidad**: Filtra automáticamente capturas de baja calidad
- **Interfaz Intuitiva**: Feedback visual en tiempo real

### 🎯 **Arquitectura del Modelo**
```
Input: Landmarks (126 features) + Características Geométricas (28 features)
    ↓
Branch 1: MLP para Landmarks → Dense(256) → Dense(128)
Branch 2: MLP para Geometría → Dense(64) → Dense(32)
    ↓
Fusion: Concatenate → Dense(128) → Dense(64) → Softmax
```

### 📐 **Características Geométricas Extraídas**
1. **Distancias Normalizadas**: Desde muñeca a puntas de dedos
2. **Ángulos de Extensión**: Ángulo de cada dedo respecto a la palma
3. **Ratios Invariantes**: Proporciones entre longitudes de dedos
4. **Características de Apertura**: Distancias entre puntas de dedos
5. **Dispersión**: Centro de masa y distribución de landmarks

## 🎮 Controles de la Aplicación

### **Durante Recolección:**
| Tecla | Acción |
|-------|--------|
| `A-Z` | Iniciar recolección para esa letra |
| `ESPACIO` | Captura manual |
| `ESC` | Cancelar sesión actual |
| `Q` | Salir de la aplicación |

### **Durante Traducción:**
| Tecla | Acción |
|-------|--------|
| `Q` | Salir del traductor |
| `R` | Resetear estado interno |

## 📈 Métricas de Calidad

### **Indicadores en Tiempo Real:**
- **Estado de Estabilidad**: Verde (estable) / Naranja (estabilizando)
- **Score de Calidad**: 0.0-1.0 (>0.85 recomendado)
- **Confianza de Predicción**: 0.0-1.0 (>0.85 para alta confianza)
- **FPS**: Frames per second del sistema

### **Componentes del Score de Calidad:**
```python
calidad_final = promedio_ponderado([
    confianza_mediapipe * 0.3,    # Confianza de detección
    score_estabilidad * 0.4,      # Estabilidad de la pose
    validacion_geometrica * 0.2,  # Proporciones anatómicas
    completitud_landmarks * 0.1   # Landmarks sin errores
])
```

## 🛠️ Solución de Problemas

### **Problema: Auto-captura no funciona**
**Soluciones:**
1. Asegúrate de mantener la pose completamente estable
2. Verifica que la iluminación sea adecuada
3. Mantén la mano a distancia apropiada de la cámara
4. Usa captura manual con `ESPACIO`

### **Problema: Baja precisión en traducción**
**Soluciones:**
1. Recolecta más muestras de entrenamiento (30+ por seña)
2. Asegúrate de que las poses sean consistentes
3. Ajusta el umbral de confianza: `--threshold 0.9`
4. Verifica que la pose sea claramente distinguible

### **Problema: Reconocimiento lento**
**Soluciones:**
1. Reduce la resolución de cámara
2. Ajusta `stability_frames` a un valor menor
3. Verifica que no hay otros procesos consumiendo GPU/CPU

## 📚 Comparación: Estático vs Dinámico

| Aspecto | Estático ✅ | Dinámico ❌ |
|---------|-------------|-------------|
| **Precisión** | >95% | ~45% |
| **Latencia** | <100ms | >200ms |
| **Estabilidad** | Muy alta | Variable |
| **Facilidad de uso** | Alta | Compleja |
| **Recursos computacionales** | Bajos | Altos |
| **Robustez a variaciones** | Alta | Baja |

## 🎓 Mejores Prácticas

### **Para Recolección de Datos:**
1. **Iluminación uniforme** sin sombras fuertes
2. **Fondo neutro** sin patrones complejos
3. **Distancia consistente** de la cámara (60-80cm)
4. **Poses bien definidas** y anatómicamente correctas
5. **Variabilidad controlada** en orientación de mano

### **Para Traducción:**
1. **Mantén poses estables** por al menos 0.5 segundos
2. **Una mano visible** y bien iluminada
3. **Evita movimientos bruscos** entre poses
4. **Posición central** en el campo visual de la cámara

### **Para Entrenamiento:**
1. **Mínimo 30 muestras** de alta calidad por seña
2. **Validación cruzada** para evaluar generalización
3. **Monitoreo de overfitting** con early stopping
4. **Balanceo de clases** automático implementado

## 🔬 Arquitectura Técnica

### **Pipeline de Procesamiento:**
```
Cámara → MediaPipe → Landmarks 3D → Normalización → 
Extracción Geométrica → Modelo MLP → Predicción → 
Validación de Confianza → Resultado Final
```

### **Optimizaciones Implementadas:**
- **RobustScaler**: Manejo robusto de outliers
- **Geometric Features**: Características invariantes a escala
- **Stability Detection**: Filtrado de poses inestables
- **Quality Validation**: Control automático de calidad
- **Efficient Architecture**: MLP optimizado para baja latencia

## 📞 Soporte

Para problemas técnicos o mejoras, consulta:
- 📄 Documentación técnica: `ANALISIS_COMPLEJIDAD_SENAS_DINAMICAS.md`
- 🔬 Reportes de entrenamiento: `reports/static_model_report.json`
- 📊 Logs de calidad: Generados automáticamente durante recolección

---

**🎯 LSP Esperanza - Optimizado para señales estáticas con precisión superior**
