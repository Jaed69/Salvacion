# LSP Esperanza - Sistema de Traducción de Lenguaje de Señas

Un sistema completo de traducción de lenguaje de señas en tiempo real usando modelos bidireccionales GRU con capacidades dinámicas avanzadas.

## 🏗️ Estructura del Proyecto

```
LSP-Esperanza/
├── 📂 src/                          # Código fuente principal
│   ├── 📂 data_processing/          # Procesamiento y augmentación de datos
│   │   ├── collector.py             # Recolección de datos de señas
│   │   ├── augmentation_engine.py   # Motor de augmentación inteligente
│   │   └── __init__.py
│   ├── 📂 models/                   # Modelos de machine learning
│   │   ├── trainer.py               # Entrenamiento de modelos
│   │   └── __init__.py
│   ├── 📂 translation/              # Traducción en tiempo real
│   │   ├── real_time_translator.py  # Traductor bidireccional
│   │   └── __init__.py
│   ├── 📂 utils/                    # Utilidades auxiliares
│   │   └── __init__.py
│   └── __init__.py
├── 📂 scripts/                      # Scripts de automatización
│   ├── collect_data.py              # Script para recolectar datos
│   ├── train_model.py               # Script para entrenar modelos
│   ├── run_augmentation.py          # Script de augmentación
│   └── demo_augmentation.py         # Demo de augmentación
├── 📂 config/                       # Configuración del proyecto
│   └── settings.py                  # Configuración centralizada
├── 📂 data/                         # Datos de entrenamiento
│   └── sequences/                   # Secuencias de señas organizadas
│       ├── A/                       # Seña A
│       ├── B/                       # Seña B
│       └── J/                       # Seña J (dinámica)
├── 📂 models/                       # Modelos entrenados
│   ├── sign_model_bidirectional_dynamic.h5
│   └── label_encoder.npy
├── 📂 tests/                        # Tests y validaciones
│   └── test_translator.py
├── 📂 docs/                         # Documentación
│   ├── README_augmentation.md
│   └── README_traductor_bidireccional.md
├── 📂 reports/                      # Reportes y análisis
│   ├── session_report_*.json
│   ├── augmentation_report_*.json
│   └── augmentation_comparison_*.png
├── 📄 main.py                       # Script principal
├── 📄 requirements.txt              # Dependencias del proyecto
└── 📄 README.md                     # Este archivo
```

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Traductor

```bash
# Ejecutar con configuración por defecto
python main.py

# Ejecutar con umbral personalizado
python main.py --threshold 0.9

# Ejecutar con modelo personalizado
python main.py --model models/mi_modelo.h5
```

### 3. Recolectar Nuevos Datos

```bash
# Recolectar datos para la seña 'A'
python scripts/collect_data.py --sign A --samples 100

# Recolectar datos para señas dinámicas
python scripts/collect_data.py --sign J --samples 150
```

### 4. Entrenar Modelo

```bash
# Entrenar modelo bidireccional dinámico
python scripts/train_model.py --model-type bidirectional_dynamic --epochs 100

# Entrenar con configuración personalizada
python scripts/train_model.py --epochs 150 --batch-size 64
```

### 5. Augmentar Datos

```bash
# Ejecutar augmentación para todas las señas
python scripts/run_augmentation.py

# Demo de augmentación
python scripts/demo_augmentation.py
```

## 🎯 Características Principales

### ✅ Modelo Bidireccional Dinámico
- **GRU Bidireccional**: Analiza secuencias en ambas direcciones
- **14 Características de Movimiento**: 6 básicas + 8 dinámicas avanzadas
- **Clasificación Híbrida**: Distingue entre señas estáticas y dinámicas
- **Precisión**: 88.8% en conjunto de prueba

### ✅ Sistema de Augmentación Inteligente
- **Augmentación Específica**: Estrategias diferentes para cada tipo de seña
- **Control de Calidad**: Validación automática de secuencias generadas
- **Transformaciones Avanzadas**: Noise, time warping, rotación, escalado
- **Reportes Detallados**: Análisis completo del proceso de augmentación

### ✅ Traductor en Tiempo Real
- **Detección de Movimiento**: Análisis de patrones estáticos vs dinámicos
- **UI Avanzada**: Interfaz visual con indicadores de confianza
- **Múltiples Manos**: Soporte para 1-2 manos simultáneamente
- **Configuración Flexible**: Umbrales ajustables en tiempo real

## 📊 Señas Soportadas

### Señas Estáticas
`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`

### Señas Dinámicas
- **J**: Movimiento en gancho
- **Z**: Trazado de letra
- **HOLA**: Saludo con movimiento
- **GRACIAS**: Gesto de agradecimiento
- **POR FAVOR**: Expresión de cortesía

## 🔧 Configuración

Todas las configuraciones se encuentran centralizadas en `config/settings.py`:

- **Modelos**: Configuración de arquitecturas y parámetros
- **Cámara**: Resolución, FPS, dispositivo
- **MediaPipe**: Umbrales de detección y tracking
- **Movimiento**: Análisis de patrones dinámicos
- **UI**: Colores y elementos visuales

## 🧪 Testing

```bash
# Ejecutar test del traductor
python tests/test_translator.py

# Validar configuración del sistema
python -c "import src.translation.real_time_translator; print('✅ Sistema OK')"
```

## 📈 Métricas de Rendimiento

| Componente | Métrica | Valor |
|------------|---------|-------|
| Modelo Bidireccional | Precisión Test | 88.8% |
| Seña Dinámica J | Precisión | 100% |
| Augmentación | Calidad Promedio | 64.7% |
| Secuencias Generadas | Total | 89 |
| Tiempo Real | FPS | ~30 |
| Latencia | Predicción | <100ms |

## 🔄 Flujo de Trabajo

1. **Recolección**: `scripts/collect_data.py` → `data/sequences/`
2. **Augmentación**: `scripts/run_augmentation.py` → Incrementa dataset
3. **Entrenamiento**: `scripts/train_model.py` → `models/`
4. **Traducción**: `main.py` → Uso en tiempo real

## 🛠️ Tecnologías Utilizadas

- **TensorFlow 2.18.1**: Deep Learning y GRU bidireccional
- **OpenCV 4.11.0.86**: Procesamiento de video y UI
- **MediaPipe 0.10.21**: Detección de landmarks de manos
- **NumPy & SciPy**: Procesamiento numérico y análisis
- **Python 3.12.11**: Lenguaje principal

## 📋 Comandos Útiles

```bash
# Ver ayuda de cualquier script
python main.py --help
python scripts/train_model.py --help
python scripts/collect_data.py --help

# Modo verbose para debugging
python main.py --verbose

# Ajustar umbrales en tiempo real
# Presiona 't' durante la ejecución del traductor

# Reset de buffers
# Presiona 'r' durante la ejecución del traductor
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Comunidad de desarrolladores de lenguaje de señas
- Equipo de investigación UPC Esperanza
- Colaboradores del proyecto

---

**Proyecto LSP Esperanza** - Democratizando la comunicación através de la tecnología 🤟
