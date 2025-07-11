# 🤟 LSP Esperanza - Sistema de Traducción de Lenguaje de Señas

[![Python](https://img.shields.io/badge/Python-3.12.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.1-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0.86-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-red.svg)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un sistema completo de traducción de lenguaje de señas en tiempo real usando **modelos bidireccionales GRU** con capacidades dinámicas avanzadas. Desarrollado como parte del proyecto **UPC Esperanza** para democratizar la comunicación a través de la tecnología.

## 🎯 Características Principales

### ✨ Modelo Bidireccional Dinámico
- **🧠 GRU Bidireccional**: Analiza secuencias en ambas direcciones temporales
- **📊 14 Características de Movimiento**: 6 básicas + 8 dinámicas avanzadas
- **🔄 Clasificación Híbrida**: Distingue automáticamente entre señas estáticas y dinámicas
- **🎯 Alta Precisión**: 88.8% en conjunto de prueba, 100% en señas dinámicas

### 🔄 Sistema de Augmentación Inteligente
- **🎨 Augmentación Específica**: Estrategias diferentes para cada tipo de seña
- **✅ Control de Calidad**: Validación automática de secuencias generadas
- **🔧 Transformaciones Avanzadas**: Noise, time warping, rotación, escalado
- **📋 Reportes Detallados**: Análisis completo del proceso

### 🚀 Traductor en Tiempo Real
- **👁️ Detección de Movimiento**: Análisis de patrones estáticos vs dinámicos
- **🎨 UI Avanzada**: Interfaz visual con indicadores de confianza
- **✋ Múltiples Manos**: Soporte para 1-2 manos simultáneamente
- **⚙️ Configuración Flexible**: Umbrales ajustables en tiempo real

## 📊 Datos Incluidos

Este repositorio incluye **datos reales de entrenamiento** y el **modelo entrenado**:

- **143 secuencias** de señas recolectadas
- **3 señas completamente funcionales**: A, B, J (dinámica)
- **Modelo bidireccional entrenado** (4.30 MB)
- **89 secuencias augmentadas** con 64.7% de calidad promedio

## 🚀 Inicio Rápido

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Jaed69/Salvacion.git
cd Salvacion
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Traductor
```bash
# Método 1: Script principal
python main.py

# Método 2: Con automatización (Windows)
.\Makefile.ps1 run

# Método 3: Con umbral personalizado
python main.py --threshold 0.9
```

### 4. Verificar el Sistema
```bash
python verify_setup.py
```

## 🏗️ Estructura del Proyecto

```
LSP-Esperanza/
├── 🎯 main.py                  # Script principal
├── 🔧 verify_setup.py          # Verificación del sistema
├── ⚡ Makefile.ps1             # Automatización (Windows)
├── 📚 README.md                # Este archivo
├── 📦 requirements.txt         # Dependencias
│
├── 📂 src/                     # Código fuente modular
│   ├── 📊 data_processing/     # Recolección y augmentación
│   ├── 🤖 models/              # Entrenamiento de modelos
│   ├── 🔄 translation/         # Traducción en tiempo real
│   └── 🛠️ utils/               # Utilidades comunes
│
├── ⚙️ config/                  # Configuración centralizada
├── 📊 data/sequences/          # Datos de entrenamiento
├── 🤖 models/                  # Modelos entrenados
├── 📝 scripts/                 # Scripts de automatización
├── 🧪 tests/                   # Tests y validaciones
├── 📚 docs/                    # Documentación detallada
└── 📋 reports/                 # Reportes y análisis
```

## 📋 Señas Soportadas

### 🤚 Señas Estáticas (24)
`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`

### 👋 Señas Dinámicas (5)
- **J**: Movimiento en gancho (100% precisión)
- **Z**: Trazado de letra
- **HOLA**: Saludo con movimiento
- **GRACIAS**: Gesto de agradecimiento
- **POR FAVOR**: Expresión de cortesía

## 🛠️ Tecnologías

| Componente | Versión | Propósito |
|------------|---------|-----------|
| **Python** | 3.12.11 | Lenguaje principal |
| **TensorFlow** | 2.18.1 | Deep Learning y GRU |
| **OpenCV** | 4.11.0.86 | Procesamiento de video |
| **MediaPipe** | 0.10.21 | Detección de landmarks |
| **NumPy** | 1.26.4 | Computación numérica |
| **SciPy** | 1.16.0 | Algoritmos científicos |

## 📈 Rendimiento

| Métrica | Valor |
|---------|-------|
| **Precisión General** | 88.8% |
| **Precisión Seña J** | 100% |
| **FPS en Tiempo Real** | ~30 |
| **Latencia** | <100ms |
| **Parámetros del Modelo** | 362,883 |
| **Características** | 14 features |

## 🎮 Controles de la Aplicación

Durante la ejecución del traductor:

| Tecla | Acción |
|-------|--------|
| `q` | Salir del programa |
| `r` | Resetear buffers |
| `t` | Alternar umbral (0.6 ↔ 0.9) |
| `d` | Información de debug |

## 🔧 Scripts de Automatización

### Windows (PowerShell)
```powershell
# Ver todos los comandos
.\Makefile.ps1 help

# Configurar proyecto
.\Makefile.ps1 setup

# Entrenar modelo
.\Makefile.ps1 train

# Recolectar datos
.\Makefile.ps1 collect-A

# Ver estadísticas
.\Makefile.ps1 stats
```

### Multiplataforma
```bash
# Entrenar modelo
python scripts/train_model.py --model-type bidirectional_dynamic

# Recolectar datos
python scripts/collect_data.py --sign A --samples 100

# Augmentar datos
python scripts/run_augmentation.py
```

## 📊 Archivos Incluidos

### 🤖 Modelos Entrenados
- `models/sign_model_bidirectional_dynamic.h5` - Modelo principal (4.30 MB)
- `models/label_encoder.npy` - Codificador de etiquetas

### 📊 Datos de Entrenamiento
- `data/sequences/A/` - 13 secuencias para seña A
- `data/sequences/B/` - 30 secuencias para seña B  
- `data/sequences/J/` - 100 secuencias para seña J (incluye augmentadas)

### 📋 Reportes
- Reportes de sesiones de entrenamiento
- Análisis de augmentación de datos
- Comparaciones visuales de rendimiento

## 🚀 Extensión del Sistema

### Agregar Nuevas Señas
```bash
# 1. Recolectar datos
python scripts/collect_data.py --sign NUEVA_SEÑA --samples 100

# 2. Augmentar si es necesario
python scripts/run_augmentation.py

# 3. Re-entrenar modelo
python scripts/train_model.py --epochs 100
```

### Mejorar el Modelo
- Agregar más características de movimiento
- Implementar arquitecturas más complejas
- Optimizar hiperparámetros

## 🤝 Contribución

1. **Fork** el proyecto
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **Comunidad de LSP**: Por proporcionar conocimiento sobre lenguaje de señas
- **Equipo UPC Esperanza**: Por el apoyo y recursos del proyecto
- **Desarrolladores Open Source**: Por las herramientas que hacen esto posible

## 📞 Contacto

- **Proyecto**: UPC Esperanza
- **Repositorio**: [https://github.com/Jaed69/Salvacion](https://github.com/Jaed69/Salvacion)
- **Documentación**: Ver carpeta `docs/` para guías detalladas

---

**🎉 ¡Democratizando la comunicación a través de la tecnología!** 🤟

> *"La tecnología debe servir para conectar, no para dividir. LSP Esperanza es nuestro aporte para un mundo más inclusivo."*
