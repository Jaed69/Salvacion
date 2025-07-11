# Makefile.ps1 - Automatización de tareas para LSP Esperanza
# Uso: .\Makefile.ps1 <comando>

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

# Colores para output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    Write-Host "$Color$Message$Reset"
}

function Show-Help {
    Write-ColorOutput $Blue "🔧 LSP ESPERANZA - COMANDOS DISPONIBLES"
    Write-Host "=" * 60
    Write-Host "📊 DATOS:"
    Write-Host "  collect-A        Recolectar datos para seña A"
    Write-Host "  collect-J        Recolectar datos para seña J (dinámica)"
    Write-Host "  augment          Ejecutar augmentación de datos"
    Write-Host "  demo-augment     Demo de augmentación"
    Write-Host ""
    Write-Host "🤖 MODELO:"
    Write-Host "  train            Entrenar modelo bidireccional"
    Write-Host "  train-fast       Entrenamiento rápido (50 epochs)"
    Write-Host "  validate-model   Validar archivos del modelo"
    Write-Host ""
    Write-Host "🚀 EJECUCIÓN:"
    Write-Host "  run              Ejecutar traductor principal"
    Write-Host "  run-strict       Ejecutar con umbral alto (0.9)"
    Write-Host "  test             Ejecutar tests"
    Write-Host ""
    Write-Host "🔧 UTILIDADES:"
    Write-Host "  setup            Configurar proyecto"
    Write-Host "  clean            Limpiar archivos temporales"
    Write-Host "  stats            Mostrar estadísticas del proyecto"
    Write-Host "  deps             Instalar dependencias"
    Write-Host "  help             Mostrar esta ayuda"
    Write-Host ""
    Write-Host "🌐 GIT & GITHUB:"
    Write-Host "  git-init         Inicializar repositorio Git"
    Write-Host "  git-status       Ver estado del repositorio"
    Write-Host "  git-commit       Crear commit con mensaje personalizado"
    Write-Host "  git-push         Subir cambios a GitHub"
    Write-Host "  deploy           Desplegar completo a GitHub"
    Write-Host "=" * 60
}

function Install-Dependencies {
    Write-ColorOutput $Yellow "📦 Instalando dependencias..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput $Green "✅ Dependencias instaladas correctamente"
    } else {
        Write-ColorOutput $Red "❌ Error instalando dependencias"
    }
}

function Setup-Project {
    Write-ColorOutput $Yellow "🔧 Configurando proyecto LSP Esperanza..."
    
    # Crear directorios necesarios
    $directories = @("data/sequences", "models", "reports", "docs")
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput $Green "✅ Creado directorio: $dir"
        }
    }
    
    # Verificar Python
    python --version
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput $Red "❌ Python no encontrado"
        return
    }
    
    # Instalar dependencias
    Install-Dependencies
    
    # Mostrar información del sistema
    python src/utils/common.py
    
    Write-ColorOutput $Green "✅ Proyecto configurado correctamente"
}

function Clean-Project {
    Write-ColorOutput $Yellow "🧹 Limpiando archivos temporales..."
    
    # Limpiar cache de Python
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force
    
    # Limpiar logs antiguos (más de 7 días)
    Get-ChildItem -Path "reports" -Name "*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force
    
    Write-ColorOutput $Green "✅ Limpieza completada"
}

function Show-Stats {
    Write-ColorOutput $Blue "📊 Mostrando estadísticas del proyecto..."
    python src/utils/common.py
}

function Validate-Model {
    Write-ColorOutput $Yellow "🔍 Validando archivos del modelo..."
    python -c "
from src.utils.common import validate_model_files
import json
result = validate_model_files()
print('✅ Modelo válido:' if result['valid'] else '❌ Modelo inválido:')
print(json.dumps(result, indent=2, ensure_ascii=False))
"
}

function Run-Translator {
    param([string]$Threshold = "0.8")
    Write-ColorOutput $Yellow "🚀 Ejecutando traductor LSP Esperanza..."
    python main.py --threshold $Threshold
}

function Train-Model {
    param([int]$Epochs = 100)
    Write-ColorOutput $Yellow "🎯 Entrenando modelo bidireccional ($Epochs epochs)..."
    python scripts/train_model.py --model-type bidirectional_dynamic --epochs $Epochs
}

function Collect-Data {
    param([string]$Sign, [int]$Samples = 100)
    Write-ColorOutput $Yellow "📊 Recolectando datos para seña '$Sign' ($Samples muestras)..."
    python scripts/collect_data.py --sign $Sign --samples $Samples
}

function Run-Tests {
    Write-ColorOutput $Yellow "🧪 Ejecutando tests..."
    python tests/test_translator.py
}

function Run-Augmentation {
    Write-ColorOutput $Yellow "🔄 Ejecutando augmentación de datos..."
    python scripts/run_augmentation.py
}

function Demo-Augmentation {
    Write-ColorOutput $Yellow "🎬 Ejecutando demo de augmentación..."
    python scripts/demo_augmentation.py
}

function Initialize-Git {
    Write-ColorOutput $Yellow "🔧 Inicializando repositorio Git..."
    
    # Verificar si Git está instalado
    git --version
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput $Red "❌ Git no está instalado"
        return
    }
    
    # Inicializar repositorio si no existe
    if (!(Test-Path ".git")) {
        git init
        Write-ColorOutput $Green "✅ Repositorio Git inicializado"
    } else {
        Write-ColorOutput $Yellow "⚠️  Repositorio Git ya existe"
    }
    
    # Configurar usuario si no está configurado
    $userName = git config user.name
    $userEmail = git config user.email
    
    if (!$userName) {
        Write-ColorOutput $Yellow "📝 Configurando usuario Git..."
        git config user.name "LSP Esperanza"
        git config user.email "lsp.esperanza@example.com"
    }
    
    Write-ColorOutput $Green "✅ Git configurado correctamente"
}

function Add-RemoteOrigin {
    param([string]$RepoUrl = "https://github.com/Jaed69/Salvacion.git")
    
    Write-ColorOutput $Yellow "🔗 Configurando repositorio remoto..."
    
    # Verificar si el remoto ya existe
    $existingRemote = git remote get-url origin 2>$null
    
    if ($existingRemote) {
        Write-ColorOutput $Yellow "⚠️  Remoto 'origin' ya existe: $existingRemote"
        Write-ColorOutput $Yellow "🔄 Actualizando URL del remoto..."
        git remote set-url origin $RepoUrl
    } else {
        git remote add origin $RepoUrl
        Write-ColorOutput $Green "✅ Remoto 'origin' agregado: $RepoUrl"
    }
}

function Commit-All {
    param([string]$Message = "🚀 LSP Esperanza - Proyecto reorganizado con modelo y datos")
    
    Write-ColorOutput $Yellow "📝 Creando commit con todos los archivos..."
    
    # Verificar estado del repositorio
    $status = git status --porcelain
    if (!$status) {
        Write-ColorOutput $Yellow "⚠️  No hay cambios para commitear"
        return
    }
    
    # Agregar todos los archivos
    git add .
    
    # Mostrar archivos agregados
    Write-ColorOutput $Blue "📁 Archivos agregados:"
    git status --short
    
    # Crear commit
    git commit -m $Message
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput $Green "✅ Commit creado exitosamente"
    } else {
        Write-ColorOutput $Red "❌ Error creando commit"
    }
}

function Push-To-GitHub {
    param([string]$Branch = "main")
    
    Write-ColorOutput $Yellow "🚀 Subiendo a GitHub..."
    
    # Verificar si la rama existe localmente
    $currentBranch = git branch --show-current
    if ($currentBranch -ne $Branch) {
        Write-ColorOutput $Yellow "🔄 Cambiando a rama $Branch..."
        
        # Verificar si la rama existe
        $branchExists = git branch --list $Branch
        if (!$branchExists) {
            git checkout -b $Branch
            Write-ColorOutput $Green "✅ Rama $Branch creada"
        } else {
            git checkout $Branch
        }
    }
    
    # Push con configuración de upstream
    git push -u origin $Branch
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput $Green "✅ Proyecto subido exitosamente a GitHub"
        Write-ColorOutput $Blue "🌐 URL: https://github.com/Jaed69/Salvacion"
    } else {
        Write-ColorOutput $Red "❌ Error subiendo a GitHub"
        Write-ColorOutput $Yellow "💡 Verifica tus credenciales de GitHub"
    }
}

function Deploy-To-GitHub {
    Write-ColorOutput $Blue "🌟 DESPLEGANDO LSP ESPERANZA A GITHUB"
    Write-Host "=" * 60
    
    # Paso 1: Inicializar Git
    Initialize-Git
    
    # Paso 2: Configurar remoto
    Add-RemoteOrigin
    
    # Paso 3: Commit todos los archivos
    Commit-All
    
    # Paso 4: Push a GitHub
    Push-To-GitHub
    
    Write-Host "=" * 60
    Write-ColorOutput $Green "🎉 ¡PROYECTO DESPLEGADO EXITOSAMENTE!"
    Write-ColorOutput $Blue "🌐 Repositorio: https://github.com/Jaed69/Salvacion"
    Write-ColorOutput $Yellow "💡 El proyecto incluye:"
    Write-Host "   📊 Datos de entrenamiento (143 secuencias)"
    Write-Host "   🤖 Modelo bidireccional entrenado (4.30 MB)"
    Write-Host "   📝 Código fuente completo y organizado"
    Write-Host "   📚 Documentación detallada"
    Write-Host "   🔧 Scripts de automatización"
}

function Show-Git-Status {
    Write-ColorOutput $Blue "📊 Estado del repositorio Git..."
    
    if (!(Test-Path ".git")) {
        Write-ColorOutput $Red "❌ No es un repositorio Git"
        Write-ColorOutput $Yellow "💡 Usa '.\Makefile.ps1 git-init' para inicializar"
        return
    }
    
    # Estado general
    git status
    
    # Información del remoto
    $remote = git remote get-url origin 2>$null
    if ($remote) {
        Write-ColorOutput $Green "🔗 Remoto configurado: $remote"
    } else {
        Write-ColorOutput $Yellow "⚠️  Sin remoto configurado"
    }
    
    # Último commit
    $lastCommit = git log -1 --oneline 2>$null
    if ($lastCommit) {
        Write-ColorOutput $Blue "📝 Último commit: $lastCommit"
    }
}
# Ejecutar comando
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "setup" { Setup-Project }
    "deps" { Install-Dependencies }
    "clean" { Clean-Project }
    "stats" { Show-Stats }
    "validate-model" { Validate-Model }
    "run" { Run-Translator }
    "run-strict" { Run-Translator -Threshold "0.9" }
    "train" { Train-Model }
    "train-fast" { Train-Model -Epochs 50 }
    "collect-a" { Collect-Data -Sign "A" }
    "collect-j" { Collect-Data -Sign "J" }
    "test" { Run-Tests }
    "augment" { Run-Augmentation }
    "demo-augment" { Demo-Augmentation }
    "git-init" { Initialize-Git }
    "git-status" { Show-Git-Status }
    "git-commit" { 
        $msg = Read-Host "💬 Mensaje del commit"
        if ($msg) { Commit-All -Message $msg } else { Commit-All }
    }
    "git-push" { Push-To-GitHub }
    "deploy" { Deploy-To-GitHub }
    default {
        Write-ColorOutput $Red "❌ Comando desconocido: $Command"
        Write-ColorOutput $Yellow "💡 Usa '.\Makefile.ps1 help' para ver comandos disponibles"
    }
}
