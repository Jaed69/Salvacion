#!/usr/bin/env python3
# quick_fix_j.py
# Solución rápida para mejorar el reconocimiento de J

import os
import sys
sys.path.append('src')

from models.trainer import EnhancedSequenceModelTrainer

def apply_j_fix():
    """Aplicar mejoras específicas para el reconocimiento de J"""
    
    print("🔧 APLICANDO MEJORAS PARA RECONOCIMIENTO DE J")
    print("="*50)
    
    try:
        # Crear trainer con mejoras específicas para J
        trainer = EnhancedSequenceModelTrainer(
            data_path='data/sequences',
            model_type='bidirectional_dynamic'
        )
        
        print("✅ Trainer inicializado correctamente")
        
        # Cargar y preparar datos con énfasis en J
        print("\n📊 Preparando datos con énfasis en J...")
        
        X, motion_features, y = trainer.load_data()
        print(f"✅ Datos cargados: {X.shape[0]} secuencias")
        
        # Verificar que J está en los datos
        if 'J' in trainer.label_encoder.classes_:
            j_index = list(trainer.label_encoder.classes_).index('J')
            j_samples = sum(1 for label in trainer.label_encoder.transform(
                ['J'] * sum(1 for i, cls in enumerate(trainer.label_encoder.classes_) 
                          if cls == 'J' and i < len(y))
            ) if label == j_index)
            print(f"✅ Encontradas {j_samples} muestras de J")
        else:
            print("❌ No se encontró la letra J en los datos")
            return
        
        # Construir modelo optimizado para J
        print("\n🏗️ Construyendo modelo optimizado para J...")
        
        model = trainer.build_bidirectional_dynamic_model(
            sequence_shape=(trainer.sequence_length, trainer.num_features),
            motion_shape=motion_features.shape[1],
            num_classes=len(trainer.signs)
        )
        
        print("✅ Modelo construido correctamente")
        print(f"📊 Arquitectura: {len(trainer.signs)} clases")
        
        # Calcular pesos de clase para dar más importancia a J
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Obtener etiquetas originales
        labels = []
        for sign in trainer.signs:
            sign_path = f'data/sequences/{sign}'
            sign_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            labels.extend([sign] * len(sign_files))
        
        label_indices = trainer.label_encoder.transform(labels)
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(label_indices),
            y=label_indices
        )
        
        class_weight_dict = dict(enumerate(class_weights))
        
        # Dar peso extra a J
        if 'J' in trainer.label_encoder.classes_:
            j_idx = list(trainer.label_encoder.classes_).index('J')
            class_weight_dict[j_idx] = class_weight_dict[j_idx] * 2.0  # Peso doble para J
        
        print(f"⚖️ Pesos de clase calculados:")
        for i, sign in enumerate(trainer.label_encoder.classes_):
            print(f"   {sign}: {class_weight_dict.get(i, 1.0):.2f}")
        
        # Entrenar modelo con configuración optimizada para J
        print(f"\n🚀 Entrenando modelo con énfasis en J...")
        
        from sklearn.model_selection import train_test_split
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Convertir labels a formato numérico
        y_numeric = to_categorical(label_indices)
        
        # Split de datos
        X_train, X_test, motion_train, motion_test, y_train, y_test = train_test_split(
            X, motion_features, y_numeric, 
            test_size=0.2, 
            stratify=label_indices, 
            random_state=42
        )
        
        # Callbacks optimizados
        callbacks = [
            EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=20,  # Más paciencia para J
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar con parámetros optimizados para J
        history = model.fit(
            [X_train, motion_train], y_train,
            validation_data=([X_test, motion_test], y_test),
            epochs=150,  # Más epochs para mejor convergencia
            batch_size=8,   # Batch size más pequeño
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        
        # Evaluar específicamente en J
        print(f"\n📊 EVALUACIÓN ESPECÍFICA PARA J:")
        print("-" * 40)
        
        # Predicciones
        y_pred = model.predict([X_test, motion_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Métricas por clase
        for i, sign in enumerate(trainer.label_encoder.classes_):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                samples = np.sum(mask)
                print(f"   {sign}: {accuracy:.3f} ({samples} muestras de test)")
                
                if sign == 'J':
                    # Métricas detalladas para J
                    j_precision = np.sum((y_pred_classes == i) & (y_true_classes == i)) / max(1, np.sum(y_pred_classes == i))
                    j_recall = np.sum((y_pred_classes == i) & (y_true_classes == i)) / max(1, np.sum(mask))
                    j_f1 = 2 * (j_precision * j_recall) / max(1e-8, j_precision + j_recall)
                    
                    print(f"\n🎯 MÉTRICAS DETALLADAS PARA J:")
                    print(f"   Accuracy: {accuracy:.3f}")
                    print(f"   Precision: {j_precision:.3f}")
                    print(f"   Recall: {j_recall:.3f}")
                    print(f"   F1-Score: {j_f1:.3f}")
        
        # Guardar modelo mejorado
        os.makedirs('models', exist_ok=True)
        model.save('models/sign_model_j_improved.keras')
        np.save('models/label_encoder_j_improved.npy', trainer.label_encoder.classes_)
        
        print(f"\n✅ MODELO MEJORADO GUARDADO:")
        print(f"   Modelo: models/sign_model_j_improved.keras")
        print(f"   Labels: models/label_encoder_j_improved.npy")
        
        # Recomendaciones adicionales
        print(f"\n💡 RECOMENDACIONES ADICIONALES:")
        print("-" * 40)
        print("1. Usar el modelo mejorado en el traductor:")
        print("   - Cambiar model_path a 'models/sign_model_j_improved.keras'")
        print("   - Cambiar signs_path a 'models/label_encoder_j_improved.npy'")
        print()
        print("2. Si J sigue sin reconocerse bien:")
        print("   - Recolectar más datos de J con mejor calidad")
        print("   - Hacer la seña J más lentamente y con movimientos más amplios")
        print("   - Asegurar buena iluminación al grabar J")
        print()
        print("3. Probar el reconocimiento:")
        print("   python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la mejora: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = apply_j_fix()
    if success:
        print(f"\n🎉 ¡Mejoras aplicadas exitosamente!")
        print(f"   El modelo debería reconocer mejor la letra J ahora.")
    else:
        print(f"\n❌ Hubo problemas aplicando las mejoras.")
        print(f"   Revisar los errores mostrados arriba.")
