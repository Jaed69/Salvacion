#!/usr/bin/env python3
"""
Entrenador mejorado para señales estáticas de lenguaje de señas
Con técnicas avanzadas de augmentación y arquitectura robusta
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from collections import Counter
import pickle

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class ImprovedStaticSignTrainer:
    """Entrenador mejorado para señales estáticas con augmentación avanzada"""
    
    def __init__(self, data_path='data/sequences', augmentation_factor=5):
        self.data_path = Path(data_path)
        self.models_dir = Path('models')
        self.reports_dir = Path('reports')
        self.augmentation_factor = augmentation_factor
        
        # Configuración mejorada
        self.config = {
            'use_single_frame': True,
            'geometric_features': True,
            'advanced_augmentation': True,     # Activar augmentación avanzada
            'stability_check': True,
            'confidence_threshold': 0.75,     # Reducido para más flexibilidad
            'normalization': 'robust',
            'architecture': 'attention_mlp',  # Arquitectura con atención
            'class_balancing': True,          # Balanceo de clases
            'ensemble_training': True,        # Entrenamiento por ensamble
            'cross_validation': True,         # Validación cruzada
            'dropout_rate': 0.3,
            'l1_l2_reg': (1e-5, 1e-4),
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 150,
            'patience': 20
        }
        
        # Crear directorios
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        print("🚀 Entrenador Mejorado Inicializado")
        print(f"📈 Factor de augmentación: {augmentation_factor}x")
        print(f"⚙️ Configuración: {self.config}")

    def load_and_preprocess_data(self):
        """Carga y preprocesa datos con augmentación avanzada"""
        
        print("\n📂 CARGANDO DATOS EXISTENTES...")
        
        sequences = []
        labels = []
        quality_scores = []
        
        total_files = 0
        loaded_files = 0
        discarded_files = 0
        
        # Procesar cada carpeta de letra
        for sign_folder in sorted(self.data_path.iterdir()):
            if not sign_folder.is_dir():
                continue
                
            sign_name = sign_folder.name
            sign_files = list(sign_folder.glob('*.npy'))
            total_files += len(sign_files)
            
            sign_loaded = 0
            sign_discarded = 0
            
            print(f"\n🔤 Procesando {sign_name} ({len(sign_files)} archivos)...")
            
            for file_path in sign_files:
                try:
                    # Cargar secuencia
                    sequence = np.load(file_path)
                    
                    # Verificar dimensiones
                    if sequence.shape[1] < 126:
                        print(f"⚠️ Features incompletas {file_path.name}: {sequence.shape[1]} features")
                        sign_discarded += 1
                        continue
                    
                    # Extraer frame más estable
                    stable_frame, quality = self.extract_most_stable_frame(sequence)
                    
                    # Umbral más permisivo para datos existentes
                    if quality > 0.3:  # Muy permisivo para aprovechar todos los datos
                        sequences.append(stable_frame)
                        labels.append(sign_name)
                        quality_scores.append(quality)
                        sign_loaded += 1
                        loaded_files += 1
                    else:
                        sign_discarded += 1
                        
                except Exception as e:
                    print(f"❌ Error cargando {file_path.name}: {e}")
                    sign_discarded += 1
            
            discarded_files += sign_discarded
            
            if sign_loaded > 0:
                avg_quality = np.mean([q for i, q in enumerate(quality_scores) 
                                     if labels[i] == sign_name])
                print(f"   ✅ {sign_name}: {sign_loaded} muestras cargadas (calidad promedio: {avg_quality:.3f})")
        
        print(f"\n📊 RESUMEN DE CARGA INICIAL:")
        print(f"   Muestras originales cargadas: {loaded_files}")
        print(f"   Muestras descartadas: {discarded_files}")
        print(f"   Calidad promedio: {np.mean(quality_scores):.3f}")
        
        # Convertir a arrays
        X_original = np.array(sequences)
        y_original = np.array(labels)
        
        # Mostrar distribución original
        unique_labels, counts = np.unique(y_original, return_counts=True)
        print(f"\n📈 DISTRIBUCIÓN ORIGINAL:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count} muestras")
        
        # Aplicar augmentación avanzada
        print(f"\n🔄 APLICANDO AUGMENTACIÓN AVANZADA (Factor {self.augmentation_factor}x)...")
        X_augmented, y_augmented = self.apply_advanced_augmentation(X_original, y_original)
        
        print(f"✅ Dataset final: {len(X_augmented)} muestras ({len(X_augmented)/len(X_original):.1f}x)")
        
        return X_augmented, y_augmented

    def extract_most_stable_frame(self, sequence):
        """Extrae el frame más estable de una secuencia"""
        
        if len(sequence) == 1:
            return sequence[0], 1.0
        
        # Calcular variabilidad frame por frame
        stabilities = []
        
        for i in range(len(sequence)):
            # Comparar con frames vecinos
            if i == 0:
                # Primer frame: comparar solo con el siguiente
                diff = np.linalg.norm(sequence[i] - sequence[i+1]) if len(sequence) > 1 else 0
            elif i == len(sequence) - 1:
                # Último frame: comparar solo con el anterior
                diff = np.linalg.norm(sequence[i] - sequence[i-1])
            else:
                # Frame medio: comparar con anterior y siguiente
                diff_prev = np.linalg.norm(sequence[i] - sequence[i-1])
                diff_next = np.linalg.norm(sequence[i] - sequence[i+1])
                diff = (diff_prev + diff_next) / 2
            
            # La estabilidad es inversamente proporcional al cambio
            stability = 1.0 / (1.0 + diff)
            stabilities.append(stability)
        
        # Seleccionar el frame más estable
        best_idx = np.argmax(stabilities)
        best_frame = sequence[best_idx]
        quality_score = stabilities[best_idx]
        
        return best_frame, quality_score

    def apply_advanced_augmentation(self, X, y):
        """Aplica técnicas avanzadas de augmentación para señales estáticas"""
        
        print("🔄 Aplicando augmentación avanzada...")
        
        augmented_X = []
        augmented_y = []
        
        # Mantener datos originales
        augmented_X.extend(X)
        augmented_y.extend(y)
        
        # Calcular cuántos samples generar por clase para balancear
        unique_labels, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        
        for label in unique_labels:
            # Obtener muestras de esta clase
            label_indices = np.where(y == label)[0]
            label_samples = X[label_indices]
            current_count = len(label_samples)
            
            # Calcular cuántas muestras adicionales necesitamos
            target_count = max_count * self.augmentation_factor
            additional_needed = max(0, target_count - current_count)
            
            print(f"   {label}: {current_count} -> {target_count} muestras (+{additional_needed})")
            
            # Generar muestras adicionales
            for _ in range(additional_needed):
                # Seleccionar una muestra base aleatoria
                base_idx = np.random.choice(len(label_samples))
                base_sample = label_samples[base_idx].copy()
                
                # Aplicar transformaciones aleatorias
                augmented_sample = self.apply_static_transformations(base_sample)
                
                augmented_X.append(augmented_sample)
                augmented_y.append(label)
        
        # Convertir a arrays y mezclar
        X_aug = np.array(augmented_X)
        y_aug = np.array(augmented_y)
        
        # Mezclar el dataset
        indices = np.random.permutation(len(X_aug))
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]
        
        # Mostrar estadísticas finales
        unique_labels, counts = np.unique(y_aug, return_counts=True)
        print(f"\n📈 DISTRIBUCIÓN DESPUÉS DE AUGMENTACIÓN:")
        for label, count in zip(unique_labels, counts):
            print(f"   {label}: {count} muestras")
        
        return X_aug, y_aug

    def apply_static_transformations(self, landmarks):
        """Aplica transformaciones específicas para señales estáticas"""
        
        # Separar landmarks de las dos manos (126 = 63 + 63)
        left_hand = landmarks[:63].reshape(21, 3)
        right_hand = landmarks[63:].reshape(21, 3)
        
        # Aplicar transformaciones a cada mano
        left_hand_aug = self.transform_hand_landmarks(left_hand)
        right_hand_aug = self.transform_hand_landmarks(right_hand)
        
        # Recombinar
        augmented_landmarks = np.concatenate([
            left_hand_aug.flatten(),
            right_hand_aug.flatten()
        ])
        
        return augmented_landmarks

    def transform_hand_landmarks(self, hand_landmarks):
        """Aplica transformaciones geométricas a landmarks de una mano"""
        
        augmented = hand_landmarks.copy()
        
        # 1. Rotación ligera (±5 grados)
        if np.random.random() < 0.7:
            angle = np.random.uniform(-5, 5) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotar en el plano XY
            x_rot = augmented[:, 0] * cos_a - augmented[:, 1] * sin_a
            y_rot = augmented[:, 0] * sin_a + augmented[:, 1] * cos_a
            
            augmented[:, 0] = x_rot
            augmented[:, 1] = y_rot
        
        # 2. Escalado ligero (±10%)
        if np.random.random() < 0.6:
            scale_factor = np.random.uniform(0.9, 1.1)
            
            # Encontrar centro de la mano
            center = np.mean(augmented, axis=0)
            
            # Escalar respecto al centro
            augmented = center + (augmented - center) * scale_factor
        
        # 3. Traslación muy ligera (±2%)
        if np.random.random() < 0.5:
            translation = np.random.uniform(-0.02, 0.02, size=3)
            augmented += translation
        
        # 4. Ruido gaussiano muy ligero
        if np.random.random() < 0.8:
            noise = np.random.normal(0, 0.005, augmented.shape)
            augmented += noise
        
        # 5. Deformación ligera de dedos individuales
        if np.random.random() < 0.4:
            # Seleccionar un dedo al azar (índices 1-20, excluyendo muñeca)
            finger_start = np.random.choice([1, 5, 9, 13, 17])  # Inicio de cada dedo
            finger_indices = range(finger_start, min(finger_start + 4, 21))
            
            # Aplicar deformación muy ligera
            for idx in finger_indices:
                deformation = np.random.normal(0, 0.003, 3)
                augmented[idx] += deformation
        
        return augmented

    def extract_geometric_features(self, landmarks):
        """Extrae características geométricas para ambas manos"""
        
        # Dividir en mano izquierda (primeros 63) y mano derecha (siguientes 63)
        left_hand_landmarks = landmarks[:63].reshape(21, 3)
        right_hand_landmarks = landmarks[63:].reshape(21, 3)
        
        all_features = []
        
        # Extraer características para cada mano
        for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
            features = []
            
            # Puntos clave
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            wrist = hand_landmarks[0]
            
            # 1. Distancias desde muñeca (5 features)
            distances_from_wrist = [
                np.linalg.norm(thumb_tip - wrist),
                np.linalg.norm(index_tip - wrist),
                np.linalg.norm(middle_tip - wrist),
                np.linalg.norm(ring_tip - wrist),
                np.linalg.norm(pinky_tip - wrist)
            ]
            features.extend(distances_from_wrist)
            
            # 2. Ángulos entre dedos (5 features)
            finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            finger_bases = [hand_landmarks[2], hand_landmarks[5], hand_landmarks[9], 
                           hand_landmarks[13], hand_landmarks[17]]
            
            for i in range(len(finger_tips)):
                finger_vector = finger_tips[i] - finger_bases[i]
                wrist_vector = finger_bases[i] - wrist
                
                cos_angle = np.dot(finger_vector, wrist_vector) / (
                    np.linalg.norm(finger_vector) * np.linalg.norm(wrist_vector) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features.append(angle)
            
            # 3. Distancia thumb-index (característica muy discriminativa) (1 feature)
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            features.append(thumb_index_dist)
            
            all_features.extend(features)  # 11 features por mano
        
        return np.array(all_features)  # Total: 22 features

    def create_improved_model(self, input_landmarks_shape, input_geometric_shape, num_classes):
        """Crea un modelo mejorado con arquitectura de atención"""
        
        print(f"🏗️ Creando modelo mejorado...")
        print(f"   Landmarks input: {input_landmarks_shape}")
        print(f"   Geometric input: {input_geometric_shape}")
        print(f"   Clases: {num_classes}")
        
        # Imports locales para Keras
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                           Concatenate, Add, Multiply, Activation)
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        
        # Inputs
        landmarks_input = Input(shape=(input_landmarks_shape,), name='landmarks')
        geometric_input = Input(shape=(input_geometric_shape,), name='geometric')
        
        # Branch para landmarks con atención
        x1 = Dense(256, activation='relu', name='landmarks_dense1')(landmarks_input)
        x1 = BatchNormalization(name='landmarks_bn1')(x1)
        x1 = Dropout(self.config['dropout_rate'], name='landmarks_dropout1')(x1)
        
        x1 = Dense(128, activation='relu', name='landmarks_dense2')(x1)
        x1 = BatchNormalization(name='landmarks_bn2')(x1)
        x1 = Dropout(self.config['dropout_rate'], name='landmarks_dropout2')(x1)
        
        # Mecanismo de atención para landmarks
        attention_weights = Dense(128, activation='softmax', name='attention_weights')(x1)
        x1_attended = Multiply(name='landmarks_attention')([x1, attention_weights])
        
        x1 = Dense(64, activation='relu', name='landmarks_dense3')(x1_attended)
        x1 = BatchNormalization(name='landmarks_bn3')(x1)
        
        # Branch para características geométricas
        x2 = Dense(64, activation='relu', name='geometric_dense1')(geometric_input)
        x2 = BatchNormalization(name='geometric_bn1')(x2)
        x2 = Dropout(self.config['dropout_rate'] * 0.5, name='geometric_dropout1')(x2)
        
        x2 = Dense(32, activation='relu', name='geometric_dense2')(x2)
        x2 = BatchNormalization(name='geometric_bn2')(x2)
        
        # Fusión con atención cruzada
        combined = Concatenate(name='fusion')([x1, x2])
        
        # Capas de fusión con conexiones residuales
        x = Dense(128, activation='relu', name='fusion_dense1',
                 kernel_regularizer=l1_l2(*self.config['l1_l2_reg']))(combined)
        x = BatchNormalization(name='fusion_bn1')(x)
        x = Dropout(self.config['dropout_rate'], name='fusion_dropout1')(x)
        
        # Conexión residual
        x_skip = Dense(128, activation='linear', name='skip_connection')(combined)
        x = Add(name='residual_add')([x, x_skip])
        x = Activation('relu')(x)  # Usar Activation en lugar de tf.nn.relu
        
        x = Dense(64, activation='relu', name='fusion_dense2',
                 kernel_regularizer=l1_l2(*self.config['l1_l2_reg']))(x)
        x = BatchNormalization(name='fusion_bn2')(x)
        x = Dropout(self.config['dropout_rate'], name='fusion_dropout2')(x)
        
        # Capa de salida
        output = Dense(num_classes, activation='softmax', name='classification')(x)
        
        # Crear modelo
        model = Model(inputs=[landmarks_input, geometric_input], outputs=output)
        
        # Compilar con optimizador mejorado
        optimizer = Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo creado exitosamente")
        print(f"📊 Parámetros totales: {model.count_params():,}")
        
        return model

    def train_improved_model(self, X, y):
        """Entrena el modelo mejorado con validación cruzada"""
        
        print("\n🎯 INICIANDO ENTRENAMIENTO MEJORADO...")
        
        # Imports locales para Keras
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Preparar datos
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        num_classes = len(label_encoder.classes_)
        print(f"📊 Clases a entrenar: {num_classes}")
        print(f"📊 Nombres de clases: {list(label_encoder.classes_)}")
        
        # Normalizar landmarks
        print("🔄 Normalizando datos...")
        scaler_landmarks = RobustScaler()
        X_landmarks_scaled = scaler_landmarks.fit_transform(X)
        
        # Extraer y normalizar características geométricas
        print("🔄 Extrayendo características geométricas...")
        X_geometric = np.array([self.extract_geometric_features(sample) for sample in X])
        scaler_geometric = RobustScaler()
        X_geometric_scaled = scaler_geometric.fit_transform(X_geometric)
        
        print(f"📐 Landmarks shape: {X_landmarks_scaled.shape}")
        print(f"📐 Geometric shape: {X_geometric_scaled.shape}")
        
        # Calcular pesos de clase para balanceo
        class_weights = None
        if self.config['class_balancing']:
            class_weights_array = compute_class_weight(
                'balanced', 
                classes=np.unique(y_encoded), 
                y=y_encoded
            )
            class_weights = dict(enumerate(class_weights_array))
            print(f"⚖️ Pesos de clase calculados: {class_weights}")
        
        # Inicializar variable cv_scores
        cv_scores = []
        
        # Configurar validación cruzada
        if self.config['cross_validation']:
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_models = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_landmarks_scaled, y_encoded)):
                print(f"\n🔄 Entrenando fold {fold + 1}/5...")
                
                # Dividir datos para este fold
                X_landmarks_train = X_landmarks_scaled[train_idx]
                X_landmarks_val = X_landmarks_scaled[val_idx]
                X_geometric_train = X_geometric_scaled[train_idx]
                X_geometric_val = X_geometric_scaled[val_idx]
                y_train = y_categorical[train_idx]
                y_val = y_categorical[val_idx]
                
                # Crear modelo para este fold
                model = self.create_improved_model(
                    X_landmarks_scaled.shape[1],
                    X_geometric_scaled.shape[1],
                    num_classes
                )
                
                # Callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_accuracy',
                        patience=self.config['patience'],
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6,
                        verbose=1
                    )
                ]
                
                # Entrenar
                history = model.fit(
                    [X_landmarks_train, X_geometric_train],
                    y_train,
                    validation_data=([X_landmarks_val, X_geometric_val], y_val),
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1 if fold == 0 else 0  # Solo mostrar progreso del primer fold
                )
                
                # Evaluar fold
                val_score = model.evaluate(
                    [X_landmarks_val, X_geometric_val], 
                    y_val, 
                    verbose=0
                )[1]  # accuracy
                cv_scores.append(val_score)
                cv_models.append(model)
                
                print(f"✅ Fold {fold + 1} completado - Accuracy: {val_score:.4f}")
            
            # Seleccionar mejor modelo
            best_fold = np.argmax(cv_scores)
            best_model = cv_models[best_fold]
            
            print(f"\n🏆 RESULTADOS DE VALIDACIÓN CRUZADA:")
            print(f"   Accuracy promedio: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"   Mejor fold: {best_fold + 1} (Accuracy: {cv_scores[best_fold]:.4f})")
            
        else:
            # Entrenamiento simple sin validación cruzada
            X_train_landmarks, X_test_landmarks, y_train, y_test = train_test_split(
                X_landmarks_scaled, y_categorical, 
                test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            X_train_geometric, X_test_geometric, _, _ = train_test_split(
                X_geometric_scaled, y_categorical, 
                test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            best_model = self.create_improved_model(
                X_landmarks_scaled.shape[1],
                X_geometric_scaled.shape[1],
                num_classes
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=self.config['patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Entrenar modelo
            history = best_model.fit(
                [X_train_landmarks, X_train_geometric],
                y_train,
                validation_data=([X_test_landmarks, X_test_geometric], y_test),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        # Evaluación final en todos los datos
        print("\n📊 EVALUACIÓN FINAL...")
        final_predictions = best_model.predict([X_landmarks_scaled, X_geometric_scaled])
        final_predicted_classes = np.argmax(final_predictions, axis=1)
        final_accuracy = accuracy_score(y_encoded, final_predicted_classes)
        
        print(f"🎯 Accuracy final en todo el dataset: {final_accuracy:.4f}")
        
        # Reporte de clasificación
        print("\n📋 REPORTE DE CLASIFICACIÓN:")
        classification_rep = classification_report(
            y_encoded, 
            final_predicted_classes,
            target_names=label_encoder.classes_,
            zero_division=0
        )
        print(classification_rep)
        
        # Guardar modelo y metadatos
        print("\n💾 GUARDANDO MODELO MEJORADO...")
        
        # Guardar modelo
        model_path = self.models_dir / 'sign_model_static_improved.keras'
        best_model.save(model_path)
        print(f"✅ Modelo guardado: {model_path}")
        
        # Guardar encoder de etiquetas
        labels_path = self.models_dir / 'label_encoder_static_improved.npy'
        np.save(labels_path, label_encoder.classes_)
        print(f"✅ Etiquetas guardadas: {labels_path}")
        
        # Guardar scalers
        with open(self.models_dir / 'scaler_landmarks_static_improved.pkl', 'wb') as f:
            pickle.dump(scaler_landmarks, f)
        
        with open(self.models_dir / 'scaler_geometric_static_improved.pkl', 'wb') as f:
            pickle.dump(scaler_geometric, f)
        
        print("✅ Scalers guardados")
        
        # Guardar reporte
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report = {
            'timestamp': timestamp,
            'config': self.config,
            'final_accuracy': float(final_accuracy),
            'classification_report': classification_rep,
            'cv_scores': cv_scores,
            'num_classes': num_classes,
            'classes': list(label_encoder.classes_),
            'total_samples': len(X),
            'augmentation_factor': self.augmentation_factor
        }
        
        report_path = self.reports_dir / f'training_report_improved_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Reporte guardado: {report_path}")
        
        return best_model, label_encoder, scaler_landmarks, scaler_geometric

def main():
    """Función principal para entrenar el modelo mejorado"""
    
    print("🚀 ENTRENADOR MEJORADO DE SEÑALES ESTÁTICAS")
    print("=" * 60)
    
    # Crear entrenador con augmentación agresiva
    trainer = ImprovedStaticSignTrainer(augmentation_factor=8)  # 8x más datos
    
    # Cargar y preprocesar datos
    X, y = trainer.load_and_preprocess_data()
    
    if len(X) == 0:
        print("❌ No se encontraron datos para entrenar")
        return
    
    # Entrenar modelo mejorado
    trainer.train_improved_model(X, y)
    
    print("\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("📁 Archivos generados:")
    print("   - models/sign_model_static_improved.keras")
    print("   - models/label_encoder_static_improved.npy")
    print("   - models/scaler_landmarks_static_improved.pkl")
    print("   - models/scaler_geometric_static_improved.pkl")
    print("   - reports/training_report_improved_*.json")

if __name__ == "__main__":
    main()
