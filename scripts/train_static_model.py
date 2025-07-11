#!/usr/bin/env python3
"""
Entrenador especializado para señales estáticas de lenguaje de señas
Optimizado para alta precisión en reconocimiento de poses estáticas
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                   LayerNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
try:
    from src.utils.common import ensure_directories
except ImportError:
    # Fallback para cuando se ejecuta directamente
    def ensure_directories():
        pass

class StaticSignTrainer:
    """Entrenador especializado para señales estáticas con características geométricas"""
    
    def __init__(self, data_path='data/sequences'):
        self.data_path = Path(data_path)
        self.models_dir = Path('models')
        self.reports_dir = Path('reports')
        
        # Configuración específica para señales estáticas
        self.config = {
            'use_single_frame': True,           # Usar solo un frame representativo
            'geometric_features': True,        # Activar extracción geométrica
            'stability_check': True,          # Verificar estabilidad de la pose
            'confidence_threshold': 0.85,     # Umbral alto para estáticas
            'augmentation': False,            # Sin augmentación temporal
            'normalization': 'robust',        # RobustScaler para outliers
            'architecture': 'geometric_mlp'   # MLP optimizado para geometría
        }
        
        # Crear directorios necesarios
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎯 Entrenador de Señales Estáticas Inicializado")
        print(f"📁 Datos: {self.data_path}")
        print(f"⚙️ Configuración: {self.config}")

    def load_static_data(self):
        """Carga y procesa datos existentes específicamente para señales estáticas"""
        print("\n📊 CARGANDO DATOS EXISTENTES PARA ENTRENAMIENTO ESTÁTICO")
        print("="*50)
        
        sequences = []
        labels = []
        quality_scores = []
        
        # Obtener todas las clases disponibles
        sign_folders = [f for f in self.data_path.iterdir() 
                       if f.is_dir() and not f.name.startswith('.')]
        
        print(f"🔍 Encontradas {len(sign_folders)} clases de señas")
        
        # Estadísticas de carga
        total_files = 0
        loaded_files = 0
        discarded_files = 0
        
        for sign_folder in sorted(sign_folders):  # Ordenar para consistencia
            sign_name = sign_folder.name
            sign_files = list(sign_folder.glob('*.npy'))
            
            print(f"📝 Procesando {sign_name}: {len(sign_files)} archivos")
            total_files += len(sign_files)
            
            sign_loaded = 0
            sign_discarded = 0
            
            for file_path in sign_files:
                try:
                    # Cargar secuencia completa
                    sequence = np.load(file_path)
                    
                    # Validar dimensiones
                    if len(sequence.shape) != 2:
                        print(f"⚠️ Formato incorrecto {file_path.name}: shape {sequence.shape}")
                        sign_discarded += 1
                        continue
                    
                    # Verificar que tiene al menos 126 features (landmarks completos)
                    if sequence.shape[1] < 126:
                        print(f"⚠️ Features incompletas {file_path.name}: {sequence.shape[1]} features")
                        sign_discarded += 1
                        continue
                    
                    # Extraer frame más estable (para señales estáticas)
                    stable_frame, quality = self.extract_most_stable_frame(sequence)
                    
                    # Filtrar por calidad (más permisivo para datos existentes)
                    if quality > 0.5:  # Umbral más bajo para aprovechar datos existentes
                        sequences.append(stable_frame)
                        labels.append(sign_name)
                        quality_scores.append(quality)
                        sign_loaded += 1
                        loaded_files += 1
                    else:
                        print(f"⚠️ Calidad baja {file_path.name}: {quality:.3f}")
                        sign_discarded += 1
                        
                except Exception as e:
                    print(f"❌ Error cargando {file_path.name}: {e}")
                    sign_discarded += 1
            
            discarded_files += sign_discarded
            
            # Mostrar estadísticas por clase
            if sign_loaded > 0:
                avg_quality = np.mean([q for i, q in enumerate(quality_scores) 
                                     if labels[i] == sign_name])
                print(f"   ✅ {sign_name}: {sign_loaded} muestras cargadas (calidad promedio: {avg_quality:.3f})")
            
            if sign_discarded > 0:
                print(f"   ⚠️ {sign_name}: {sign_discarded} muestras descartadas")
        
        print(f"\n📊 RESUMEN DE CARGA:")
        print(f"   Total archivos encontrados: {total_files}")
        print(f"   Muestras cargadas exitosamente: {loaded_files}")
        print(f"   Muestras descartadas: {discarded_files}")
        print(f"   Tasa de éxito: {(loaded_files/total_files)*100:.1f}%")
        print(f"   Calidad promedio general: {np.mean(quality_scores):.3f}")
        
        # Mostrar distribución final por clase
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n📈 DISTRIBUCIÓN FINAL:")
        
        min_samples = np.min(counts)
        max_samples = np.max(counts)
        
        for label, count in zip(unique_labels, counts):
            status = "✅" if count >= 5 else "⚠️"
            print(f"   {status} {label}: {count} muestras")
        
        print(f"\n📋 Estadísticas del dataset:")
        print(f"   Clases: {len(unique_labels)}")
        print(f"   Total muestras: {len(sequences)}")
        print(f"   Muestras por clase - Min: {min_samples}, Max: {max_samples}, Promedio: {np.mean(counts):.1f}")
        
        # Verificar balance del dataset
        if max_samples / min_samples > 3:
            print(f"⚠️ Dataset desbalanceado detectado (ratio {max_samples/min_samples:.1f}:1)")
            print("   Considera recolectar más datos para las clases con pocas muestras")
        else:
            print("✅ Dataset razonablemente balanceado")
        
        return np.array(sequences), np.array(labels), np.array(quality_scores)

    def extract_most_stable_frame(self, sequence):
        """
        Extrae el frame más estable de una secuencia para señales estáticas.
        Optimizado para formato (60, 126) - 60 frames con 126 características.
        """
        if len(sequence) == 1:
            return sequence[0], 1.0
        
        if len(sequence) < 3:
            # Si hay pocos frames, usar el del medio
            return sequence[len(sequence)//2], 0.8
        
        # Estrategia mejorada para señales estáticas
        frame_scores = []
        
        for i, frame in enumerate(sequence):
            score = 0.0
            weight_count = 0
            
            # 1. Estabilidad respecto al promedio de la secuencia
            sequence_mean = np.mean(sequence, axis=0)
            distance_to_mean = np.linalg.norm(frame - sequence_mean)
            stability_score = 1.0 / (1.0 + distance_to_mean * 0.1)
            score += stability_score
            weight_count += 1
            
            # 2. Estabilidad local (comparación con vecinos)
            if i > 0 and i < len(sequence) - 1:
                prev_diff = np.linalg.norm(frame - sequence[i-1])
                next_diff = np.linalg.norm(frame - sequence[i+1])
                local_stability = 2.0 / (1.0 + prev_diff + next_diff)
                score += local_stability
                weight_count += 1
            
            # 3. Bonus por posición central (las señales estáticas suelen estar mejor en el centro)
            center_position = len(sequence) / 2
            position_bonus = 1.0 - abs(i - center_position) / center_position
            score += position_bonus * 0.5
            weight_count += 0.5
            
            # 4. Verificar que no tenga valores extremos (outliers)
            frame_std = np.std(frame)
            if frame_std < np.std(sequence_mean) * 2:  # No muy diferente del patrón general
                outlier_penalty = 1.0
            else:
                outlier_penalty = 0.7
            
            score *= outlier_penalty
            
            # Promedio ponderado
            final_score = score / weight_count
            frame_scores.append(final_score)
        
        # Seleccionar el mejor frame
        best_idx = np.argmax(frame_scores)
        best_frame = sequence[best_idx]
        quality_score = frame_scores[best_idx]
        
        # Normalizar quality score al rango [0, 1]
        quality_score = min(quality_score, 1.0)
        
        return best_frame, quality_score

    def extract_geometric_features(self, landmarks):
        """
        Extrae características geométricas específicas para señales estáticas.
        Adaptado para el formato de 126 características existente.
        """
        try:
            # Convertir a numpy array si no lo es
            landmarks = np.array(landmarks).flatten()
            
            # Asegurar que tenemos exactamente 126 características
            if len(landmarks) != 126:
                print(f"⚠️ Formato inesperado: {len(landmarks)} features, esperado 126")
                # Padding o truncate según sea necesario
                if len(landmarks) < 126:
                    landmarks = np.pad(landmarks, (0, 126 - len(landmarks)), 'constant')
                else:
                    landmarks = landmarks[:126]
            
            # Asumir estructura: 2 manos × 21 landmarks × 3 coords = 126
            # Mano 1: landmarks[0:63], Mano 2: landmarks[63:126]
            hand1 = landmarks[:63].reshape(-1, 3) if len(landmarks) >= 63 else None
            hand2 = landmarks[63:126].reshape(-1, 3) if len(landmarks) >= 126 else None
            
            features = []
            
            # Procesar cada mano por separado
            for hand_idx, hand in enumerate([hand1, hand2]):
                if hand is None or hand.shape[0] < 21:
                    # Si no hay datos de mano, agregar features dummy
                    features.extend([0.0] * 11)  # 11 features por mano
                    continue
                
                try:
                    # Landmarks clave (índices estándar de MediaPipe)
                    wrist = hand[0]
                    thumb_tip = hand[4] if len(hand) > 4 else hand[0]
                    index_tip = hand[8] if len(hand) > 8 else hand[0]
                    middle_tip = hand[12] if len(hand) > 12 else hand[0]
                    ring_tip = hand[16] if len(hand) > 16 else hand[0]
                    pinky_tip = hand[20] if len(hand) > 20 else hand[0]
                    
                    # 1. Distancias desde muñeca a puntas de dedos
                    distances = [
                        np.linalg.norm(thumb_tip - wrist),
                        np.linalg.norm(index_tip - wrist),
                        np.linalg.norm(middle_tip - wrist),
                        np.linalg.norm(ring_tip - wrist),
                        np.linalg.norm(pinky_tip - wrist)
                    ]
                    features.extend(distances)
                    
                    # 2. Dispersión de landmarks (compactitud de la mano)
                    center = np.mean(hand[:, :2], axis=0)  # Solo x,y
                    dispersions = [np.linalg.norm(landmark[:2] - center) for landmark in hand]
                    features.extend([
                        np.mean(dispersions),  # Dispersión promedio
                        np.std(dispersions),   # Variabilidad
                        np.max(dispersions),   # Extensión máxima
                    ])
                    
                    # 3. Características de apertura de mano
                    # Distancia entre pulgar e índice (indicador de apertura)
                    thumb_index_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
                    features.append(thumb_index_dist)
                    
                    # 4. Aspecto ratio de la mano (forma general)
                    x_coords = hand[:, 0]
                    y_coords = hand[:, 1]
                    width = np.max(x_coords) - np.min(x_coords)
                    height = np.max(y_coords) - np.min(y_coords)
                    aspect_ratio = width / (height + 1e-8)
                    features.append(aspect_ratio)
                    
                    # 5. Características de curvatura
                    # Usar landmarks de articulaciones medias para determinar flexión
                    if len(hand) >= 21:
                        finger_joints = [hand[3], hand[7], hand[11], hand[15], hand[19]]  # MCP joints
                        finger_tips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
                        
                        # Promedio de curvatura de dedos
                        curvatures = []
                        for joint, tip in zip(finger_joints, finger_tips):
                            curve = np.linalg.norm(tip - joint)
                            curvatures.append(curve)
                        
                        features.append(np.mean(curvatures))
                    else:
                        features.append(0.0)
                        
                except Exception as e:
                    print(f"⚠️ Error procesando mano {hand_idx}: {e}")
                    # En caso de error, llenar con zeros
                    features.extend([0.0] * 11)
            
            # Asegurar que siempre devolvemos exactamente 22 features (11 por mano)
            while len(features) < 22:
                features.append(0.0)
            features = features[:22]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"⚠️ Error general en extracción de características: {e}")
            # Fallback: devolver array de zeros
            return np.zeros(22, dtype=np.float32)

    def build_static_model(self, input_shape, geometric_features_shape, num_classes):
        """Construye modelo MLP optimizado para señales estáticas"""
        
        # Branch 1: Landmarks originales
        landmarks_input = Input(shape=(input_shape,), name='landmarks_input')
        landmarks_norm = LayerNormalization()(landmarks_input)
        
        landmarks_dense1 = Dense(256, activation='relu')(landmarks_norm)
        landmarks_dense1 = BatchNormalization()(landmarks_dense1)
        landmarks_dense1 = Dropout(0.3)(landmarks_dense1)
        
        landmarks_dense2 = Dense(128, activation='relu')(landmarks_dense1)
        landmarks_dense2 = BatchNormalization()(landmarks_dense2)
        landmarks_dense2 = Dropout(0.2)(landmarks_dense2)
        
        # Branch 2: Características geométricas
        geometric_input = Input(shape=(geometric_features_shape,), name='geometric_input')
        geometric_norm = LayerNormalization()(geometric_input)
        
        geometric_dense1 = Dense(64, activation='relu')(geometric_norm)
        geometric_dense1 = BatchNormalization()(geometric_dense1)
        geometric_dense1 = Dropout(0.2)(geometric_dense1)
        
        geometric_dense2 = Dense(32, activation='relu')(geometric_dense1)
        geometric_dense2 = BatchNormalization()(geometric_dense2)
        
        # Fusión de características
        merged = Concatenate()([landmarks_dense2, geometric_dense2])
        
        # Capas de clasificación
        final_dense1 = Dense(128, activation='relu')(merged)
        final_dense1 = BatchNormalization()(final_dense1)
        final_dense1 = Dropout(0.4)(final_dense1)
        
        final_dense2 = Dense(64, activation='relu')(final_dense1)
        final_dense2 = Dropout(0.3)(final_dense2)
        
        # Salida
        output = Dense(num_classes, activation='softmax', name='output')(final_dense2)
        
        # Crear modelo
        model = Model(inputs=[landmarks_input, geometric_input], outputs=output)
        
        # Compilar con optimizador específico para estáticas
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def train_static_model(self):
        """Entrena modelo específicamente optimizado para señales estáticas"""
        print("\n🚀 ENTRENANDO MODELO ESTÁTICO")
        print("="*50)
        
        # Cargar datos
        sequences, labels, quality_scores = self.load_static_data()
        
        # Extraer características geométricas
        print("🔧 Extrayendo características geométricas...")
        geometric_features = []
        for seq in sequences:
            try:
                geo_features = self.extract_geometric_features(seq)
                # Asegurar que sea un array 1D
                if isinstance(geo_features, (list, tuple)):
                    geo_features = np.array(geo_features).flatten()
                elif isinstance(geo_features, np.ndarray):
                    geo_features = geo_features.flatten()
                geometric_features.append(geo_features)
            except Exception as e:
                print(f"⚠️ Error extrayendo características: {e}")
                # Usar características dummy en caso de error
                geometric_features.append(np.zeros(22))  # 22 features geométricas por defecto
        
        geometric_features = np.array(geometric_features)
        
        print(f"📐 Características geométricas: {geometric_features.shape}")
        
        # Preparar labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        y = to_categorical(labels_encoded)
        
        # Normalización robusta
        scaler_landmarks = RobustScaler()
        scaler_geometric = RobustScaler()
        
        sequences_scaled = scaler_landmarks.fit_transform(sequences)
        geometric_scaled = scaler_geometric.fit_transform(geometric_features)
        
        # Split estratificado
        X_land_train, X_land_test, X_geo_train, X_geo_test, y_train, y_test = train_test_split(
            sequences_scaled, geometric_scaled, y, 
            test_size=0.2, stratify=labels_encoded, random_state=42
        )
        
        print(f"📊 Conjunto de entrenamiento: {len(X_land_train)} muestras")
        print(f"📊 Conjunto de prueba: {len(X_land_test)} muestras")
        
        # Construir modelo
        model = self.build_static_model(
            input_shape=sequences_scaled.shape[1],
            geometric_features_shape=geometric_scaled.shape[1],
            num_classes=len(label_encoder.classes_)
        )
        
        print("\n🏗️ Arquitectura del modelo:")
        model.summary()
        
        # Callbacks optimizados para estáticas
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.models_dir / 'static_model_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        print("\n🎯 Iniciando entrenamiento...")
        history = model.fit(
            [X_land_train, X_geo_train], y_train,
            validation_data=([X_land_test, X_geo_test], y_test),
            epochs=150,        # Más épocas para convergencia fina
            batch_size=32,     # Batch size moderado
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluación final
        print("\n📊 EVALUACIÓN FINAL")
        print("="*30)
        
        # Predicciones
        y_pred = model.predict([X_land_test, X_geo_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Accuracy general
        accuracy = np.mean(y_pred_classes == y_true_classes)
        print(f"🎯 Accuracy general: {accuracy:.4f}")
        
        # Reporte detallado
        report = classification_report(
            y_true_classes, y_pred_classes, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        print("\n📈 Métricas por clase:")
        # El classification_report devuelve un dict cuando output_dict=True
        if isinstance(report, dict):
            for class_name in label_encoder.classes_:
                if class_name in report:
                    class_metrics = report[class_name]
                    if isinstance(class_metrics, dict):
                        precision = class_metrics.get('precision', 0.0)
                        recall = class_metrics.get('recall', 0.0) 
                        f1_score = class_metrics.get('f1-score', 0.0)
                        print(f"   {class_name}: Precision={precision:.3f}, "
                              f"Recall={recall:.3f}, "
                              f"F1={f1_score:.3f}")
        else:
            # Fallback si el reporte no es un dict
            print("   Reporte detallado de clasificación:")
            print(report)
        
        # Guardar modelo y metadatos
        model.save(self.models_dir / 'sign_model_static.keras')
        np.save(self.models_dir / 'label_encoder_static.npy', label_encoder.classes_)
        
        # Guardar parámetros del scaler correctamente
        import pickle
        with open(self.models_dir / 'scaler_landmarks_static.pkl', 'wb') as f:
            pickle.dump(scaler_landmarks, f)
        with open(self.models_dir / 'scaler_geometric_static.pkl', 'wb') as f:
            pickle.dump(scaler_geometric, f)
        
        # Generar matriz de confusión
        self.plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_)
        
        # Guardar reporte
        import json
        report_data = {
            'model_type': 'static_mlp',
            'accuracy': float(accuracy),
            'classes': list(label_encoder.classes_),
            'classification_report': report,
            'training_samples': len(X_land_train),
            'test_samples': len(X_land_test),
            'features': {
                'landmarks': sequences_scaled.shape[1],
                'geometric': geometric_scaled.shape[1]
            }
        }
        
        with open(self.reports_dir / 'static_model_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✅ Modelo guardado en: {self.models_dir / 'sign_model_static.keras'}")
        print(f"✅ Reporte guardado en: {self.reports_dir / 'static_model_report.json'}")
        
        return model, history, report_data

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Genera matriz de confusión para análisis visual"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Matriz de Confusión - Modelo Estático')
        plt.xlabel('Predicción')
        plt.ylabel('Realidad')
        plt.tight_layout()
        
        plt.savefig(self.reports_dir / 'confusion_matrix_static.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matriz de confusión guardada en: {self.reports_dir / 'confusion_matrix_static.png'}")

if __name__ == "__main__":
    trainer = StaticSignTrainer()
    model, history, report = trainer.train_static_model()
    
    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"🎯 Accuracy final: {report['accuracy']:.4f}")
