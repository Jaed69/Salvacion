#!/usr/bin/env python3
# static_signs_trainer.py
# Entrenador simplificado para señas estáticas A y B

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class StaticSignsTrainer:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.sequence_length = 50
        self.num_features = 126
        
        # Solo usar señas estáticas A y B
        self.static_signs = ['A', 'B']
        self.label_encoder = LabelEncoder()
        
        print(f"🔍 Entrenador para señas estáticas: {', '.join(self.static_signs)}")

    def load_static_data(self):
        """Cargar solo datos de señas estáticas A y B"""
        print("\n📊 CARGANDO DATOS DE SEÑAS ESTÁTICAS")
        print("="*50)
        
        sequences = []
        labels = []
        
        for sign in self.static_signs:
            sign_path = f'{self.data_path}/{sign}'
            if not os.path.exists(sign_path):
                print(f"⚠️ No se encontró directorio para {sign}")
                continue
                
            sign_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            print(f"📁 {sign}: {len(sign_files)} archivos")
            
            for file in sign_files:
                seq = np.load(f'{sign_path}/{file}')
                
                # Para señas estáticas, tomar el frame más estable (promedio de frames del medio)
                # Esto elimina el ruido del inicio y final del movimiento
                start_frame = len(seq) // 4
                end_frame = 3 * len(seq) // 4
                stable_frames = seq[start_frame:end_frame]
                
                # Usar el promedio de los frames estables como característica
                if len(stable_frames) > 0:
                    static_features = np.mean(stable_frames, axis=0)
                else:
                    static_features = np.mean(seq, axis=0)
                
                sequences.append(static_features)
                labels.append(sign)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        print(f"\n📈 Resumen de datos:")
        for sign in self.static_signs:
            count = np.sum(labels == sign)
            print(f"   {sign}: {count} muestras")
        
        return sequences, labels

    def create_static_features(self, sequences):
        """Crear características específicas para señas estáticas"""
        print("\n🔧 EXTRAYENDO CARACTERÍSTICAS ESTÁTICAS")
        print("="*50)
        
        static_features = []
        
        for seq in sequences:
            # Para señas estáticas, las características importantes son:
            # 1. Posiciones de landmarks de mano
            hand_coords = seq[-63:]  # Últimos 63 features son de la mano
            
            # 2. Posiciones relativas entre dedos (ángulos y distancias)
            # Extraer coordenadas X, Y, Z
            x_coords = hand_coords[0::3]  # Cada 3er elemento desde 0
            y_coords = hand_coords[1::3]  # Cada 3er elemento desde 1
            z_coords = hand_coords[2::3]  # Cada 3er elemento desde 2
            
            # 3. Características geométricas estáticas
            features = []
            
            # Centroide de la mano
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            centroid_z = np.mean(z_coords)
            features.extend([centroid_x, centroid_y, centroid_z])
            
            # Dispersión de los puntos (qué tan extendida está la mano)
            spread_x = np.std(x_coords)
            spread_y = np.std(y_coords)
            spread_z = np.std(z_coords)
            features.extend([spread_x, spread_y, spread_z])
            
            # Distancias desde cada dedo al centroide (forma de la mano)
            for i in range(21):  # 21 landmarks de mano
                dist_to_centroid = np.sqrt(
                    (x_coords[i] - centroid_x)**2 + 
                    (y_coords[i] - centroid_y)**2 + 
                    (z_coords[i] - centroid_z)**2
                )
                features.append(dist_to_centroid)
            
            # Ángulos entre dedos (importantes para A vs B)
            # Pulgar (landmarks 1-4), índice (5-8), medio (9-12), anular (13-16), meñique (17-20)
            finger_tips = [4, 8, 12, 16, 20]  # Tips de cada dedo
            finger_angles = []
            
            for i in range(len(finger_tips)):
                for j in range(i+1, len(finger_tips)):
                    tip1_idx = finger_tips[i]
                    tip2_idx = finger_tips[j]
                    
                    # Vector del centroide a cada punta
                    vec1 = np.array([x_coords[tip1_idx] - centroid_x, 
                                   y_coords[tip1_idx] - centroid_y])
                    vec2 = np.array([x_coords[tip2_idx] - centroid_x, 
                                   y_coords[tip2_idx] - centroid_y])
                    
                    # Ángulo entre vectores
                    dot_product = np.dot(vec1, vec2)
                    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    if norms > 1e-8:
                        angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                        finger_angles.append(angle)
                    else:
                        finger_angles.append(0)
            
            features.extend(finger_angles)
            
            # Características específicas para distinguir A vs B
            # A: puño cerrado con pulgar hacia el lado
            # B: mano abierta con dedos juntos
            
            # Apertura de la mano (distancia promedio entre dedos consecutivos)
            finger_distances = []
            for i in range(len(finger_tips)-1):
                tip1_idx = finger_tips[i]
                tip2_idx = finger_tips[i+1]
                
                dist = np.sqrt(
                    (x_coords[tip1_idx] - x_coords[tip2_idx])**2 + 
                    (y_coords[tip1_idx] - y_coords[tip2_idx])**2
                )
                finger_distances.append(dist)
            
            avg_finger_distance = np.mean(finger_distances)
            features.append(avg_finger_distance)
            
            # Posición del pulgar relativa a los otros dedos
            thumb_tip = 4
            other_fingers = [8, 12, 16, 20]
            
            thumb_isolation = 0
            for finger_tip in other_fingers:
                dist = np.sqrt(
                    (x_coords[thumb_tip] - x_coords[finger_tip])**2 + 
                    (y_coords[thumb_tip] - y_coords[finger_tip])**2
                )
                thumb_isolation += dist
            
            thumb_isolation /= len(other_fingers)
            features.append(thumb_isolation)
            
            static_features.append(features)
        
        static_features = np.array(static_features)
        print(f"✅ Extraídas {static_features.shape[1]} características por muestra")
        
        return static_features

    def build_static_model(self, input_shape, num_classes):
        """Modelo simple y efectivo para señas estáticas"""
        print(f"\n🏗️ CONSTRUYENDO MODELO ESTÁTICO")
        print("="*40)
        
        model = Sequential([
            # Entrada
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Capa oculta 1
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            # Capa oculta 2
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Capa de salida
            Dense(num_classes, activation='softmax')
        ])
        
        # Compilar con parámetros optimizados para clasificación estática
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        print(f"✅ Modelo creado con {model.count_params()} parámetros")
        model.summary()
        
        return model

    def train_static_model(self):
        """Entrenar modelo para señas estáticas"""
        print("\n🚀 ENTRENANDO MODELO ESTÁTICO")
        print("="*40)
        
        # Cargar datos
        sequences, labels = self.load_static_data()
        
        if len(sequences) == 0:
            print("❌ No se encontraron datos para entrenar")
            return None, None
        
        # Crear características estáticas
        features = self.create_static_features(sequences)
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Codificar etiquetas
        labels_encoded = self.label_encoder.fit_transform(labels)
        y = to_categorical(labels_encoded)
        
        print(f"\n📊 Datos de entrenamiento:")
        print(f"   Características: {features_scaled.shape}")
        print(f"   Etiquetas: {y.shape}")
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, y, 
            test_size=0.2, 
            stratify=labels_encoded, 
            random_state=42
        )
        
        # Construir modelo
        model = self.build_static_model(
            input_shape=features_scaled.shape[1],
            num_classes=len(self.static_signs)
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=20,
                restore_best_weights=True,
                min_delta=0.01
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
        print(f"\n🎯 Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar
        print(f"\n📊 EVALUACIÓN FINAL:")
        print("="*30)
        
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Accuracy por clase
        for i, sign in enumerate(self.label_encoder.classes_):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                samples = np.sum(mask)
                print(f"   {sign}: {accuracy:.3f} ({samples} muestras de test)")
        
        # Accuracy general
        overall_accuracy = np.mean(y_pred_classes == y_true_classes)
        print(f"\n🎯 Accuracy general: {overall_accuracy:.3f}")
        
        # Guardar modelo
        os.makedirs('models', exist_ok=True)
        model.save('models/static_signs_model.keras')
        np.save('models/static_signs_encoder.npy', self.label_encoder.classes_)
        
        # Guardar scaler para usar en predicción
        import joblib
        joblib.dump(scaler, 'models/static_signs_scaler.pkl')
        
        print(f"\n✅ MODELO GUARDADO:")
        print(f"   Modelo: models/static_signs_model.keras")
        print(f"   Encoder: models/static_signs_encoder.npy")
        print(f"   Scaler: models/static_signs_scaler.pkl")
        
        return model, history

if __name__ == "__main__":
    # Entrenar modelo estático
    trainer = StaticSignsTrainer()
    model, history = trainer.train_static_model()
    
    if model is not None:
        print(f"\n🎉 ¡Entrenamiento completado!")
        print(f"   El modelo está optimizado para reconocer señas estáticas A y B")
        print(f"   Usa características geométricas estables sin dependencia del movimiento")
    else:
        print(f"\n❌ No se pudo entrenar el modelo")
