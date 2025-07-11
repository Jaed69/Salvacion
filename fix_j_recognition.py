#!/usr/bin/env python3
# fix_j_recognition.py
# Script para diagnosticar y mejorar el reconocimiento de la letra J

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, LSTM, GRU, Bidirectional, Dense, 
                                   Dropout, Conv1D, MaxPooling1D, Concatenate, 
                                   BatchNormalization, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class JRecognitionFixer:
    def __init__(self, data_path='data/sequences'):
        self.data_path = data_path
        self.sequence_length = 50
        self.num_features = 126
        
        # Cargar datos
        self.signs = np.array([name for name in os.listdir(data_path) 
                              if os.path.isdir(os.path.join(data_path, name))])
        self.label_encoder = LabelEncoder()
        
        print(f"🔍 Encontradas {len(self.signs)} señas: {', '.join(self.signs)}")

    def analyze_j_characteristics(self):
        """Análisis detallado de las características específicas de J"""
        print("\n📊 ANÁLISIS DETALLADO DE LA LETRA J")
        print("="*50)
        
        j_files = [f for f in os.listdir(f'{self.data_path}/J') if f.endswith('.npy')]
        j_sequences = []
        
        movement_patterns = []
        curvature_patterns = []
        direction_patterns = []
        
        for file in j_files:
            seq = np.load(f'{self.data_path}/J/{file}')
            j_sequences.append(seq)
            
            # Análisis de patrón de movimiento específico de J
            hand_coords = seq[:, -63:]  # Solo coordenadas de mano
            
            # 1. Patrón de curvatura (J tiene una curva característica)
            # Tomar primeros 21 puntos (x, y, z de cada landmark)
            x_coords = hand_coords[:, 0::3]  # Coordenadas X
            y_coords = hand_coords[:, 1::3]  # Coordenadas Y
            
            # Calcular curvatura promedio en X e Y
            x_curvature = self.calculate_curvature(np.mean(x_coords, axis=1))
            y_curvature = self.calculate_curvature(np.mean(y_coords, axis=1))
            curvature_patterns.append([x_curvature, y_curvature])
            
            # 2. Dirección dominante del movimiento
            start_point = np.mean(hand_coords[:5], axis=0)
            end_point = np.mean(hand_coords[-5:], axis=0)
            direction_vector = end_point - start_point
            direction_patterns.append(direction_vector[:6])  # Primeros 6 componentes
            
            # 3. Patrón de velocidad (J tiene aceleración inicial y desaceleración)
            velocities = []
            for i in range(len(hand_coords)-1):
                vel = np.linalg.norm(hand_coords[i+1] - hand_coords[i])
                velocities.append(vel)
            
            # Análizar perfil de velocidad
            vel_profile = np.array(velocities)
            movement_patterns.append([
                np.max(vel_profile),           # Velocidad máxima
                np.mean(vel_profile),          # Velocidad promedio
                np.std(vel_profile),           # Variabilidad de velocidad
                np.argmax(vel_profile) / len(vel_profile),  # Posición del pico normalizada
                vel_profile[-1] / vel_profile[0] if vel_profile[0] > 0 else 0  # Ratio final/inicial
            ])
        
        # Estadísticas
        movement_patterns = np.array(movement_patterns)
        curvature_patterns = np.array(curvature_patterns)
        direction_patterns = np.array(direction_patterns)
        
        print(f"📈 Patrones de Movimiento (n={len(j_files)}):")
        print(f"   Vel. máxima: {np.mean(movement_patterns[:, 0]):.6f} ± {np.std(movement_patterns[:, 0]):.6f}")
        print(f"   Vel. promedio: {np.mean(movement_patterns[:, 1]):.6f} ± {np.std(movement_patterns[:, 1]):.6f}")
        print(f"   Variabilidad: {np.mean(movement_patterns[:, 2]):.6f} ± {np.std(movement_patterns[:, 2]):.6f}")
        print(f"   Pos. pico vel: {np.mean(movement_patterns[:, 3]):.3f} ± {np.std(movement_patterns[:, 3]):.3f}")
        
        print(f"\n🔄 Patrones de Curvatura:")
        print(f"   Curvatura X: {np.mean(curvature_patterns[:, 0]):.6f} ± {np.std(curvature_patterns[:, 0]):.6f}")
        print(f"   Curvatura Y: {np.mean(curvature_patterns[:, 1]):.6f} ± {np.std(curvature_patterns[:, 1]):.6f}")
        
        print(f"\n🧭 Patrones de Dirección:")
        for i in range(3):
            print(f"   Comp. {i+1}: {np.mean(direction_patterns[:, i]):.6f} ± {np.std(direction_patterns[:, i]):.6f}")
        
        return j_sequences, movement_patterns, curvature_patterns, direction_patterns

    def calculate_curvature(self, trajectory):
        """Calcula la curvatura de una trayectoria 1D"""
        if len(trajectory) < 3:
            return 0
        
        # Derivadas primera y segunda
        first_deriv = np.gradient(trajectory)
        second_deriv = np.gradient(first_deriv)
        
        # Curvatura = |f''| / (1 + f'^2)^(3/2)
        curvature = np.abs(second_deriv) / (1 + first_deriv**2)**(3/2)
        return np.mean(curvature[~np.isnan(curvature)])

    def create_j_specific_features(self, sequences):
        """Crear características específicas para reconocer J"""
        j_features = []
        
        for seq in sequences:
            hand_coords = seq[:, -63:]  # Solo mano
            
            # 1. Características de curvatura
            x_coords = hand_coords[:, 0::3]
            y_coords = hand_coords[:, 1::3]
            
            x_curve = self.calculate_curvature(np.mean(x_coords, axis=1))
            y_curve = self.calculate_curvature(np.mean(y_coords, axis=1))
            
            # 2. Características direccionales
            start_point = np.mean(hand_coords[:5], axis=0)
            mid_point = np.mean(hand_coords[len(hand_coords)//2-2:len(hand_coords)//2+3], axis=0)
            end_point = np.mean(hand_coords[-5:], axis=0)
            
            # Vectores de movimiento
            first_half_vector = mid_point - start_point
            second_half_vector = end_point - mid_point
            
            # Cambio de dirección (importante para J)
            direction_change = np.dot(first_half_vector, second_half_vector) / (
                np.linalg.norm(first_half_vector) * np.linalg.norm(second_half_vector) + 1e-8)
            
            # 3. Características de velocidad específicas para J
            velocities = []
            for i in range(len(hand_coords)-1):
                vel = np.linalg.norm(hand_coords[i+1] - hand_coords[i])
                velocities.append(vel)
            
            vel_profile = np.array(velocities)
            
            # Detectar patrón J: inicio rápido, medio lento, final curvo
            initial_speed = np.mean(vel_profile[:len(vel_profile)//3])
            middle_speed = np.mean(vel_profile[len(vel_profile)//3:2*len(vel_profile)//3])
            final_speed = np.mean(vel_profile[2*len(vel_profile)//3:])
            
            # 4. Simetría temporal (J no es simétrico)
            forward_half = hand_coords[:len(hand_coords)//2]
            backward_half = hand_coords[len(hand_coords)//2:]
            asymmetry = np.mean(np.abs(forward_half - np.flip(backward_half, axis=0)))
            
            # 5. Características específicas de forma J
            # Ratio de movimiento vertical vs horizontal
            vertical_movement = np.std(hand_coords[:, 1::3])  # Movimiento en Y
            horizontal_movement = np.std(hand_coords[:, 0::3])  # Movimiento en X
            movement_ratio = vertical_movement / (horizontal_movement + 1e-8)
            
            j_features.append([
                x_curve,              # Curvatura X
                y_curve,              # Curvatura Y  
                direction_change,     # Cambio de dirección
                initial_speed,        # Velocidad inicial
                middle_speed,         # Velocidad media
                final_speed,          # Velocidad final
                asymmetry,            # Asimetría temporal
                movement_ratio,       # Ratio vertical/horizontal
                np.max(vel_profile),  # Velocidad máxima
                np.std(vel_profile)   # Variabilidad de velocidad
            ])
        
        return np.array(j_features)

    def build_j_specialized_model(self, sequence_shape, feature_shape, num_classes):
        """Modelo especializado para reconocer J con mejor precisión"""
        
        # Branch 1: Análisis temporal bidireccional con atención
        sequence_input = Input(shape=sequence_shape, name='sequence_input')
        
        # Normalización de entrada
        seq_norm = LayerNormalization()(sequence_input)
        
        # Conv1D para patrones locales
        conv1 = Conv1D(64, 5, activation='relu', padding='same')(seq_norm)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(128, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # GRU bidireccional con más capacidad
        bigru1 = Bidirectional(
            GRU(256, return_sequences=True, activation='tanh', 
                dropout=0.3, recurrent_dropout=0.2)
        )(conv2)
        bigru1 = LayerNormalization()(bigru1)
        
        # Segunda capa GRU bidireccional
        bigru2 = Bidirectional(
            GRU(128, return_sequences=True, activation='tanh',
                dropout=0.3, recurrent_dropout=0.2)
        )(bigru1)
        bigru2 = LayerNormalization()(bigru2)
        
        # Capa de atención para enfocarse en partes importantes
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.2
        )(bigru2, bigru2)
        attention = LayerNormalization()(attention)
        
        # Pooling final
        gru_final = GRU(128, return_sequences=False, activation='tanh', dropout=0.3)(attention)
        
        # Branch 2: Características específicas de J
        feature_input = Input(shape=(feature_shape,), name='feature_input')
        feature_norm = LayerNormalization()(feature_input)
        
        feature_dense1 = Dense(128, activation='relu')(feature_norm)
        feature_dense1 = BatchNormalization()(feature_dense1)
        feature_dense1 = Dropout(0.4)(feature_dense1)
        
        feature_dense2 = Dense(64, activation='relu')(feature_dense1)
        feature_dense2 = BatchNormalization()(feature_dense2)
        feature_dense2 = Dropout(0.3)(feature_dense2)
        
        feature_dense3 = Dense(32, activation='relu')(feature_dense2)
        
        # Fusión con pesos adaptativos
        merged = Concatenate()([gru_final, feature_dense3])
        
        # Capas de clasificación con más capacidad
        final_dense1 = Dense(256, activation='relu')(merged)
        final_dense1 = BatchNormalization()(final_dense1)
        final_dense1 = Dropout(0.5)(final_dense1)
        
        final_dense2 = Dense(128, activation='relu')(final_dense1)
        final_dense2 = BatchNormalization()(final_dense2)
        final_dense2 = Dropout(0.4)(final_dense2)
        
        final_dense3 = Dense(64, activation='relu')(final_dense2)
        final_dense3 = Dropout(0.3)(final_dense3)
        
        # Salida con regularización adicional
        output = Dense(num_classes, activation='softmax', name='output')(final_dense3)
        
        # Crear modelo
        model = Model(inputs=[sequence_input, feature_input], outputs=output)
        
        # Compilar con parámetros optimizados para J
        optimizer = Adam(
            learning_rate=0.0005,  # Learning rate más bajo para mejor convergencia
            beta_1=0.9,
            beta_2=0.999,
            decay=1e-6
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy', 'precision', 'recall']
        )
        
        return model

    def prepare_data_for_j_recognition(self):
        """Preparar datos con énfasis en el reconocimiento de J"""
        print("\n🔄 PREPARANDO DATOS PARA RECONOCIMIENTO DE J")
        print("="*50)
        
        sequences = []
        labels = []
        
        # Cargar todas las secuencias
        for sign in self.signs:
            sign_path = f'{self.data_path}/{sign}'
            sign_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
            
            for file in sign_files:
                seq = np.load(f'{sign_path}/{file}')
                # Padear/truncar a longitud fija
                if len(seq) > self.sequence_length:
                    seq = seq[:self.sequence_length]
                else:
                    padding = np.tile(seq[-1], (self.sequence_length - len(seq), 1))
                    seq = np.vstack([seq, padding])
                
                sequences.append(seq)
                labels.append(sign)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        # Codificar labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"📊 Total de secuencias: {len(sequences)}")
        for sign in self.signs:
            count = np.sum(labels == sign)
            print(f"   {sign}: {count} muestras")
        
        # Crear características específicas para J
        j_features = self.create_j_specific_features(sequences)
        
        # Aplicar escalado robusto para manejar outliers
        scaler_seq = RobustScaler()
        scaler_feat = RobustScaler()
        
        # Reshape para scaler
        seq_reshaped = sequences.reshape(-1, sequences.shape[-1])
        seq_scaled = scaler_seq.fit_transform(seq_reshaped)
        sequences_scaled = seq_scaled.reshape(sequences.shape)
        
        features_scaled = scaler_feat.fit_transform(j_features)
        
        # Convertir labels a one-hot
        y = to_categorical(labels_encoded)
        
        # Calcular pesos de clase para balancear J
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels_encoded),
            y=labels_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Dar peso extra a J si es necesario
        j_index = self.label_encoder.transform(['J'])[0]
        class_weight_dict[j_index] = class_weight_dict[j_index] * 1.5  # Peso extra para J
        
        print(f"\n⚖️ Pesos de clase:")
        for i, sign in enumerate(self.label_encoder.classes_):
            print(f"   {sign}: {class_weight_dict[i]:.3f}")
        
        return sequences_scaled, features_scaled, y, class_weight_dict

    def train_j_model(self):
        """Entrenar modelo especializado en reconocimiento de J"""
        print("\n🚀 ENTRENANDO MODELO ESPECIALIZADO EN J")
        print("="*50)
        
        # Preparar datos
        sequences, features, y, class_weights = self.prepare_data_for_j_recognition()
        
        # Split estratificado
        X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            sequences, features, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
        )
        
        # Construir modelo
        model = self.build_j_specialized_model(
            sequence_shape=(self.sequence_length, self.num_features),
            feature_shape=features.shape[1],
            num_classes=len(self.signs)
        )
        
        print(f"\n🏗️ Arquitectura del modelo:")
        model.summary()
        
        # Callbacks mejorados
        callbacks = [
            EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar modelo
        print(f"\n🎯 Iniciando entrenamiento...")
        history = model.fit(
            [X_seq_train, X_feat_train], y_train,
            validation_data=([X_seq_test, X_feat_test], y_test),
            epochs=100,
            batch_size=16,  # Batch size más pequeño para mejor convergencia
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        print(f"\n📊 EVALUACIÓN FINAL:")
        print("="*30)
        
        # Predicciones
        y_pred = model.predict([X_seq_test, X_feat_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Accuracy por clase
        for i, sign in enumerate(self.label_encoder.classes_):
            mask = y_true_classes == i
            if np.sum(mask) > 0:
                accuracy = np.mean(y_pred_classes[mask] == y_true_classes[mask])
                print(f"   {sign}: {accuracy:.3f} ({np.sum(mask)} muestras)")
        
        # Accuracy específica para J
        j_index = self.label_encoder.transform(['J'])[0]
        j_mask = y_true_classes == j_index
        if np.sum(j_mask) > 0:
            j_accuracy = np.mean(y_pred_classes[j_mask] == y_true_classes[j_mask])
            j_precision = np.sum((y_pred_classes == j_index) & (y_true_classes == j_index)) / np.sum(y_pred_classes == j_index)
            j_recall = np.sum((y_pred_classes == j_index) & (y_true_classes == j_index)) / np.sum(j_mask)
            
            print(f"\n🎯 MÉTRICAS ESPECÍFICAS PARA J:")
            print(f"   Accuracy: {j_accuracy:.3f}")
            print(f"   Precision: {j_precision:.3f}")
            print(f"   Recall: {j_recall:.3f}")
            print(f"   F1-Score: {2 * (j_precision * j_recall) / (j_precision + j_recall):.3f}")
        
        # Guardar modelo mejorado
        os.makedirs('models', exist_ok=True)
        model.save('models/sign_model_j_specialized.keras')
        np.save('models/label_encoder_j_specialized.npy', self.label_encoder.classes_)
        
        print(f"\n✅ Modelo guardado en: models/sign_model_j_specialized.keras")
        return model, history

if __name__ == "__main__":
    # Ejecutar análisis y entrenamiento
    fixer = JRecognitionFixer()
    
    # Análizar características de J
    j_seqs, movement_patterns, curvature_patterns, direction_patterns = fixer.analyze_j_characteristics()
    
    # Entrenar modelo especializado
    model, history = fixer.train_j_model()
    
    print(f"\n🎉 ¡Proceso completado! El modelo especializado debería reconocer mejor la letra J.")
