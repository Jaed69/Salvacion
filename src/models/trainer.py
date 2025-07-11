# enhanced_model_trainer.py
# Algoritmo mejorado para distinguir señas estáticas vs dinámicas

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class EnhancedSequenceModelTrainer:
    def __init__(self, data_path='data/sequences', model_type='bidirectional_dynamic'):
        self.data_path = data_path
        self.model_type = model_type
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"El directorio de datos no existe: {self.data_path}")
            
        self.signs = np.array([name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))])
        
        if len(self.signs) == 0:
            raise ValueError(f"No se encontraron señas en {data_path}")
        
        self.label_encoder = LabelEncoder()
        self.sequence_length = 50
        self.num_features = 21 * 3 * 2  # 126 features
        
        print(f"🔍 Encontradas {len(self.signs)} señas: {', '.join(self.signs)}")

    def extract_motion_features(self, sequences):
        """Extrae características específicas de movimiento para distinguir señas estáticas vs dinámicas"""
        motion_features = []
        
        for seq in sequences:
            # 1. Varianza temporal (cuánto se mueve)
            temporal_variance = np.var(seq, axis=0).mean()
            
            # 2. Diferencias entre frames consecutivos
            frame_diffs = np.mean([np.mean(np.abs(seq[i+1] - seq[i])) for i in range(len(seq)-1)])
            
            # 3. Velocidad promedio de los landmarks de la mano (últimos 63 features)
            hand_landmarks = seq[:, -63:]  # Últimos 21 puntos * 3 coordenadas
            hand_velocity = np.mean([np.mean(np.abs(hand_landmarks[i+1] - hand_landmarks[i])) 
                                   for i in range(len(hand_landmarks)-1)])
            
            # 4. Aceleración (cambio en velocidad)
            velocities = [np.mean(np.abs(seq[i+1] - seq[i])) for i in range(len(seq)-1)]
            acceleration = np.mean([abs(velocities[i+1] - velocities[i]) for i in range(len(velocities)-1)])
            
            # 5. Frecuencia dominante del movimiento (FFT)
            fft_magnitude = np.abs(np.fft.fft(seq.flatten()))
            dominant_freq = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
            
            # 6. Entropía del movimiento
            movement_entropy = -np.sum(temporal_variance * np.log(temporal_variance + 1e-8))
            
            motion_features.append([
                temporal_variance,
                frame_diffs, 
                hand_velocity,
                acceleration,
                dominant_freq,
                movement_entropy
            ])
        
        return np.array(motion_features)

    def extract_dynamic_motion_features(self, sequences):
        """Extrae características específicas para señas dinámicas con análisis bidireccional"""
        dynamic_features = []
        
        for seq in sequences:
            # 1. Análisis de trayectoria completa (inicio → fin)
            start_position = np.mean(seq[:5], axis=0)  # Primeros 5 frames
            end_position = np.mean(seq[-5:], axis=0)   # Últimos 5 frames
            trajectory_vector = end_position - start_position
            trajectory_magnitude = np.linalg.norm(trajectory_vector)
            
            # 2. Curvatura de la trayectoria (útil para J, Z)
            # Calcular derivadas primera y segunda
            first_derivative = np.diff(seq, axis=0)
            second_derivative = np.diff(first_derivative, axis=0)
            
            # Calcular curvatura usando el cambio de dirección en lugar de producto cruz
            # Para vectores multidimensionales, usamos el ángulo entre vectores consecutivos
            curvatures = []
            for i in range(len(second_derivative)):
                v1 = first_derivative[i]
                v2 = second_derivative[i]
                # Calcular curvatura como la magnitud del cambio de dirección
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 1e-8 and v2_norm > 1e-8:
                    # Usar el producto punto para calcular el coseno del ángulo
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Evitar errores numéricos
                    curvature_val = 1 - abs(cos_angle)  # Curvatura basada en cambio de dirección
                    curvatures.append(curvature_val)
                else:
                    curvatures.append(0)
            curvature = np.mean(curvatures) if curvatures else 0
            
            # 3. Análisis de velocidad variable (aceleración/desaceleración)
            velocities = [np.linalg.norm(seq[i+1] - seq[i]) for i in range(len(seq)-1)]
            velocity_profile = np.array(velocities)
            
            # Detectar fases: aceleración inicial, velocidad constante, desaceleración final
            velocity_std = np.std(velocity_profile)
            velocity_trend = np.polyfit(range(len(velocity_profile)), velocity_profile, 1)[0]
            
            # 4. Análisis de simetría temporal (importante para detectar movimientos direccionales)
            forward_half = seq[:len(seq)//2]
            backward_half = seq[len(seq)//2:]
            temporal_symmetry = np.mean(np.abs(forward_half - np.flip(backward_half, axis=0)))
            
            # 5. Frecuencia dominante en diferentes ejes (X, Y, Z)
            fft_x = np.abs(np.fft.fft(seq[:, 0]))  # Análisis frecuencial eje X
            fft_y = np.abs(np.fft.fft(seq[:, 1]))  # Análisis frecuencial eje Y
            
            dominant_freq_x = np.argmax(fft_x[1:len(fft_x)//2]) + 1
            dominant_freq_y = np.argmax(fft_y[1:len(fft_y)//2]) + 1
            
            # 6. Análisis de patrón repetitivo (para detectar bucles como en J)
            autocorr = np.correlate(velocity_profile, velocity_profile, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            repetition_score = np.max(autocorr[1:]) / autocorr[0]  # Normalizado
            
            dynamic_features.append([
                trajectory_magnitude,
                curvature,
                velocity_std,
                velocity_trend,
                temporal_symmetry,
                dominant_freq_x,
                dominant_freq_y,
                repetition_score
            ])
        
        return np.array(dynamic_features)

    def augment_static_sequences(self, sequences, labels, target_movement_ratio=2.0):
        """Data augmentation específico para señas estáticas"""
        augmented_sequences = []
        augmented_labels = []
        
        for seq, label in zip(sequences, labels):
            augmented_sequences.append(seq)
            augmented_labels.append(label)
            
            # Calcular si es una seña estática (baja varianza)
            temporal_variance = np.var(seq, axis=0).mean()
            
            if temporal_variance < 0.0001:  # Umbral para señas estáticas
                # Agregar ruido sutil para crear variaciones
                for _ in range(3):  # 3 versiones aumentadas por seña estática
                    noise_scale = 0.001
                    augmented_seq = seq + np.random.normal(0, noise_scale, seq.shape)
                    augmented_sequences.append(augmented_seq)
                    augmented_labels.append(label)
                
                # Pequeñas traslaciones temporales
                for shift in [-2, -1, 1, 2]:
                    shifted_seq = np.roll(seq, shift, axis=0)
                    augmented_sequences.append(shifted_seq)
                    augmented_labels.append(label)
        
        return np.array(augmented_sequences), np.array(augmented_labels)

    def load_data_enhanced(self):
        """Carga datos con características mejoradas para análisis dinámico"""
        sequences, labels = [], []
        
        self.label_encoder.fit(self.signs)

        for sign in self.signs:
            sign_path = os.path.join(self.data_path, sign)
            for seq_file in os.listdir(sign_path):
                res = np.load(os.path.join(sign_path, seq_file))
                sequences.append(res)
                labels.append(self.label_encoder.transform([sign])[0])
        
        # Padded sequences
        X_raw = pad_sequences(sequences, maxlen=self.sequence_length, padding='post', truncating='post', dtype='float32')
        
        # Data augmentation para señas estáticas
        print("🔄 Aplicando data augmentation para señas estáticas...")
        X_augmented, labels_augmented = self.augment_static_sequences(X_raw, labels)
        
        # Extraer características de movimiento estándar
        print("📊 Extrayendo características de movimiento estándar...")
        motion_features = self.extract_motion_features(X_augmented)
        
        # Extraer características dinámicas específicas para modelo bidireccional
        print("🔄 Extrayendo características dinámicas avanzadas...")
        dynamic_features = self.extract_dynamic_motion_features(X_augmented)
        
        # Combinar características
        combined_features = np.concatenate([motion_features, dynamic_features], axis=1)
        
        # Convertir etiquetas a one-hot
        y = to_categorical(labels_augmented).astype(int)
        
        print(f"📈 Dataset aumentado: {X_raw.shape[0]} → {X_augmented.shape[0]} secuencias")
        print(f"📊 Características combinadas: {combined_features.shape}")
        
        # Guardar codificador
        np.save('data/label_encoder.npy', self.label_encoder.classes_)
        
        return X_augmented, combined_features, y

    def build_hybrid_model(self, sequence_shape, motion_shape, num_classes):
        """Modelo híbrido que combina análisis de secuencia y características de movimiento"""
        
        # Branch para secuencias temporales
        sequence_input = Input(shape=sequence_shape, name='sequence_input')
        
        # Conv1D para capturar patrones locales
        conv1 = Conv1D(32, 3, activation='relu')(sequence_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(64, 3, activation='relu')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # GRU para análisis temporal
        gru1 = GRU(128, return_sequences=True, activation='relu')(conv2)
        gru1 = Dropout(0.3)(gru1)
        gru2 = GRU(64, return_sequences=False, activation='relu')(gru1)
        gru2 = Dropout(0.3)(gru2)
        
        # Branch para características de movimiento
        motion_input = Input(shape=(motion_shape,), name='motion_input')
        motion_dense = Dense(32, activation='relu')(motion_input)
        motion_dense = BatchNormalization()(motion_dense)
        motion_dense = Dropout(0.3)(motion_dense)
        motion_dense = Dense(16, activation='relu')(motion_dense)
        
        # Concatenar ambos branches
        merged = Concatenate()([gru2, motion_dense])
        
        # Capas finales
        dense1 = Dense(128, activation='relu')(merged)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        output = Dense(num_classes, activation='softmax', name='classification_output')(dense2)
        
        model = Model(inputs=[sequence_input, motion_input], outputs=output)
        
        # Optimizador con learning rate adaptativo
        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        return model

    def build_standard_model(self, input_shape, num_classes):
        """Modelo estándar mejorado"""
        model = Sequential()
        
        model.add(Input(shape=input_shape))
        
        # Capas convolucionales para capturar patrones
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))
        
        # Capas recurrentes
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(Dropout(0.3))
        model.add(GRU(64, return_sequences=False, activation='relu'))
        model.add(Dropout(0.3))
        
        # Capas densas
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        optimizer = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
        return model

    def build_bidirectional_dynamic_model(self, sequence_shape, motion_shape, num_classes):
        """Modelo especializado en señas dinámicas con GRU bidireccional"""
        
        # Branch para secuencias temporales con análisis bidireccional
        sequence_input = Input(shape=sequence_shape, name='sequence_input')
        
        # Conv1D inicial para capturar patrones locales
        conv1 = Conv1D(32, 5, activation='relu', padding='same')(sequence_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(0.2)(conv1)
        
        conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        conv2 = Dropout(0.2)(conv2)
        
        # GRU bidireccional para capturar contexto temporal completo
        # Primera capa bidireccional
        bigru1 = Bidirectional(
            GRU(128, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
        )(conv2)
        bigru1 = BatchNormalization()(bigru1)
        
        # Segunda capa bidireccional con menor número de unidades
        bigru2 = Bidirectional(
            GRU(64, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
        )(bigru1)
        bigru2 = BatchNormalization()(bigru2)
        
        # Capa final unidireccional para consolidar información
        gru_final = GRU(64, return_sequences=False, activation='tanh', dropout=0.3)(bigru2)
        
        # Branch para características dinámicas específicas
        motion_input = Input(shape=(motion_shape,), name='motion_input')
        motion_dense = Dense(64, activation='relu')(motion_input)
        motion_dense = BatchNormalization()(motion_dense)
        motion_dense = Dropout(0.3)(motion_dense)
        motion_dense = Dense(32, activation='relu')(motion_dense)
        motion_dense = Dropout(0.2)(motion_dense)
        
        # Fusión de características temporales y dinámicas
        merged = Concatenate()([gru_final, motion_dense])
        
        # Capas densas finales con regularización
        dense1 = Dense(128, activation='relu')(merged)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.3)(dense2)
        
        # Capa de salida
        output = Dense(num_classes, activation='softmax', name='classification_output')(dense2)
        
        model = Model(inputs=[sequence_input, motion_input], outputs=output)
        
        # Optimizador especializado para señas dinámicas
        optimizer = Adam(learning_rate=0.0005, decay=1e-6)  # Learning rate más bajo para estabilidad
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )
        
        return model

    def train_enhanced(self, epochs=100):
        """Entrenamiento mejorado con callbacks y métricas detalladas"""
        if self.model_type == 'bidirectional_dynamic':
            X_sequences, motion_features, y = self.load_data_enhanced()
            
            # Split manteniendo estratificación
            indices = np.arange(len(X_sequences))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))
            
            X_seq_train, X_seq_test = X_sequences[train_idx], X_sequences[test_idx]
            X_motion_train, X_motion_test = motion_features[train_idx], motion_features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            num_classes = len(self.signs)
            model = self.build_bidirectional_dynamic_model(
                sequence_shape=(self.sequence_length, self.num_features),
                motion_shape=motion_features.shape[1],
                num_classes=num_classes
            )
            
            # Callbacks específicos para señas dinámicas
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-7)
            
            print("Resumen del modelo bidireccional dinámico:")
            print(model.summary())
            
            print(f"\n🚀 Iniciando entrenamiento bidireccional con {epochs} épocas...")
            print(f"📊 Datos de entrenamiento: {X_seq_train.shape[0]} muestras")
            print(f"📊 Datos de prueba: {X_seq_test.shape[0]} muestras")
            print(f"🔄 Características dinámicas: {motion_features.shape[1]} features")
            
            history = model.fit(
                [X_seq_train, X_motion_train], y_train,
                epochs=epochs,
                validation_data=([X_seq_test, X_motion_test], y_test),
                batch_size=8,  # Batch size más pequeño para estabilidad con bidireccional
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluación detallada
            test_loss, test_accuracy = model.evaluate([X_seq_test, X_motion_test], y_test, verbose=0)
            
            print(f"\n📈 Resultados finales del modelo bidireccional:")
            print(f"📈 Precisión en datos de prueba: {test_accuracy:.3f}")
            print(f"📈 Pérdida en datos de prueba: {test_loss:.3f}")
            
            # Análisis específico para señas dinámicas
            predictions = model.predict([X_seq_test, X_motion_test])
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # Análisis de señas dinámicas específicas
            dynamic_signs = ['J', 'Z', 'Ñ', 'RR', 'LL']
            for sign in dynamic_signs:
                if sign in self.signs:
                    sign_idx = self.label_encoder.transform([sign])[0]
                    sign_mask = true_classes == sign_idx
                    if np.any(sign_mask):
                        sign_accuracy = np.mean(predicted_classes[sign_mask] == sign_idx)
                        print(f"   Precisión para '{sign}' (dinámica): {sign_accuracy:.3f}")
            
            model.save('data/sign_model_bidirectional_dynamic.h5')
            print(f"\n✅ Modelo bidireccional guardado como 'data/sign_model_bidirectional_dynamic.h5'")
            
        elif self.model_type == 'hybrid':
            # Código del modelo híbrido original
            X_sequences, motion_features, y = self.load_data_enhanced()
            
            # Split manteniendo estratificación
            indices = np.arange(len(X_sequences))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))
            
            X_seq_train, X_seq_test = X_sequences[train_idx], X_sequences[test_idx]
            X_motion_train, X_motion_test = motion_features[train_idx], motion_features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            num_classes = len(self.signs)
            model = self.build_hybrid_model(
                sequence_shape=(self.sequence_length, self.num_features),
                motion_shape=motion_features.shape[1],
                num_classes=num_classes
            )
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7)
            
            print("Resumen del modelo híbrido:")
            print(model.summary())
            
            history = model.fit(
                [X_seq_train, X_motion_train], y_train,
                epochs=epochs,
                validation_data=([X_seq_test, X_motion_test], y_test),
                batch_size=16,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            test_loss, test_accuracy = model.evaluate([X_seq_test, X_motion_test], y_test, verbose=0)
            print(f"\n📈 Precisión en datos de prueba: {test_accuracy:.3f}")
            
            model.save('data/sign_model_hybrid.h5')
            print(f"\n✅ Modelo híbrido guardado como 'data/sign_model_hybrid.h5'")
            
        else:
            # Entrenamiento estándar mejorado
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            num_classes = len(self.signs)
            model = self.build_standard_model(input_shape=(self.sequence_length, self.num_features), num_classes=num_classes)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7)
            
            print("Resumen del modelo estándar mejorado:")
            print(model.summary())
            
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), 
                              batch_size=16, callbacks=[early_stopping, reduce_lr])
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"\n📈 Precisión en datos de prueba: {test_accuracy:.3f}")
            
            model.save(f'data/sign_model_{self.model_type}_enhanced.h5')
            print(f"\nModelo guardado como 'data/sign_model_{self.model_type}_enhanced.h5'")
        
        return model, history

    def load_data(self):
        """Método estándar de carga para compatibilidad"""
        sequences, labels = [], []
        
        self.label_encoder.fit(self.signs)

        for sign in self.signs:
            sign_path = os.path.join(self.data_path, sign)
            for seq_file in os.listdir(sign_path):
                res = np.load(os.path.join(sign_path, seq_file))
                sequences.append(res)
                labels.append(self.label_encoder.transform([sign])[0])
        
        X = pad_sequences(sequences, maxlen=self.sequence_length, padding='post', truncating='post', dtype='float32')
        y = to_categorical(labels).astype(int)
        
        np.save('data/label_encoder.npy', self.label_encoder.classes_)
        
        return X, y

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo mejorado de reconocimiento de señas')
    parser.add_argument('--data-path', default='data/sequences', help='Ruta al directorio de datos')
    parser.add_argument('--model-type', default='bidirectional_dynamic', 
                       choices=['bidirectional_dynamic', 'hybrid', 'gru', 'lstm'], 
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de entrenamiento')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Iniciando entrenamiento del modelo mejorado...")
        print(f"🧠 Tipo de modelo: {args.model_type}")
        
        trainer = EnhancedSequenceModelTrainer(
            data_path=args.data_path, 
            model_type=args.model_type
        )
        
        model, history = trainer.train_enhanced(epochs=args.epochs)
        
        print("\n✅ Entrenamiento completado exitosamente!")
        print("💡 Usa el modelo híbrido para mejor distinción entre señas estáticas y dinámicas")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
