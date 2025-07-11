# hybrid_real_time_translator.py
# Traductor en tiempo real mejorado con modelo híbrido para distinguir señas estáticas y dinámicas

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from scipy.spatial.distance import euclidean
import statistics

class BidirectionalRealTimeTranslator:
    def __init__(self, model_path='models/sign_model_bidirectional_dynamic.h5', signs_path='models/label_encoder.npy'):
        """
        Traductor con modelo bidireccional para señas estáticas y dinámicas
        """
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            print(f"❌ Modelo no encontrado en: {model_path}")
            print("💡 Entrena primero el modelo con:")
            print("   python scripts/train_model.py --model-type bidirectional_dynamic")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Verificar que el archivo de labels existe
        if not os.path.exists(signs_path):
            print(f"❌ Archivo de etiquetas no encontrado: {signs_path}")
            print("💡 Entrena primero el modelo para generar las etiquetas")
            raise FileNotFoundError(f"Archivo de etiquetas no encontrado: {signs_path}")
        
        self.model = load_model(model_path)
        self.signs = np.load(signs_path)
        self.sequence_length = 50
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.prediction_threshold = 0.8  # Umbral más alto para mayor precisión
        
        # Configuración específica para señas estáticas vs dinámicas
        self.static_signs = {'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'}
        self.dynamic_signs = {'J', 'Z', 'HOLA', 'GRACIAS', 'POR FAVOR'}  # J requiere movimiento
        
        # Buffer para análisis de movimiento
        self.movement_buffer = deque(maxlen=20)  # Últimos 20 frames para detectar movimiento
        self.stability_buffer = deque(maxlen=15)  # Buffer para detectar estabilidad
        self.movement_threshold = 0.02  # Umbral para detectar movimiento significativo
        
        # Configuración visual mejorada
        self.prediction_history = deque(maxlen=8)
        self.confidence_history = deque(maxlen=8)
        self.movement_history = deque(maxlen=8)
        self.last_prediction_time = time.time()
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.current_movement_level = 0.0
        self.prediction_flash_time = 0
        
        # Colores para UI mejorada
        self.ui_colors = {
            'primary': (64, 128, 255),      # Azul moderno
            'success': (46, 204, 113),      # Verde éxito
            'warning': (255, 193, 7),       # Amarillo advertencia
            'danger': (231, 76, 60),        # Rojo peligro
            'dark': (52, 73, 94),           # Gris oscuro
            'light': (236, 240, 241),       # Gris claro
            'background': (44, 62, 80),     # Fondo oscuro
            'text': (255, 255, 255),        # Texto blanco
            'accent': (155, 89, 182),       # Púrpura acento
            'static': (52, 152, 219),       # Azul para señas estáticas
            'dynamic': (230, 126, 34)       # Naranja para señas dinámicas
        }
        
        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,  # Mayor precisión
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _extract_landmarks(self, hand_landmarks):
        """Extrae landmarks normalizados de una mano"""
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks

    def _calculate_movement_level(self, current_landmarks):
        """
        Calcula el nivel de movimiento entre frames consecutivos
        """
        if len(self.movement_buffer) < 2:
            return 0.0
        
        # Comparar con el frame anterior
        previous_landmarks = self.movement_buffer[-1]
        
        # Calcular distancia euclidiana entre landmarks
        if len(current_landmarks) == len(previous_landmarks):
            distance = euclidean(current_landmarks, previous_landmarks)
            return min(distance, 1.0)  # Normalizar a [0, 1]
        
        return 0.0

    def _calculate_motion_features(self, sequence):
        """
        Calcula las 14 características de movimiento que requiere el modelo bidireccional dinámico
        """
        if len(sequence) < 2:
            return np.array([0.0] * 14)  # 6 básicas + 8 dinámicas avanzadas
        
        # Convertir a numpy array si no lo es
        sequence = np.array(sequence)
        
        # CARACTERÍSTICAS BÁSICAS (6 features)
        # 1. Varianza temporal
        temporal_variance = np.var(sequence, axis=0).mean()
        
        # 2. Movimiento entre frames consecutivos
        frame_diffs = np.mean([np.mean(np.abs(sequence[i+1] - sequence[i])) 
                              for i in range(len(sequence)-1)])
        
        # 3. Velocidad de manos (últimos 126 features)
        hand_landmarks = sequence[:, -126:]  # Últimos 21 puntos * 3 coordenadas * 2 manos
        hand_velocity = np.mean([np.mean(np.abs(hand_landmarks[i+1] - hand_landmarks[i])) 
                               for i in range(len(hand_landmarks)-1)])
        
        # 4. Aceleración (cambio en velocidad)
        velocities = [np.mean(np.abs(sequence[i+1] - sequence[i])) 
                     for i in range(len(sequence)-1)]
        acceleration = np.mean([abs(velocities[i+1] - velocities[i]) 
                              for i in range(len(velocities)-1)]) if len(velocities) > 1 else 0
        
        # 5. Frecuencia dominante del movimiento (FFT)
        fft_magnitude = np.abs(np.fft.fft(sequence.flatten()))
        dominant_freq = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1 if len(fft_magnitude) > 2 else 1
        
        # 6. Entropía del movimiento
        movement_entropy = -np.sum(temporal_variance * np.log(temporal_variance + 1e-8))
        
        # CARACTERÍSTICAS DINÁMICAS AVANZADAS (8 features)
        # 7. Magnitud de trayectoria (inicio → fin)
        start_position = np.mean(sequence[:5], axis=0) if len(sequence) >= 5 else sequence[0]
        end_position = np.mean(sequence[-5:], axis=0) if len(sequence) >= 5 else sequence[-1]
        trajectory_magnitude = np.linalg.norm(end_position - start_position)
        
        # 8. Curvatura de la trayectoria (útil para J, Z)
        if len(sequence) > 2:
            first_derivative = np.diff(sequence, axis=0)
            second_derivative = np.diff(first_derivative, axis=0)
            curvatures = []
            for i in range(len(second_derivative)):
                v1 = first_derivative[i]
                v2 = second_derivative[i]
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 1e-8 and v2_norm > 1e-8:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    curvature_val = 1 - abs(cos_angle)
                    curvatures.append(curvature_val)
                else:
                    curvatures.append(0)
            curvature = np.mean(curvatures) if curvatures else 0
        else:
            curvature = 0
        
        # 9. Desviación estándar de velocidad
        velocity_profile = np.array(velocities)
        velocity_std = np.std(velocity_profile)
        
        # 10. Tendencia de velocidad (aceleración/desaceleración)
        try:
            velocity_trend = np.polyfit(range(len(velocity_profile)), velocity_profile, 1)[0] if len(velocity_profile) > 1 else 0
        except:
            velocity_trend = 0
        
        # 11. Simetría temporal
        if len(sequence) > 4:
            forward_half = sequence[:len(sequence)//2]
            backward_half = sequence[len(sequence)//2:]
            min_len = min(len(forward_half), len(backward_half))
            temporal_symmetry = np.mean(np.abs(forward_half[:min_len] - np.flip(backward_half[:min_len], axis=0)))
        else:
            temporal_symmetry = 0
        
        # 12. Frecuencia dominante X
        if sequence.shape[1] > 0:
            fft_x = np.abs(np.fft.fft(sequence[:, 0]))
            dominant_freq_x = np.argmax(fft_x[1:len(fft_x)//2]) + 1 if len(fft_x) > 2 else 1
        else:
            dominant_freq_x = 1
        
        # 13. Frecuencia dominante Y
        if sequence.shape[1] > 1:
            fft_y = np.abs(np.fft.fft(sequence[:, 1]))
            dominant_freq_y = np.argmax(fft_y[1:len(fft_y)//2]) + 1 if len(fft_y) > 2 else 1
        else:
            dominant_freq_y = 1
        
        # 14. Puntuación de repetición (para detectar bucles como en J)
        if len(velocity_profile) > 1:
            autocorr = np.correlate(velocity_profile, velocity_profile, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            repetition_score = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] > 0 and len(autocorr) > 1 else 0
        else:
            repetition_score = 0
        
        return np.array([
            temporal_variance, frame_diffs, hand_velocity, acceleration, dominant_freq, movement_entropy,
            trajectory_magnitude, curvature, velocity_std, velocity_trend, temporal_symmetry,
            dominant_freq_x, dominant_freq_y, repetition_score
        ])

    def _analyze_movement_pattern(self):
        """
        Analiza el patrón de movimiento para determinar si es estático o dinámico
        """
        if len(self.movement_history) < 5:
            return "unknown", 0.0
        
        # Calcular estadísticas del movimiento
        movement_values = list(self.movement_history)
        avg_movement = statistics.mean(movement_values)
        max_movement = max(movement_values)
        movement_variance = statistics.variance(movement_values) if len(movement_values) > 1 else 0
        
        # Determinar tipo de seña basado en movimiento
        if avg_movement < self.movement_threshold and max_movement < self.movement_threshold * 2:
            return "static", avg_movement
        elif avg_movement > self.movement_threshold * 3 or movement_variance > 0.01:
            return "dynamic", avg_movement
        else:
            return "transitional", avg_movement

    def _should_predict_static_sign(self, predicted_sign, movement_type, confidence):
        """
        Determina si se debe predecir una seña estática basado en el análisis de movimiento
        """
        if predicted_sign in self.static_signs:
            # Para señas estáticas, requiere poca o nula movilidad
            if movement_type == "static" and confidence > self.prediction_threshold:
                return True
            elif movement_type == "transitional" and confidence > self.prediction_threshold + 0.1:
                return True
        
        return False

    def _should_predict_dynamic_sign(self, predicted_sign, movement_type, confidence):
        """
        Determina si se debe predecir una seña dinámica basado en el análisis de movimiento
        """
        if predicted_sign in self.dynamic_signs:
            # Para señas dinámicas, requiere movimiento detectado
            if movement_type == "dynamic" and confidence > self.prediction_threshold:
                return True
            elif movement_type == "transitional" and confidence > self.prediction_threshold + 0.15:
                return True
        
        return False

    def _draw_overlay_panel(self, frame, x, y, width, height, color, alpha=0.7):
        """Dibuja un panel translúcido para overlays"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def _draw_text_with_background(self, frame, text, position, font_scale=1.0, 
                                  color=(255, 255, 255), bg_color=(0, 0, 0), 
                                  thickness=2, padding=10):
        """Dibuja texto con fondo para mejor legibilidad"""
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x, text_y = position
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    def _draw_movement_analysis(self, frame, x, y):
        """Dibuja el análisis de movimiento con características híbridas"""
        movement_type, movement_level = self._analyze_movement_pattern()
        
        # Panel de fondo más grande para mostrar más información
        panel_height = 180
        self._draw_overlay_panel(frame, x, y, 350, panel_height, self.ui_colors['background'], 0.85)
        
        # Título
        cv2.putText(frame, "ANALISIS BIDIRECCIONAL DINAMICO", (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['accent'], 2)
        
        # Tipo de movimiento
        type_color = self.ui_colors['static'] if movement_type == "static" else \
                    self.ui_colors['dynamic'] if movement_type == "dynamic" else \
                    self.ui_colors['warning']
        
        cv2.putText(frame, f"Tipo: {movement_type.upper()}", (x + 10, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, type_color, 2)
        
        # Nivel de movimiento
        cv2.putText(frame, f"Nivel: {movement_level:.3f}", (x + 10, y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['light'], 1)
        
        # Mostrar características de movimiento si hay secuencia completa
        if len(self.sequence_buffer) == self.sequence_length:
            motion_features = self._calculate_motion_features(list(self.sequence_buffer))
            
            # Mostrar las 14 características (6 básicas + 8 dinámicas)
            feature_names = ["Var.Temp", "Mov.Frame", "Vel.Mano", "Acel", "Freq.Dom", "Entropía",
                           "Trayect", "Curvatura", "Vel.Std", "Vel.Trend", "Simetría", 
                           "Freq.X", "Freq.Y", "Repetición"]
            for i, (name, value) in enumerate(zip(feature_names, motion_features)):
                y_pos = y + 90 + (i * 12)  # Espaciado más pequeño para 14 características
                cv2.putText(frame, f"{name}: {value:.4f}", (x + 10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.ui_colors['light'], 1)
        
        # Barra de movimiento
        bar_x = x + 180
        bar_y = y + 50
        bar_width = 150
        bar_height = 15
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors['dark'], -1)
        
        # Barra de nivel actual
        fill_width = int(bar_width * min(movement_level / 0.1, 1.0))  # Normalizar a 0.1 como máximo
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     type_color, -1)
        
        # Línea de umbral
        threshold_x = bar_x + int(bar_width * (self.movement_threshold / 0.1))
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_height), 
                self.ui_colors['danger'], 2)
        
        # Indicadores de tipo de seña esperado
        if self.current_prediction:
            expected_type = "ESTATICA" if self.current_prediction in self.static_signs else \
                          "DINAMICA" if self.current_prediction in self.dynamic_signs else "MIXTA"
            expected_color = self.ui_colors['static'] if expected_type == "ESTATICA" else \
                           self.ui_colors['dynamic'] if expected_type == "DINAMICA" else \
                           self.ui_colors['warning']
            
            cv2.putText(frame, f"Esperado: {expected_type}", (x + 10, y + 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, expected_color, 1)

    def _draw_sign_classification(self, frame, x, y):
        """Dibuja información de clasificación de señas"""
        if not self.current_prediction:
            return
        
        # Panel de fondo
        panel_height = 100
        self._draw_overlay_panel(frame, x, y, 280, panel_height, self.ui_colors['background'], 0.85)
        
        # Título
        cv2.putText(frame, "CLASIFICACION", (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['primary'], 2)
        
        # Tipo de seña actual
        is_static = self.current_prediction in self.static_signs
        is_dynamic = self.current_prediction in self.dynamic_signs
        
        if is_static:
            sign_type = "ESTATICA"
            type_color = self.ui_colors['static']
            icon = "🤚"
        elif is_dynamic:
            sign_type = "DINAMICA"
            type_color = self.ui_colors['dynamic']
            icon = "👋"
        else:
            sign_type = "MIXTA"
            type_color = self.ui_colors['warning']
            icon = "✋"
        
        cv2.putText(frame, f"{icon} {self.current_prediction}: {sign_type}", (x + 10, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 2)
        
        # Compatibilidad con movimiento detectado
        movement_type, _ = self._analyze_movement_pattern()
        compatible = (is_static and movement_type == "static") or \
                    (is_dynamic and movement_type == "dynamic") or \
                    movement_type == "transitional"
        
        compatibility_text = "✓ COMPATIBLE" if compatible else "✗ NO COMPATIBLE"
        compatibility_color = self.ui_colors['success'] if compatible else self.ui_colors['danger']
        
        cv2.putText(frame, compatibility_text, (x + 10, y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, compatibility_color, 2)

    def _draw_confidence_bar(self, frame, confidence, x, y, width=200, height=20):
        """Dibuja una barra de confianza visual mejorada"""
        # Fondo de la barra
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.ui_colors['dark'], -1)
        
        # Barra de progreso con gradiente de color
        fill_width = int(width * confidence)
        if confidence > 0.9:
            color = self.ui_colors['success']
        elif confidence > 0.8:
            color = self.ui_colors['primary']
        elif confidence > 0.6:
            color = self.ui_colors['warning']
        else:
            color = self.ui_colors['danger']
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Borde
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.ui_colors['light'], 2)
        
        # Línea de umbral
        threshold_x = x + int(width * self.prediction_threshold)
        cv2.line(frame, (threshold_x, y), (threshold_x, y + height), 
                self.ui_colors['accent'], 2)
        
        # Texto de porcentaje
        percentage_text = f"{confidence*100:.1f}%"
        text_x = x + width + 10
        text_y = y + height - 5
        cv2.putText(frame, percentage_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors['text'], 2)

    def _draw_main_prediction(self, frame):
        """Dibuja la predicción principal con análisis híbrido"""
        frame_height, frame_width = frame.shape[:2]
        
        # Panel principal centrado
        panel_width = 700
        panel_height = 150
        panel_x = (frame_width - panel_width) // 2
        panel_y = frame_height - panel_height - 20
        
        # Efecto de flash para nuevas predicciones
        current_time = time.time()
        flash_alpha = 0.9
        if current_time - self.prediction_flash_time < 0.5:
            flash_alpha = 0.95 + 0.05 * np.sin((current_time - self.prediction_flash_time) * 20)
        
        self._draw_overlay_panel(frame, panel_x, panel_y, panel_width, panel_height, 
                               self.ui_colors['background'], flash_alpha)
        
        if self.current_prediction:
            # Determinar color basado en tipo de seña
            is_static = self.current_prediction in self.static_signs
            is_dynamic = self.current_prediction in self.dynamic_signs
            prediction_color = self.ui_colors['static'] if is_static else \
                             self.ui_colors['dynamic'] if is_dynamic else \
                             self.ui_colors['primary']
            
            # Texto principal de predicción - más grande
            self._draw_text_with_background(
                frame, self.current_prediction, 
                (panel_x + 30, panel_y + 60), 
                font_scale=2.5, color=prediction_color, 
                bg_color=self.ui_colors['dark'], thickness=3, padding=15
            )
            
            # Barra de confianza mejorada
            conf_x = panel_x + 30
            conf_y = panel_y + 85
            self._draw_confidence_bar(frame, self.current_confidence, conf_x, conf_y, 400, 25)
            
            # Indicador de compatibilidad de movimiento
            movement_type, movement_level = self._analyze_movement_pattern()
            
            # Verificar compatibilidad
            if is_static and movement_type == "static":
                status_text = "✓ SEÑA ESTATICA DETECTADA"
                status_color = self.ui_colors['success']
            elif is_dynamic and movement_type == "dynamic":
                status_text = "✓ SEÑA DINAMICA DETECTADA"
                status_color = self.ui_colors['success']
            elif movement_type == "transitional":
                status_text = "⚡ ANALIZANDO MOVIMIENTO..."
                status_color = self.ui_colors['warning']
            else:
                status_text = "⚠ VERIFICAR MOVIMIENTO"
                status_color = self.ui_colors['danger']
            
            cv2.putText(frame, status_text, (panel_x + 450, panel_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Tiempo de la predicción
            time_text = f"Ultima actualizacion: {datetime.now().strftime('%H:%M:%S')}"
            cv2.putText(frame, time_text, (panel_x + 450, panel_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ui_colors['light'], 1)
        else:
            # Mensaje de espera
            wait_text = "Esperando deteccion de manos..."
            cv2.putText(frame, wait_text, (panel_x + 30, panel_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.ui_colors['light'], 2)

    def process_frame(self, frame):
        """Procesa un frame y realiza la predicción híbrida"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Extraer landmarks de todas las manos detectadas
            all_landmarks = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extraer características
                landmarks = self._extract_landmarks(hand_landmarks)
                all_landmarks.extend(landmarks)
            
            # Asegurar que tenemos el número correcto de características
            if len(all_landmarks) < 126:  # 21 * 3 * 2 = 126 características para 2 manos
                # Rellenar con ceros si faltan manos
                all_landmarks.extend([0.0] * (126 - len(all_landmarks)))
            elif len(all_landmarks) > 126:
                # Truncar si hay demasiadas características
                all_landmarks = all_landmarks[:126]
            
            # Calcular nivel de movimiento
            current_movement = self._calculate_movement_level(all_landmarks)
            self.movement_history.append(current_movement)
            self.current_movement_level = current_movement
            
            # Agregar landmarks al buffer de movimiento
            self.movement_buffer.append(all_landmarks)
            
            # Agregar al buffer de secuencia
            self.sequence_buffer.append(all_landmarks)
            
            # Realizar predicción si el buffer está lleno
            if len(self.sequence_buffer) == self.sequence_length:
                sequence = np.array(list(self.sequence_buffer))
                sequence = np.expand_dims(sequence, axis=0)
                
                # Calcular características de movimiento para el modelo híbrido
                motion_features = self._calculate_motion_features(list(self.sequence_buffer))
                motion_features = np.expand_dims(motion_features, axis=0)
                
                # Realizar predicción con ambas entradas
                predictions = self.model.predict([sequence, motion_features], verbose=0)
                predicted_index = np.argmax(predictions)
                confidence = predictions[0][predicted_index]
                predicted_sign = self.signs[predicted_index]
                
                # Análisis de movimiento
                movement_type, movement_level = self._analyze_movement_pattern()
                
                # Aplicar lógica híbrida para validar predicción
                should_predict = False
                
                if predicted_sign in self.static_signs:
                    should_predict = self._should_predict_static_sign(predicted_sign, movement_type, confidence)
                elif predicted_sign in self.dynamic_signs:
                    should_predict = self._should_predict_dynamic_sign(predicted_sign, movement_type, confidence)
                else:
                    # Para señas que no están clasificadas, usar umbral estándar
                    should_predict = confidence > self.prediction_threshold
                
                # Actualizar predicción si es válida
                if should_predict:
                    if predicted_sign != self.current_prediction:
                        self.prediction_flash_time = time.time()
                    
                    self.current_prediction = predicted_sign
                    self.current_confidence = confidence
                    self.last_prediction_time = time.time()
                    
                    # Agregar al historial
                    self.prediction_history.append(predicted_sign)
                    self.confidence_history.append(confidence)
                
                # Limpiar predicción antigua si ha pasado mucho tiempo sin detección válida
                elif time.time() - self.last_prediction_time > 2.0:
                    self.current_prediction = ""
                    self.current_confidence = 0.0
        
        else:
            # No se detectaron manos, limpiar buffers gradualmente
            if time.time() - self.last_prediction_time > 3.0:
                self.current_prediction = ""
                self.current_confidence = 0.0
                # Limpiar buffer de movimiento cuando no hay manos
                self.movement_history.clear()
        
        return frame

    def run(self):
        """Ejecuta el traductor en tiempo real"""
        print("🚀 Iniciando Traductor Bidireccional de Lenguaje de Señas")
        print("📋 Señas estáticas:", ", ".join(sorted(self.static_signs)))
        print("📋 Señas dinámicas:", ", ".join(sorted(self.dynamic_signs)))
        print("⚡ Presiona 'q' para salir, 'r' para resetear, 't' para ajustar umbral")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Voltear frame horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar frame
            frame = self.process_frame(frame)
            
            # Dibujar UI mejorada
            self._draw_main_prediction(frame)
            self._draw_movement_analysis(frame, 20, 20)
            self._draw_sign_classification(frame, 20, 180)
            
            # Información del sistema en la esquina superior derecha
            frame_height, frame_width = frame.shape[:2]
            info_text = [
                f"Modelo: Bidireccional GRU",
                f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}",
                f"Umbral: {self.prediction_threshold:.1f}",
                f"Movimiento: {self.current_movement_level:.3f}",
                f"Features: 14 (6+8 dinamicas)"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (frame_width - 300, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['light'], 1)
            
            # Mostrar frame
            cv2.imshow('Traductor Bidireccional LSP - Dinamicas Mejoradas', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Resetear buffers
                self.sequence_buffer.clear()
                self.movement_buffer.clear()
                self.movement_history.clear()
                self.prediction_history.clear()
                self.confidence_history.clear()
                self.current_prediction = ""
                self.current_confidence = 0.0
                print("🔄 Buffers reseteados")
            elif key == ord('t'):
                # Ajustar umbral
                self.prediction_threshold = 0.9 if self.prediction_threshold < 0.9 else 0.6
                print(f"🎯 Umbral ajustado a: {self.prediction_threshold}")
            elif key == ord('d'):
                # Modo debug - mostrar información detallada
                print(f"Debug - Predicción: {self.current_prediction}, Confianza: {self.current_confidence:.3f}")
                print(f"Debug - Movimiento: {self._analyze_movement_pattern()}")
                print(f"Debug - Buffer size: {len(self.sequence_buffer)}")
        
        # Limpiar recursos
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Traductor bidireccional cerrado")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traductor bidireccional de lenguaje de señas con GRU dinámico')
    parser.add_argument('--model', default='models/sign_model_bidirectional_dynamic.h5', help='Ruta al modelo bidireccional')
    parser.add_argument('--signs', default='models/label_encoder.npy', help='Ruta al archivo de señas')
    parser.add_argument('--threshold', type=float, default=0.8, help='Umbral de confianza')
    
    args = parser.parse_args()
    
    try:
        translator = BidirectionalRealTimeTranslator(
            model_path=args.model,
            signs_path=args.signs
        )
        translator.prediction_threshold = args.threshold
        translator.run()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que el modelo bidireccional existe en models/sign_model_bidirectional_dynamic.h5")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
