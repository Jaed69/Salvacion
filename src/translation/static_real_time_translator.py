#!/usr/bin/env python3
"""
Traductor en tiempo real optimizado para señales estáticas
Enfocado en alta precisión y baja latencia para poses estáticas
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time
import json
from pathlib import Path
import sys
import os

def ensure_dir(path):
    """Asegura que el directorio existe"""
    os.makedirs(path, exist_ok=True)

class StaticRealTimeTranslator:
    """Traductor en tiempo real especializado para señales estáticas"""
    
    def __init__(self, model_path=None, 
                 labels_path=None,
                 confidence_threshold=0.85,
                 camera_index=0,
                 show_landmarks=True):
        
        # Auto-detectar el mejor modelo disponible
        if model_path is None:
            # Buscar modelo mejorado primero
            improved_model = Path('models/sign_model_static_improved.keras')
            static_model = Path('models/sign_model_static.keras')
            
            if improved_model.exists():
                self.model_path = improved_model
                self.labels_path = Path('models/label_encoder_static_improved.npy')
                print("🚀 Usando modelo mejorado detectado automáticamente")
            elif static_model.exists():
                self.model_path = static_model  
                self.labels_path = Path('models/label_encoder_static.npy')
                print("📦 Usando modelo estático básico")
            else:
                self.model_path = Path('models/sign_model_bidirectional_dynamic.keras')
                self.labels_path = Path('models/label_encoder.npy')
                print("⚡ Usando modelo dinámico como fallback")
        else:
            self.model_path = Path(model_path)
            self.labels_path = Path(labels_path) if labels_path else Path('models/label_encoder.npy')
        self.camera_index = camera_index
        self.show_landmarks = show_landmarks
        
        # Configuración optimizada para tiempo real (más flexible)
        self.config = {
            'confidence_threshold': max(0.7, confidence_threshold - 0.15),  # Reducir umbral base
            'stability_frames': 4,                          # Menos frames para confirmar (era 8)
            'prediction_cooldown': 8,                       # Cooldown más corto (era 15)
            'hand_confidence_min': 0.6,                     # Más tolerante con MediaPipe (era 0.8)
            'stabilization_window': 3,                      # Ventana más pequeña (era 5)
            'geometric_validation': False,                  # Desactivar validación estricta
            'show_confidence': True,                        # Mostrar confianza en pantalla
            'show_stability_indicator': True,               # Mostrar indicador de estabilidad
            'show_landmarks': show_landmarks,               # Mostrar landmarks
            'movement_tolerance': 0.15,                     # Tolerancia al movimiento (nuevo)
            'min_prediction_confidence': 0.4,               # Confianza mínima para mostrar (reducido de 0.6)
            'continuous_prediction': True                   # Predicción continua (nuevo)
        }
        
        # Estado del sistema
        self.is_stable = False
        self.stability_counter = 0
        self.prediction_cooldown_counter = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Contadores de debug
        self.debug_prediction_calls = 0
        self.debug_hands_detected = 0
        self.debug_predictions_made = 0
        
        # Buffers para estabilización
        self.landmarks_buffer = deque(maxlen=self.config['stabilization_window'])
        self.prediction_history = deque(maxlen=10)
        
        # Inicializar MediaPipe (ignorar errores de lint)
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(  # type: ignore
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore
        
        # Cargar modelo y metadatos
        self.load_model()
        
        print("🎯 Traductor Estático Inicializado")
        print(f"📊 Clases disponibles: {len(self.class_names)}")
        print(f"⚙️ Configuración: {self.config}")

    def load_model(self):
        """Carga modelo y todos los metadatos necesarios"""
        
        try:
            # Cargar modelo
            self.model = tf.keras.models.load_model(self.model_path)  # type: ignore
            print(f"✅ Modelo cargado: {self.model_path}")
            
            # Cargar nombres de clases
            self.class_names = np.load(self.labels_path, allow_pickle=True)
            print(f"✅ Clases cargadas: {list(self.class_names)}")
            
            # Cargar scalers (compatible con .pkl y .npy)
            # Buscar scalers mejorados primero
            scaler_landmarks_pkl = self.model_path.parent / 'scaler_landmarks_static_improved.pkl'
            scaler_geometric_pkl = self.model_path.parent / 'scaler_geometric_static_improved.pkl'
            
            # Fallback a scalers estáticos
            if not scaler_landmarks_pkl.exists():
                scaler_landmarks_pkl = self.model_path.parent / 'scaler_landmarks_static.pkl'
            if not scaler_geometric_pkl.exists():
                scaler_geometric_pkl = self.model_path.parent / 'scaler_geometric_static.pkl'
                
            # Fallback a formato .npy
            scaler_landmarks_npy = self.model_path.parent / 'scaler_landmarks_static.npy'
            scaler_geometric_npy = self.model_path.parent / 'scaler_geometric_static.npy'
            
            # Cargar scaler de landmarks
            if scaler_landmarks_pkl.exists():
                import pickle
                with open(scaler_landmarks_pkl, 'rb') as f:
                    scaler_landmarks = pickle.load(f)
                self.scaler_landmarks_mean = scaler_landmarks.center_
                self.scaler_landmarks_scale = scaler_landmarks.scale_
                print("✅ Scaler landmarks cargado (.pkl)")
            elif scaler_landmarks_npy.exists():
                scaler_data = np.load(scaler_landmarks_npy, allow_pickle=True).item()
                self.scaler_landmarks_mean = scaler_data['mean']
                self.scaler_landmarks_scale = scaler_data['scale']
                print("✅ Scaler landmarks cargado (.npy)")
            else:
                print("⚠️ Scaler landmarks no encontrado, usando normalización básica")
                self.scaler_landmarks_mean = None
                self.scaler_landmarks_scale = None
            
            # Cargar scaler geométrico
            if scaler_geometric_pkl.exists():
                import pickle
                with open(scaler_geometric_pkl, 'rb') as f:
                    scaler_geometric = pickle.load(f)
                self.scaler_geometric_mean = scaler_geometric.center_
                self.scaler_geometric_scale = scaler_geometric.scale_
                print("✅ Scaler geométrico cargado (.pkl)")
            elif scaler_geometric_npy.exists():
                scaler_data = np.load(scaler_geometric_npy, allow_pickle=True).item()
                self.scaler_geometric_mean = scaler_data['mean']
                self.scaler_geometric_scale = scaler_data['scale']
                print("✅ Scaler geométrico cargado (.npy)")
            else:
                print("⚠️ Scaler geométrico no encontrado, usando normalización básica")
                self.scaler_geometric_mean = None
                self.scaler_geometric_scale = None
                
        except Exception as e:
            raise FileNotFoundError(f"Error cargando modelo: {e}")

    def extract_geometric_features(self, landmarks):
        """Extrae características geométricas para ambas manos (idénticas al entrenamiento)"""
        
        # Dividir en mano izquierda (primeros 63) y mano derecha (siguientes 63)
        left_hand_landmarks = landmarks[:63].reshape(21, 3)
        right_hand_landmarks = landmarks[63:].reshape(21, 3)
        
        all_features = []
        
        # Extraer características para cada mano
        for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
            features = []
            
            # 1. Distancias desde muñeca
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            wrist = hand_landmarks[0]
            
            distances_from_wrist = [
                np.linalg.norm(thumb_tip - wrist),
                np.linalg.norm(index_tip - wrist),
                np.linalg.norm(middle_tip - wrist),
                np.linalg.norm(ring_tip - wrist),
                np.linalg.norm(pinky_tip - wrist)
            ]
            features.extend(distances_from_wrist)
            
            # 2. Ángulos entre dedos (simplificado para evitar errores)
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
            
            # 3. Una distancia inter-dedo (para completar 11 características por mano)
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            features.append(thumb_index_dist)
            
            all_features.extend(features)
        
        return np.array(all_features)  # Debe ser exactamente 22 características (11 por mano)

    def normalize_landmarks(self, landmarks):
        """Normaliza landmarks usando scaler del entrenamiento"""
        
        if self.scaler_landmarks_mean is not None:
            # Usar RobustScaler del entrenamiento
            normalized = (landmarks - self.scaler_landmarks_mean) / self.scaler_landmarks_scale
        else:
            # Normalización básica
            normalized = (landmarks - np.mean(landmarks)) / (np.std(landmarks) + 1e-8)
        
        return normalized

    def normalize_geometric_features(self, features):
        """Normaliza características geométricas"""
        
        if self.scaler_geometric_mean is not None:
            normalized = (features - self.scaler_geometric_mean) / self.scaler_geometric_scale
        else:
            normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return normalized

    def detect_stability(self, landmarks):
        """
        Detecta estabilidad relativa (más flexible para tiempo real)
        """
        
        self.landmarks_buffer.append(landmarks)
        
        if len(self.landmarks_buffer) < self.config['stabilization_window']:
            return True  # Asumir estable si no hay suficientes muestras
        
        # Calcular variabilidad en la ventana
        landmarks_array = np.array(self.landmarks_buffer)
        
        # Calcular movimiento entre frames consecutivos
        if len(landmarks_array) >= 2:
            frame_diffs = []
            for i in range(1, len(landmarks_array)):
                diff = np.linalg.norm(landmarks_array[i] - landmarks_array[i-1])
                frame_diffs.append(diff)
            
            avg_movement = np.mean(frame_diffs)
            
            # Umbral más tolerante para movimiento natural
            movement_threshold = self.config.get('movement_tolerance', 0.15)
            
            is_currently_stable = avg_movement < movement_threshold
        else:
            is_currently_stable = True
        
        # Sistema de conteo más flexible
        if is_currently_stable:
            self.stability_counter += 1
        else:
            # Decrementar gradualmente en lugar de resetear a 0
            self.stability_counter = max(0, self.stability_counter - 1)
        
        # Confirmar estabilidad con menos frames requeridos
        required_stable_frames = self.config['stability_frames']
        self.is_stable = self.stability_counter >= required_stable_frames
        
        return self.is_stable

    def predict_sign(self, landmarks):
        """Realiza predicción de seña estática con modo continuo"""
        
        # Sistema de cooldown más flexible
        if self.prediction_cooldown_counter > 0:
            self.prediction_cooldown_counter -= 1
            # En modo continuo, seguir prediciendo pero con cooldown reducido
            if not self.config.get('continuous_prediction', False):
                print(f"🔧 Cooldown activo: {self.prediction_cooldown_counter}")
                return self.current_prediction, self.current_confidence
        
        try:
            # Debug: verificar entrada (solo una vez cada 60 frames)
            if self.debug_prediction_calls % 60 == 1:
                print(f"🔧 Landmarks shape: {landmarks.shape}")
            
            # Normalizar landmarks
            landmarks_normalized = self.normalize_landmarks(landmarks)
            
            # Extraer características geométricas
            geometric_features = self.extract_geometric_features(landmarks)
            geometric_normalized = self.normalize_geometric_features(geometric_features)
            
            # Preparar inputs para el modelo
            landmarks_input = landmarks_normalized.reshape(1, -1)
            geometric_input = geometric_normalized.reshape(1, -1)
            
            # Predicción
            prediction = self.model.predict([landmarks_input, geometric_input], verbose=0)
            
            # Obtener clase y confianza
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Umbral más bajo para mostrar predicciones
            min_confidence = self.config.get('min_prediction_confidence', 0.4)
            
            # Validar confianza con umbral más flexible
            if confidence >= min_confidence:
                print(f"✅ {predicted_class} ({confidence:.3f})")
                # Si la confianza es alta, actualizar predicción principal
                if confidence >= self.config['confidence_threshold']:
                    self.current_prediction = predicted_class
                    self.current_confidence = confidence
                    # Reiniciar cooldown solo para predicciones de alta confianza
                    self.prediction_cooldown_counter = self.config['prediction_cooldown']
                
                # Contar predicción realizada
                self.debug_predictions_made += 1
                
                # Devolver la predicción actual (incluso si es de confianza media)
                return predicted_class, confidence
            else:
                # Debug ocasional para predicciones rechazadas (solo cada 120 frames)
                if self.debug_prediction_calls % 120 == 1:
                    print(f"❌ {predicted_class} ({confidence:.3f}) < {min_confidence}")
                # Si la confianza es muy baja, mantener predicción anterior
                return self.current_prediction, self.current_confidence
                
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    def draw_info(self, image, landmarks_3d=None):
        """Dibuja información en pantalla"""
        
        h, w = image.shape[:2]
        
        # Panel de información
        panel_height = 150
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel.fill(40)  # Fondo gris oscuro
        
        # Estado de estabilidad
        stability_color = (0, 255, 0) if self.is_stable else (0, 165, 255)
        stability_text = "ESTABLE" if self.is_stable else f"ESTABILIZANDO ({self.stability_counter}/{self.config['stability_frames']})"
        cv2.putText(panel, f"Estado: {stability_text}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        
        # Predicción actual
        if self.current_prediction:
            pred_color = (0, 255, 0) if self.current_confidence >= self.config['confidence_threshold'] else (0, 165, 255)
            cv2.putText(panel, f"Seña: {self.current_prediction}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
            cv2.putText(panel, f"Confianza: {self.current_confidence:.3f}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        else:
            cv2.putText(panel, "Seña: ---", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Cooldown
        if self.prediction_cooldown_counter > 0:
            cv2.putText(panel, f"Cooldown: {self.prediction_cooldown_counter}", (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Historial de predicciones
        if len(self.prediction_history) > 0:
            recent_predictions = list(self.prediction_history)[-3:]
            history_text = " -> ".join([p['class'] for p in recent_predictions])
            cv2.putText(panel, f"Historial: {history_text}", (w//2, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instrucciones
        cv2.putText(panel, "Mantén la pose estable para reconocimiento", (w//2, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(panel, "Presiona 'q' para salir, 'r' para reset", (w//2, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Combinar panel con imagen
        combined = np.vstack([panel, image])
        
        return combined

    def run(self):
        """Ejecuta el traductor en tiempo real"""
        
        print("\n🚀 INICIANDO TRADUCTOR ESTÁTICO")
        print("="*50)
        print("📹 Presiona 'q' para salir")
        print("📹 Presiona 'r' para resetear estado")
        print("📹 Mantén poses estables para mejor reconocimiento")
        print("="*50)
        
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Métricas de performance
        frame_count = 0
        start_time = time.time()
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error leyendo cámara")
                    break
                
                # Flip horizontal para efecto espejo
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Procesar con MediaPipe
                results = self.hands.process(frame_rgb)  # type: ignore
                
                # Variables para landmarks
                landmarks_3d = None
                hand_detected = False
                prediction = None
                confidence = 0.0
                
                if results.multi_hand_landmarks:  # type: ignore
                    self.debug_hands_detected += 1
                    hand_detected = True
                    
                    # Procesar ambas manos (como en el entrenamiento)
                    left_hand_landmarks = np.zeros(63)  # Inicializar con ceros
                    right_hand_landmarks = np.zeros(63)  # Inicializar con ceros
                    
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):  # type: ignore
                        # Dibujar landmarks
                        self.mp_draw.draw_landmarks(  # type: ignore
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,  # type: ignore
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # type: ignore
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)  # type: ignore
                        )
                        
                        # Extraer coordenadas 3D
                        hand_coords = []
                        for landmark in hand_landmarks.landmark:
                            hand_coords.extend([landmark.x, landmark.y, landmark.z])
                        
                        hand_coords = np.array(hand_coords)
                        
                        # Determinar qué mano es (simplificado: primera mano = izquierda, segunda = derecha)
                        # En un caso real podrías usar results.multi_handedness para determinar L/R
                        if idx == 0:
                            left_hand_landmarks = hand_coords
                        elif idx == 1:
                            right_hand_landmarks = hand_coords
                    
                    # Combinar ambas manos en el formato esperado (126 características)
                    landmarks_3d = np.concatenate([left_hand_landmarks, right_hand_landmarks])
                    
                    # Detectar estabilidad (ahora más flexible)
                    self.detect_stability(landmarks_3d)
                    
                    # Realizar predicción con ambas manos
                    self.debug_prediction_calls += 1
                    prediction, confidence = self.predict_sign(landmarks_3d)
                    
                    # Debug: mostrar predicciones en consola
                    if prediction and confidence > 0.5:
                        print(f"🔍 Predicción: {prediction} (confianza: {confidence:.3f})")
                    
                    # Ajustar confianza basado en estabilidad
                    if not self.is_stable and confidence:
                        # Reducir confianza si no está estable, pero no bloquear completamente
                        confidence = confidence * 0.8  # Penalizar ligeramente
                
                # Si no hay mano, resetear estado
                if not hand_detected:
                    self.is_stable = False
                    self.stability_counter = 0
                    self.landmarks_buffer.clear()
                
                # Dibujar información
                display_frame = self.draw_info(frame, landmarks_3d)
                
                # Calcular FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                
                # Mostrar FPS
                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (display_frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar frame
                cv2.imshow('LSP Esperanza - Traductor Estático', display_frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset estado
                    self.is_stable = False
                    self.stability_counter = 0
                    self.prediction_cooldown_counter = 0
                    self.current_prediction = None
                    self.current_confidence = 0.0
                    self.landmarks_buffer.clear()
                    self.prediction_history.clear()
                    print("🔄 Estado reseteado")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n👋 Traductor detenido por usuario")
        
        finally:
            # Limpiar recursos
            cap.release()
            cv2.destroyAllWindows()
            
            # Estadísticas finales
            total_time = time.time() - start_time
            print(f"\n📊 Estadísticas de sesión:")
            print(f"   Frames procesados: {frame_count}")
            print(f"   Tiempo total: {total_time:.2f}s")
            print(f"   FPS promedio: {frame_count/total_time:.2f}")
            print(f"   Manos detectadas: {self.debug_hands_detected}")
            print(f"   Llamadas a predicción: {self.debug_prediction_calls}")
            print(f"   Predicciones realizadas: {self.debug_predictions_made}")

if __name__ == "__main__":
    translator = StaticRealTimeTranslator()
    translator.run()
