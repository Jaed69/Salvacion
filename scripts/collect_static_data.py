#!/usr/bin/env python3
"""
Recolector de datos optimizado para señales estáticas
Enfocado en capturar poses estables con alta calidad
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
from pathlib import Path
import json

class StaticDataCollector:
    """Recolector especializado para señales estáticas con validación de calidad"""
    
    def __init__(self, output_dir='data/sequences'):
        self.output_dir = Path(output_dir)
        
        # Configuración específica para señales estáticas
        self.config = {
            'stability_threshold': 0.001,        # Muy estricto para estáticas
            'stability_frames_required': 20,     # Frames consecutivos estables
            'quality_threshold': 0.85,           # Calidad mínima MediaPipe
            'samples_per_sign': 30,              # Muestras objetivo por seña
            'recording_duration': 45,            # Frames a capturar (1.5s a 30fps)
            'stabilization_window': 10,          # Ventana para detectar estabilidad
            'geometric_validation': True,        # Validar características geométricas
            'auto_capture': True,                # Captura automática cuando detecta estabilidad
            'show_quality_metrics': True         # Mostrar métricas en tiempo real
        }
        
        # Estado del sistema
        self.current_sign = None
        self.samples_collected = 0
        self.is_recording = False
        self.recording_buffer = []
        self.stability_counter = 0
        self.quality_scores = deque(maxlen=30)
        
        # Buffers para análisis
        self.landmarks_buffer = deque(maxlen=self.config['stabilization_window'])
        self.stability_history = deque(maxlen=100)
        
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(  # type: ignore
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Recolector de Datos Estáticos Inicializado")
        print(f"📁 Directorio de salida: {self.output_dir}")
        print(f"⚙️ Configuración: {self.config}")

    def detect_stability(self, landmarks):
        """Detecta si la pose actual es estable"""
        
        self.landmarks_buffer.append(landmarks.copy())
        
        if len(self.landmarks_buffer) < self.config['stabilization_window']:
            return False, 0.0
        
        # Calcular variabilidad en la ventana
        landmarks_array = np.array(self.landmarks_buffer)
        
        # Variabilidad promedio por landmark
        variances = np.var(landmarks_array, axis=0)
        mean_variance = np.mean(variances)
        
        # Calcular score de estabilidad (0-1, 1 = muy estable)
        stability_score = max(0, 1 - (mean_variance / self.config['stability_threshold']))
        
        # Es estable si la varianza está por debajo del umbral
        is_stable = mean_variance < self.config['stability_threshold']
        
        # Actualizar contador de estabilidad
        if is_stable:
            self.stability_counter += 1
        else:
            self.stability_counter = 0
        
        # Agregar a historial
        self.stability_history.append({
            'timestamp': time.time(),
            'variance': mean_variance,
            'is_stable': is_stable,
            'stability_score': stability_score
        })
        
        return is_stable, stability_score

    def validate_geometric_quality(self, landmarks):
        """Valida la calidad geométrica de la pose"""
        
        try:
            # Reshape para obtener landmarks 3D
            hand_landmarks = landmarks[-63:].reshape(21, 3)
            
            # 1. Verificar que la mano esté bien extendida/visible
            wrist = hand_landmarks[0]
            finger_tips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
            
            # Distancias desde muñeca a puntas de dedos
            distances = [np.linalg.norm(tip - wrist) for tip in finger_tips]
            
            # La mano debe tener una extensión mínima
            min_extension = 0.15  # Distancia mínima normalizada
            if max(distances) < min_extension:
                return False, "Mano muy cerrada o poco visible"
            
            # 2. Verificar que no hay landmarks anómalos
            # Todas las coordenadas deben estar en rango válido
            if np.any(hand_landmarks < -1) or np.any(hand_landmarks > 2):
                return False, "Landmarks fuera de rango"
            
            # 3. Verificar proporciones anatómicas básicas
            # El dedo medio debe ser más largo que los otros
            middle_length = np.linalg.norm(hand_landmarks[12] - hand_landmarks[9])
            thumb_length = np.linalg.norm(hand_landmarks[4] - hand_landmarks[2])
            
            if middle_length < thumb_length * 0.8:
                return False, "Proporciones anatómicas anómalas"
            
            return True, "Geometría válida"
            
        except Exception as e:
            return False, f"Error en validación: {e}"

    def calculate_quality_score(self, landmarks, hand_confidence):
        """Calcula score de calidad general"""
        
        # Componentes del score de calidad
        scores = []
        
        # 1. Confianza de MediaPipe (0-1)
        mp_confidence = min(hand_confidence, 1.0)
        scores.append(mp_confidence)
        
        # 2. Estabilidad actual (0-1)
        _, stability_score = self.detect_stability(landmarks)
        scores.append(stability_score)
        
        # 3. Validación geométrica (0 o 1)
        is_valid, _ = self.validate_geometric_quality(landmarks)
        geometric_score = 1.0 if is_valid else 0.0
        scores.append(geometric_score)
        
        # 4. Completitud de landmarks (verificar que no hay NaN)
        completeness_score = 1.0 if not np.any(np.isnan(landmarks)) else 0.0
        scores.append(completeness_score)
        
        # Score final ponderado
        weights = [0.3, 0.4, 0.2, 0.1]  # Estabilidad es más importante
        final_score = np.average(scores, weights=weights)
        
        return final_score, {
            'mediapipe_confidence': mp_confidence,
            'stability_score': stability_score,
            'geometric_score': geometric_score,
            'completeness_score': completeness_score
        }

    def start_recording_session(self, sign_name):
        """Inicia sesión de grabación para una seña específica"""
        
        self.current_sign = sign_name
        self.samples_collected = 0
        
        # Crear directorio para la seña
        sign_dir = self.output_dir / sign_name
        sign_dir.mkdir(exist_ok=True)
        
        # Contar muestras existentes
        existing_files = list(sign_dir.glob('*.npy'))
        start_index = len(existing_files)
        
        print(f"\n🎯 INICIANDO RECOLECCIÓN PARA: {sign_name}")
        print(f"📊 Muestras existentes: {len(existing_files)}")
        print(f"🎯 Objetivo: {self.config['samples_per_sign']} muestras")
        print(f"📋 Configuración:")
        print(f"   • Estabilidad requerida: {self.config['stability_frames_required']} frames")
        print(f"   • Umbral de calidad: {self.config['quality_threshold']}")
        print(f"   • Captura automática: {self.config['auto_capture']}")
        print("="*50)
        
        return start_index

    def save_sample(self, landmarks_sequence, quality_info):
        """Guarda una muestra con metadatos de calidad"""
        
        if not self.current_sign:
            return False
        
        # Crear nombre de archivo con timestamp
        timestamp = int(time.time() * 1000)  # Timestamp en milisegundos
        filename = f"{timestamp}_{self.current_sign}_static.npy"
        
        sign_dir = self.output_dir / self.current_sign
        file_path = sign_dir / filename
        
        # Guardar landmarks
        np.save(file_path, landmarks_sequence)
        
        # Guardar metadatos
        metadata = {
            'timestamp': timestamp,
            'sign': self.current_sign,
            'sample_type': 'static',
            'quality_score': quality_info['final_score'],
            'quality_details': quality_info['details'],
            'frames_count': len(landmarks_sequence),
            'config_used': self.config.copy()
        }
        
        metadata_path = sign_dir / f"{timestamp}_{self.current_sign}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.samples_collected += 1
        
        print(f"✅ Muestra guardada: {filename}")
        print(f"📊 Calidad: {quality_info['final_score']:.3f}")
        print(f"📈 Progreso: {self.samples_collected}/{self.config['samples_per_sign']}")
        
        return True

    def draw_interface(self, image, landmarks=None, quality_info=None):
        """Dibuja interfaz de usuario con métricas"""
        
        h, w = image.shape[:2]
        
        # Panel de información
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel.fill(30)
        
        # Información de sesión
        if self.current_sign:
            cv2.putText(panel, f"Recolectando: {self.current_sign}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(panel, f"Muestras: {self.samples_collected}/{self.config['samples_per_sign']}", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(panel, "Presiona una tecla para iniciar recolección", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Estado de estabilidad
        stability_frames_needed = max(0, self.config['stability_frames_required'] - self.stability_counter)
        
        if self.stability_counter > 0:
            stability_color = (0, 255, 0) if stability_frames_needed == 0 else (0, 165, 255)
            stability_text = "ESTABLE" if stability_frames_needed == 0 else f"Estabilizando... ({stability_frames_needed} frames)"
            cv2.putText(panel, f"Estado: {stability_text}", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        
        # Calidad actual
        if quality_info:
            quality_color = (0, 255, 0) if quality_info['final_score'] >= self.config['quality_threshold'] else (0, 100, 255)
            cv2.putText(panel, f"Calidad: {quality_info['final_score']:.3f}", (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
            
            # Detalles de calidad
            details = quality_info['details']
            y_offset = 145
            for key, value in details.items():
                cv2.putText(panel, f"{key}: {value:.2f}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 15
        
        # Estado de grabación
        if self.is_recording:
            cv2.putText(panel, "🔴 GRABANDO", (w - 150, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instrucciones
        instructions = [
            "Teclas: A, B, C... para iniciar recolección",
            "ESPACIO: Captura manual | ESC: Cancelar | Q: Salir"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(panel, instruction, (w//2, 115 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Combinar panel con imagen
        combined = np.vstack([panel, image])
        
        return combined

    def process_frame(self, frame):
        """Procesa un frame completo"""
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)  # type: ignore
        
        landmarks_3d = None
        quality_info = None
        hand_detected = False
        
        if results.multi_hand_landmarks:  # type: ignore
            for hand_landmarks in results.multi_hand_landmarks:  # type: ignore
                hand_detected = True
                
                # Dibujar landmarks
                self.mp_draw.draw_landmarks(  # type: ignore
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,  # type: ignore
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # type: ignore
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)  # type: ignore
                )
                
                # Extraer coordenadas 3D
                landmarks_3d = []
                for landmark in hand_landmarks.landmark:  # type: ignore
                    landmarks_3d.extend([landmark.x, landmark.y, landmark.z])  # type: ignore
                
                landmarks_3d = np.array(landmarks_3d)
                
                # Calcular calidad
                hand_confidence = 1.0  # MediaPipe no proporciona confidence en esta versión
                final_score, details = self.calculate_quality_score(landmarks_3d, hand_confidence)
                
                quality_info = {
                    'final_score': final_score,
                    'details': details
                }
                
                self.quality_scores.append(final_score)
                
                # Lógica de auto-captura
                if (self.current_sign and 
                    self.config['auto_capture'] and 
                    not self.is_recording and
                    self.stability_counter >= self.config['stability_frames_required'] and
                    final_score >= self.config['quality_threshold']):
                    
                    # Iniciar grabación automática
                    self.start_capture()
                
                break  # Solo procesar una mano
        
        # Si no hay mano, resetear contadores
        if not hand_detected:
            self.stability_counter = 0
            self.landmarks_buffer.clear()
        
        # Procesar grabación
        if self.is_recording and landmarks_3d is not None:
            self.recording_buffer.append(landmarks_3d.copy())
            
            # Verificar si completamos la grabación
            if len(self.recording_buffer) >= self.config['recording_duration']:
                self.finish_capture(quality_info)
        
        return frame, landmarks_3d, quality_info

    def start_capture(self):
        """Inicia captura de muestra"""
        
        if not self.current_sign:
            return
        
        print(f"🔴 Iniciando captura para {self.current_sign}...")
        self.is_recording = True
        self.recording_buffer = []

    def finish_capture(self, quality_info):
        """Finaliza captura y guarda muestra"""
        
        self.is_recording = False
        
        if len(self.recording_buffer) > 0 and quality_info:
            # Convertir buffer a array
            sequence = np.array(self.recording_buffer)
            
            # Guardar muestra
            success = self.save_sample(sequence, quality_info)
            
            if success:
                print(f"✅ Captura completada - Calidad: {quality_info['final_score']:.3f}")
            else:
                print("❌ Error guardando muestra")
        
        # Limpiar buffer
        self.recording_buffer = []

    def run(self):
        """Ejecuta el recolector de datos"""
        
        print("\n📊 RECOLECTOR DE DATOS ESTÁTICOS")
        print("="*50)
        print("📋 Instrucciones:")
        print("   • Presiona A, B, C, etc. para recolectar esa seña")
        print("   • Mantén poses estables para activar auto-captura")
        print("   • ESPACIO para captura manual")
        print("   • ESC para cancelar sesión actual")
        print("   • Q para salir")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip horizontal para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Procesar frame
                processed_frame, landmarks, quality_info = self.process_frame(frame)
                
                # Dibujar interfaz
                display_frame = self.draw_interface(processed_frame, landmarks, quality_info)
                
                # Mostrar
                cv2.imshow('LSP Esperanza - Recolector Estático', display_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == 27:  # ESC
                    self.current_sign = None
                    self.is_recording = False
                    self.recording_buffer = []
                    print("❌ Sesión cancelada")
                elif key == ord(' '):  # Espacio para captura manual
                    if self.current_sign and not self.is_recording:
                        self.start_capture()
                elif key >= ord('a') and key <= ord('z'):
                    # Iniciar recolección para letra
                    letter = chr(key).upper()
                    self.start_recording_session(letter)
                elif key >= ord('A') and key <= ord('Z'):
                    # Iniciar recolección para letra
                    letter = chr(key)
                    self.start_recording_session(letter)
        
        except KeyboardInterrupt:
            print("\n👋 Recolección detenida por usuario")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 Resumen de sesión:")
            print(f"   Muestras recolectadas: {self.samples_collected}")
            if len(self.quality_scores) > 0:
                print(f"   Calidad promedio: {np.mean(self.quality_scores):.3f}")

if __name__ == "__main__":
    collector = StaticDataCollector()
    collector.run()
