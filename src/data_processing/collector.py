# enhanced_data_collector.py
# Colector de datos mejorado para señas dinámicas vs estáticas

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
from datetime import datetime

class EnhancedDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configuración mejorada para cámara
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Configuración de recolección optimizada para GRU bidireccional
        self.sequence_length = 60  # Aumentado para mejor contexto temporal
        self.frame_buffer = []
        self.current_frame_handedness = []  # Tracking de lateralidad por frame
        self.recording = False
        self.current_sign = ""
        self.sign_type = "static"  # static, dynamic, phrase
        
        # Contador de secuencias
        self.sequence_count = 0
        self.session_data = {
            'start_time': datetime.now().isoformat(),
            'collected_signs': {},
            'quality_metrics': []
        }
        
        # Cargar plan de recolección
        self.load_collection_plan()
        
        # Señas prioritarias según el plan
        self.priority_signs = {
            'CRÍTICO': ['J', 'Z', 'Ñ', 'RR'],
            'ALTO': ['ADIÓS', 'SÍ', 'NO', 'CÓMO'],
            'MEDIO': ['QUÉ', 'DÓNDE', 'CUÁNDO', 'LL'],
            'BAJO': ['100', '1000']
        }
        
        # Configuración por tipo de seña optimizada para GRU bidireccional
        self.sign_config = {
            'static': {
                'duration': 4.0,  # Aumentado para mejor contexto bidireccional
                'stability_required': True,
                'movement_threshold': 0.02,
                'description': 'Mantener posición estable con contexto temporal'
            },
            'dynamic': {
                'duration': 6.0,  # Aumentado para capturar movimiento completo
                'stability_required': False,
                'movement_threshold': 0.05,
                'description': 'Movimiento completo con inicio y fin claros'
            },
            'phrase': {
                'duration': 8.0,  # Aumentado para expresiones complejas
                'stability_required': False,
                'movement_threshold': 0.03,
                'description': 'Expresión natural completa'
            }
        }

    def load_collection_plan(self):
        """Carga el plan de recolección si existe"""
        try:
            with open('plan_mejora_dataset.json', 'r', encoding='utf-8') as f:
                self.plan = json.load(f)
            print("📋 Plan de recolección cargado")
        except FileNotFoundError:
            print("⚠️  Plan de recolección no encontrado")
            self.plan = None

    def classify_sign_type(self, sign):
        """Clasifica el tipo de seña"""
        static_signs = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
            'V', 'W', 'X', 'Y'
        }
        
        dynamic_signs = {
            'J', 'Z', 'Ñ', 'RR', 'LL'
        }
        
        if sign in static_signs:
            return 'static'
        elif sign in dynamic_signs:
            return 'dynamic'
        else:
            return 'phrase'

    def extract_landmarks(self, hand_landmarks):
        """Extrae landmarks de una mano"""
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z
        
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])
        return landmarks

    def calculate_movement_quality(self, sequence):
        """Calcula métricas de calidad del movimiento"""
        if len(sequence) < 2:
            return {'movement_avg': 0, 'stability': 0, 'coverage': 0}
        
        # Calcular movimiento promedio
        movements = []
        for i in range(1, len(sequence)):
            movement = np.linalg.norm(np.array(sequence[i]) - np.array(sequence[i-1]))
            movements.append(movement)
        
        movement_avg = np.mean(movements)
        
        # Calcular estabilidad (menor varianza = más estable)
        stability = 1.0 / (1.0 + np.var(sequence, axis=0).mean())
        
        # Cobertura (qué tan completa es la secuencia)
        coverage = len(sequence) / self.sequence_length
        
        return {
            'movement_avg': movement_avg,
            'stability': stability,
            'coverage': coverage,
            'movements': movements
        }

    def evaluate_sequence_quality(self, sequence, sign_type):
        """Evalúa la calidad de una secuencia según el tipo de seña"""
        quality = self.calculate_movement_quality(sequence)
        
        # Criterios específicos por tipo
        if sign_type == 'static':
            # Para señas estáticas: alta estabilidad, poco movimiento
            movement_score = 1.0 - min(quality['movement_avg'] / 0.05, 1.0)
            stability_score = quality['stability']
            type_score = movement_score * 0.7 + stability_score * 0.3
        
        elif sign_type == 'dynamic':
            # Para señas dinámicas: movimiento adecuado, cobertura completa
            movement_score = min(quality['movement_avg'] / 0.08, 1.0)
            coverage_score = quality['coverage']
            type_score = movement_score * 0.6 + coverage_score * 0.4
        
        else:  # phrase
            # Para frases: balance entre movimiento y estabilidad
            movement_score = min(quality['movement_avg'] / 0.06, 1.0)
            coverage_score = quality['coverage']
            type_score = movement_score * 0.5 + coverage_score * 0.5
        
        # Puntuación final (0-100)
        final_score = int(type_score * 100)
        
        return {
            'score': final_score,
            'movement_avg': quality['movement_avg'],
            'stability': quality['stability'],
            'coverage': quality['coverage'],
            'quality_level': 'EXCELENTE' if final_score >= 80 else
                           'BUENA' if final_score >= 60 else
                           'REGULAR' if final_score >= 40 else 'MALA'
        }

    def get_next_priority_sign(self):
        """Obtiene la siguiente seña prioritaria a recolectar"""
        if not self.plan:
            return None, None
        
        # Revisar prioridades
        for priority in self.plan['plan_recoleccion']['prioridades']:
            for sign in priority['items']:
                sign_path = f"data/sequences/{sign}"
                current_count = len(os.listdir(sign_path)) if os.path.exists(sign_path) else 0
                target = priority['objetivo_por_item']
                
                if current_count < target:
                    return sign, priority['tipo']
        
        return None, None

    def draw_collection_ui(self, frame):
        """Dibuja interfaz mejorada de recolección"""
        height, width = frame.shape[:2]
        
        # Panel principal
        panel_height = 200
        panel_y = height - panel_height
        cv2.rectangle(frame, (0, panel_y), (width, height), (40, 40, 40), -1)
        
        # Información de la seña actual
        if self.current_sign:
            sign_type_color = (100, 255, 100) if self.sign_type == 'static' else \
                             (255, 100, 100) if self.sign_type == 'dynamic' else \
                             (100, 100, 255)
            
            cv2.putText(frame, f"SEÑA: {self.current_sign}", (20, panel_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            cv2.putText(frame, f"TIPO: {self.sign_type.upper()}", (20, panel_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, sign_type_color, 2)
            
            # Configuración específica
            config = self.sign_config[self.sign_type]
            cv2.putText(frame, config['description'], (20, panel_y + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Duración recomendada
            cv2.putText(frame, f"Duracion: {config['duration']}s", (20, panel_y + 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Estado de grabación
        if self.recording:
            # Indicador de grabación
            cv2.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 70, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Información de lateralidad en tiempo real
            if self.current_frame_handedness:
                last_frame = self.current_frame_handedness[-1]
                hands_text = []
                if last_frame['right']:
                    hands_text.append("DERECHA")
                if last_frame['left']:
                    hands_text.append("IZQUIERDA")
                if hands_text:
                    cv2.putText(frame, f"Manos: {' + '.join(hands_text)}", (width - 250, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Progreso del buffer
            progress = len(self.frame_buffer) / self.sequence_length
            progress_width = int(300 * progress)
            cv2.rectangle(frame, (20, panel_y + 130), (320, panel_y + 150), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, panel_y + 130), (20 + progress_width, panel_y + 150), 
                         (0, 255, 0), -1)
            cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}", 
                       (20, panel_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instrucciones
        instructions = [
            "ESPACIO: Iniciar/Parar grabacion",
            "S: Cambiar seña",
            "Q: Salir",
            "R: Reiniciar sesion"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 350, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Contador de sesión
        cv2.putText(frame, f"Secuencias recolectadas: {self.sequence_count}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Próxima seña prioritaria
        next_sign, priority = self.get_next_priority_sign()
        if next_sign:
            cv2.putText(frame, f"Siguiente prioritaria: {next_sign} ({priority})", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    def save_sequence(self):
        """Guarda secuencia con evaluación de calidad"""
        if len(self.frame_buffer) < self.sequence_length:
            print(f"⚠️  Secuencia incompleta: {len(self.frame_buffer)}/{self.sequence_length}")
            return False
        
        # Crear directorio si no existe
        save_dir = f"data/sequences/{self.current_sign}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizar secuencia para análisis bidireccional
        optimized_sequence = self.optimize_sequence_for_bidirectional(self.frame_buffer)
        
        # Evaluar calidad
        quality = self.evaluate_sequence_quality(optimized_sequence, self.sign_type)
        
        # Analizar lateralidad de la secuencia
        handedness_analysis = self.analyze_sequence_handedness()
        
        # Nombre de archivo con timestamp, calidad y lateralidad
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handedness_suffix = self.get_handedness_suffix(handedness_analysis)
        filename = f"{timestamp}_q{quality['score']}_{handedness_suffix}.npy"
        filepath = os.path.join(save_dir, filename)
        
        # Guardar secuencia optimizada
        sequence_array = np.array(optimized_sequence)
        np.save(filepath, sequence_array)
        
        # Actualizar estadísticas de sesión
        if self.current_sign not in self.session_data['collected_signs']:
            self.session_data['collected_signs'][self.current_sign] = []
        
        self.session_data['collected_signs'][self.current_sign].append({
            'filename': filename,
            'quality': quality,
            'handedness': handedness_analysis,
            'timestamp': timestamp,
            'type': self.sign_type
        })
        
        self.session_data['quality_metrics'].append(quality)
        self.sequence_count += 1
        
        print(f"✅ Secuencia guardada: {filename}")
        print(f"📊 Calidad: {quality['quality_level']} ({quality['score']}/100)")
        print(f"📈 Movimiento promedio: {quality['movement_avg']:.4f}")
        print(f"👋 Lateralidad: {handedness_analysis['dominant_hand']} ({handedness_analysis['usage_stats']['right_percentage']:.1f}% derecha, {handedness_analysis['usage_stats']['left_percentage']:.1f}% izquierda)")
        
        # Limpiar buffer de lateralidad para próxima secuencia
        self.current_frame_handedness = []
        
        return True

    def save_session_report(self):
        """Guarda reporte de la sesión"""
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # Estadísticas de calidad
        if self.session_data['quality_metrics']:
            scores = [q['score'] for q in self.session_data['quality_metrics']]
            self.session_data['quality_summary'] = {
                'avg_score': np.mean(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'excellent_count': sum(1 for s in scores if s >= 80),
                'good_count': sum(1 for s in scores if 60 <= s < 80),
                'regular_count': sum(1 for s in scores if 40 <= s < 60),
                'poor_count': sum(1 for s in scores if s < 40)
            }
        
        # Guardar reporte
        report_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Reporte de sesión guardado: {report_filename}")

    def input_sign_name(self):
        """Solicita nombre de seña al usuario"""
        print("\n📝 CONFIGURACIÓN DE SEÑA")
        print("=" * 30)
        
        # Mostrar señas prioritarias
        next_sign, priority = self.get_next_priority_sign()
        if next_sign:
            print(f"🎯 Siguiente prioritaria: {next_sign} ({priority})")
            use_priority = input(f"¿Recolectar {next_sign}? (s/n): ").lower() == 's'
            if use_priority:
                sign = next_sign
            else:
                sign = input("Nombre de la seña: ").upper()
        else:
            sign = input("Nombre de la seña: ").upper()
        
        # Clasificar tipo automáticamente
        sign_type = self.classify_sign_type(sign)
        print(f"Tipo detectado: {sign_type}")
        
        # Permitir override manual
        override = input(f"Cambiar tipo? (static/dynamic/phrase) o Enter para mantener: ").strip()
        if override in ['static', 'dynamic', 'phrase']:
            sign_type = override
        
        self.current_sign = sign
        self.sign_type = sign_type
        
        print(f"✅ Configurado: {sign} ({sign_type})")
        return True

    def extract_landmarks_with_handedness(self, results):
        """Extrae landmarks considerando la lateralidad de las manos"""
        # Inicializar arrays para mano derecha e izquierda
        right_hand_landmarks = [0.0] * 63  # 21 landmarks * 3 coordenadas
        left_hand_landmarks = [0.0] * 63
        
        # Información de lateralidad para metadata
        hands_detected = {'right': False, 'left': False}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # MediaPipe da la lateralidad desde la perspectiva de la persona
                # (no del observador), pero como flippeamos el frame, debemos invertir
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Invertir lateralidad debido al flip del frame
                if hand_label == "Left":
                    actual_hand = "right"  # Su mano izquierda aparece a la derecha en el frame flippeado
                else:
                    actual_hand = "left"   # Su mano derecha aparece a la izquierda en el frame flippeado
                
                # Solo procesar si la confianza es alta
                if confidence > 0.7:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    
                    if actual_hand == "right":
                        right_hand_landmarks = landmarks
                        hands_detected['right'] = True
                    else:
                        left_hand_landmarks = landmarks
                        hands_detected['left'] = True
        
        # Almacenar información de lateralidad en el frame buffer metadata
        if not hasattr(self, 'current_frame_handedness'):
            self.current_frame_handedness = []
        self.current_frame_handedness.append(hands_detected)
        
        # Combinar landmarks: [mano_derecha(63), mano_izquierda(63)] = 126 total
        all_landmarks = right_hand_landmarks + left_hand_landmarks
        
        return all_landmarks

    def analyze_sequence_handedness(self):
        """Analiza la lateralidad de una secuencia completa"""
        if not self.current_frame_handedness:
            return {'dominant_hand': 'unknown', 'usage_stats': {}}
        
        right_count = sum(1 for frame in self.current_frame_handedness if frame['right'])
        left_count = sum(1 for frame in self.current_frame_handedness if frame['left'])
        both_count = sum(1 for frame in self.current_frame_handedness if frame['right'] and frame['left'])
        total_frames = len(self.current_frame_handedness)
        
        # Determinar mano dominante
        if right_count > left_count * 1.5:
            dominant_hand = 'right'
        elif left_count > right_count * 1.5:
            dominant_hand = 'left'
        elif both_count > total_frames * 0.3:
            dominant_hand = 'both'
        else:
            dominant_hand = 'mixed'
        
        usage_stats = {
            'right_frames': right_count,
            'left_frames': left_count,
            'both_frames': both_count,
            'total_frames': total_frames,
            'right_percentage': (right_count / total_frames * 100) if total_frames > 0 else 0,
            'left_percentage': (left_count / total_frames * 100) if total_frames > 0 else 0,
            'both_percentage': (both_count / total_frames * 100) if total_frames > 0 else 0
        }
        
        return {
            'dominant_hand': dominant_hand,
            'usage_stats': usage_stats
        }
    
    def get_handedness_suffix(self, handedness_analysis):
        """Genera sufijo para el nombre del archivo basado en lateralidad"""
        dominant = handedness_analysis['dominant_hand']
        
        suffix_map = {
            'right': 'RH',      # Right Hand
            'left': 'LH',       # Left Hand  
            'both': 'BH',       # Both Hands
            'mixed': 'MH',      # Mixed Hands
            'unknown': 'UH'     # Unknown Handedness
        }
        
        return suffix_map.get(dominant, 'UH')

    def optimize_sequence_for_bidirectional(self, sequence):
        """Optimiza secuencias para análisis bidireccional"""
        if len(sequence) < self.sequence_length:
            return sequence
        
        # Para señas dinámicas, asegurar que el movimiento esté centrado
        if self.sign_type == 'dynamic':
            # Detectar el inicio y fin del movimiento significativo
            movement_magnitudes = []
            for i in range(1, len(sequence)):
                movement = np.linalg.norm(np.array(sequence[i]) - np.array(sequence[i-1]))
                movement_magnitudes.append(movement)
            
            # Encontrar el rango de movimiento significativo
            threshold = np.mean(movement_magnitudes) + np.std(movement_magnitudes)
            significant_frames = [i for i, mag in enumerate(movement_magnitudes) if mag > threshold]
            
            if significant_frames:
                start_movement = max(0, significant_frames[0] - 5)  # Contexto previo
                end_movement = min(len(sequence), significant_frames[-1] + 10)  # Contexto posterior
                
                # Extraer secuencia centrada en el movimiento
                movement_sequence = sequence[start_movement:end_movement]
                
                # Rellenar o truncar a longitud exacta
                if len(movement_sequence) < self.sequence_length:
                    # Padding simétrico
                    pad_total = self.sequence_length - len(movement_sequence)
                    pad_start = pad_total // 2
                    pad_end = pad_total - pad_start
                    
                    # Usar el primer y último frame para padding
                    padded_sequence = ([movement_sequence[0]] * pad_start + 
                                     movement_sequence + 
                                     [movement_sequence[-1]] * pad_end)
                    return padded_sequence
                else:
                    # Truncar manteniendo el centro
                    excess = len(movement_sequence) - self.sequence_length
                    start_trim = excess // 2
                    return movement_sequence[start_trim:start_trim + self.sequence_length]
        
        # Para señas estáticas, usar la secuencia completa
        return sequence

    def run(self):
        """Ejecuta el colector mejorado"""
        print("🚀 COLECTOR DE DATOS MEJORADO")
        print("🎯 Especializado en señas dinámicas vs estáticas")
        print("=" * 50)
        
        # Configurar primera seña
        if not self.input_sign_name():
            return
        
        print("\n📹 Iniciando captura de video...")
        print("Presiona ESPACIO para iniciar/parar grabación")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Voltear frame
            frame = cv2.flip(frame, 1)
            
            # Procesar con MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Si está grabando, extraer landmarks con lateralidad
                if self.recording:
                    all_landmarks = self.extract_landmarks_with_handedness(results)
                    self.frame_buffer.append(all_landmarks)
                    
                    # Parar automáticamente si se llena el buffer
                    if len(self.frame_buffer) >= self.sequence_length:
                        self.recording = False
                        self.save_sequence()
                        self.frame_buffer = []
            
            # Dibujar UI
            self.draw_collection_ui(frame)
            
            # Mostrar frame
            cv2.imshow('Colector Mejorado - Señas Dinámicas', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                # Iniciar/parar grabación
                if not self.recording:
                    self.recording = True
                    self.frame_buffer = []
                    self.current_frame_handedness = []  # Limpiar buffer de lateralidad
                    print(f"🔴 Iniciando grabación de {self.current_sign}")
                else:
                    self.recording = False
                    if len(self.frame_buffer) > 0:
                        self.save_sequence()
                    self.frame_buffer = []
                    self.current_frame_handedness = []  # Limpiar buffer de lateralidad
                    print("⏹️  Grabación detenida")
            
            elif key == ord('s'):
                # Cambiar seña
                self.recording = False
                self.frame_buffer = []
                self.current_frame_handedness = []  # Limpiar buffer de lateralidad
                self.input_sign_name()
            
            elif key == ord('r'):
                # Reiniciar sesión
                self.save_session_report()
                self.session_data = {
                    'start_time': datetime.now().isoformat(),
                    'collected_signs': {},
                    'quality_metrics': []
                }
                self.sequence_count = 0
                print("🔄 Sesión reiniciada")
            
            elif key == ord('q'):
                break
        
        # Limpiar recursos
        self.save_session_report()
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Sesión de recolección finalizada")
        print(f"📊 Total recolectado: {self.sequence_count} secuencias")

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    collector.run()
