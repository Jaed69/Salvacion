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

    def determine_sign_type(self, sign):
        """Determina el tipo de seña basado en su nombre"""
        # Lista de señas estáticas conocidas
        static_signs = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'HOLA', 'GRACIAS', 'POR_FAVOR', 'DISCULPE'
        ]
        
        if sign.upper() in static_signs:
            return "Estática"
        else:
            return "Dinámica"

    def calculate_sequence_quality(self, sequence_data):
        """Calcula la calidad de una secuencia basada en varios factores"""
        if len(sequence_data) == 0:
            return 0.0
        
        try:
            # Factor 1: Completitud de la secuencia
            completeness = min(len(sequence_data) / self.sequence_length, 1.0)
            
            # Factor 2: Consistencia de detección (porcentaje de frames con landmarks válidos)
            valid_frames = sum(1 for frame in sequence_data if np.any(frame))
            consistency = valid_frames / len(sequence_data)
            
            # Factor 3: Estabilidad (variación en las posiciones)
            if valid_frames > 1:
                # Calcular variación promedio entre frames consecutivos
                variations = []
                for i in range(1, len(sequence_data)):
                    if np.any(sequence_data[i]) and np.any(sequence_data[i-1]):
                        diff = np.mean(np.abs(sequence_data[i] - sequence_data[i-1]))
                        variations.append(diff)
                
                if variations:
                    avg_variation = np.mean(variations)
                    # Normalizar la estabilidad (menos variación = mayor calidad)
                    stability = max(0, 1 - (avg_variation / 0.1))  # 0.1 es un umbral ajustable
                else:
                    stability = 0.5
            else:
                stability = 0.5
            
            # Combinar factores con pesos
            quality = (completeness * 0.4 + consistency * 0.4 + stability * 0.2) * 100
            return min(quality, 100.0)
            
        except Exception as e:
            print(f"Error calculando calidad: {e}")
            return 50.0  # Calidad por defecto en caso de error

    def determine_predominant_handedness(self):
        """Determina la mano predominante en la secuencia actual"""
        if not self.current_frame_handedness:
            return "unknown"
        
        # Contar ocurrencias de cada mano
        left_count = self.current_frame_handedness.count('Left')
        right_count = self.current_frame_handedness.count('Right')
        
        if left_count > right_count:
            return "left"
        elif right_count > left_count:
            return "right"
        else:
            return "both"

    def collect_data_for_sign_with_progress(self, sign, num_samples, output_dir="data/sequences"):
        """
        Método específico para recolectar datos por lotes con indicador de progreso
        """
        self.current_sign = sign.upper()
        self.sign_type = self.determine_sign_type(self.current_sign)
        
        # Configurar directorio de salida
        output_path = os.path.join(output_dir, self.current_sign)
        os.makedirs(output_path, exist_ok=True)
        
        collected_samples = 0
        samples_this_session = 0
        
        print(f"\n🎯 RECOLECTANDO: {self.current_sign} ({self.sign_type})")
        print(f"📊 Meta: {num_samples} muestras")
        print("=" * 50)
        print("📝 Instrucciones:")
        print("   ESPACIO: Iniciar/Parar grabación")
        print("   Q: Terminar recolección")
        print("   R: Reiniciar muestra actual")
        print("=" * 50)
        
        while collected_samples < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Voltear frame
            frame = cv2.flip(frame, 1)
            
            # Procesar con MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extraer landmarks si hay manos detectadas
            if results.multi_hand_landmarks:
                # Dibujar landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Si está grabando, agregar al buffer
                if self.recording:
                    landmarks_frame = self.extract_landmarks_with_handedness(results)
                    if landmarks_frame is not None:
                        self.frame_buffer.append(landmarks_frame)
                        
                        # Verificar si el buffer está completo
                        if len(self.frame_buffer) >= self.sequence_length:
                            self.recording = False
                            print(f"✅ Buffer completo ({self.sequence_length} frames)")
                            
                            # Guardar secuencia automáticamente
                            success = self.save_sequence_to_path(output_path)
                            if success:
                                collected_samples += 1
                                samples_this_session += 1
                                print(f"✅ Muestra {collected_samples}/{num_samples} guardada automáticamente")
                                
                                # Mostrar progreso cada 5 muestras
                                if collected_samples % 5 == 0:
                                    progress_percent = (collected_samples / num_samples) * 100
                                    print(f"📈 Progreso: {progress_percent:.1f}% ({collected_samples}/{num_samples})")
                            else:
                                print("❌ Error al guardar muestra")
                            
                            # Limpiar buffer para siguiente secuencia
                            self.frame_buffer = []
                            self.current_frame_handedness = []
                            print("🔄 Listo para siguiente secuencia")
            
            # Dibujar UI con progreso avanzado
            frame = self.draw_progress_ui(frame, collected_samples, num_samples, samples_this_session)
            
            # Mostrar frame
            cv2.imshow('Recolector LSP Esperanza - Modo Lote', frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                # Alternar grabación
                if not self.recording:
                    self.recording = True
                    self.frame_buffer = []
                    self.current_frame_handedness = []
                    print(f"🔴 Grabando muestra {collected_samples + 1}/{num_samples}...")
                else:
                    self.recording = False
                    if len(self.frame_buffer) >= self.sequence_length:
                        # Guardar secuencia
                        success = self.save_sequence_to_path(output_path)
                        if success:
                            collected_samples += 1
                            samples_this_session += 1
                            print(f"✅ Muestra {collected_samples}/{num_samples} guardada")
                            
                            # Mostrar progreso cada 5 muestras
                            if collected_samples % 5 == 0:
                                progress_percent = (collected_samples / num_samples) * 100
                                print(f"📈 Progreso: {progress_percent:.1f}% ({collected_samples}/{num_samples})")
                        else:
                            print("❌ Error al guardar muestra")
                    else:
                        print(f"⚠️  Secuencia muy corta: {len(self.frame_buffer)}/{self.sequence_length} frames")
                    
                    self.frame_buffer = []
                    self.current_frame_handedness = []
            
            elif key == ord('r'):
                # Reiniciar muestra actual
                if self.recording:
                    self.frame_buffer = []
                    self.current_frame_handedness = []
                    print("🔄 Muestra reiniciada")
            
            elif key == ord('q'):
                print(f"\n⏹️  Recolección detenida por el usuario")
                print(f"📊 Recolectadas: {collected_samples}/{num_samples} muestras")
                break
        
        # Limpiar recursos
        cv2.destroyAllWindows()
        
        if collected_samples >= num_samples:
            print(f"\n🎉 ¡RECOLECCIÓN COMPLETADA!")
            print(f"✅ {collected_samples} muestras recolectadas exitosamente")
        
        return collected_samples

    def draw_progress_ui(self, frame, current, target, session_count):
        """Dibuja UI avanzada con indicadores de progreso y calidad"""
        height, width = frame.shape[:2]
        
        # Panel principal expandido
        panel_height = 180
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Fondo gris oscuro
        
        # Borde del panel
        cv2.rectangle(panel, (0, 0), (width-1, panel_height-1), (100, 100, 100), 2)
        
        # === SECCIÓN SUPERIOR: INFORMACIÓN DE LA SEÑA ===
        # Título principal
        cv2.putText(panel, f"RECOLECTANDO: {self.current_sign}", (15, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Tipo de seña con color
        sign_type_color = (100, 255, 100) if self.sign_type == "Estática" else (255, 150, 100)
        cv2.putText(panel, f"TIPO: {self.sign_type.upper()}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, sign_type_color, 2)
        
        # === SECCIÓN PROGRESO GENERAL ===
        progress = (current / target) * 100 if target > 0 else 0
        progress_text = f"PROGRESO: {current}/{target} ({progress:.1f}%)"
        cv2.putText(panel, progress_text, (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        
        # Barra de progreso principal
        bar_width = width - 300
        bar_height = 25
        bar_x, bar_y = 15, 85
        
        # Fondo de la barra con borde
        cv2.rectangle(panel, (bar_x-1, bar_y-1), (bar_x + bar_width+1, bar_y + bar_height+1), (150, 150, 150), 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
        
        # Relleno de progreso
        if target > 0:
            fill_width = int((current / target) * bar_width)
            # Gradiente de color según progreso
            if progress < 25:
                color = (0, 100, 255)  # Azul - inicio
            elif progress < 50:
                color = (0, 200, 255)  # Cyan
            elif progress < 75:
                color = (100, 255, 100)  # Verde
            else:
                color = (100, 255, 255)  # Amarillo - casi completo
            
            cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            
            # Marcadores cada 25%
            for i in range(1, 4):
                marker_x = bar_x + int((i * 0.25) * bar_width)
                cv2.line(panel, (marker_x, bar_y), (marker_x, bar_y + bar_height), (200, 200, 200), 1)
        
        # === SECCIÓN ESTADO DE GRABACIÓN ===
        status_y = 120
        
        # Estado principal
        if self.recording:
            # Indicador de grabación parpadeante
            blink = int(time.time() * 3) % 2
            record_color = (0, 0, 255) if blink else (100, 100, 255)
            status_text = "🔴 GRABANDO"
            cv2.circle(panel, (width - 50, status_y - 5), 8, record_color, -1)
        else:
            # Verificar si acabamos de completar una secuencia
            if len(self.frame_buffer) == 0 and session_count > 0:
                status_text = "✅ SECUENCIA COMPLETADA"
                record_color = (0, 255, 0)
            else:
                status_text = "⚪ LISTO"
                record_color = (255, 255, 255)
        
        cv2.putText(panel, status_text, (15, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, record_color, 2)
        
        # === BARRA DE BUFFER (PROGRESO DE SECUENCIA ACTUAL) ===
        buffer_progress = len(self.frame_buffer) / self.sequence_length if self.sequence_length > 0 else 0
        buffer_bar_width = 200
        buffer_bar_height = 15
        buffer_bar_x = width - 250
        buffer_bar_y = status_y + 10
        
        # Fondo del buffer
        cv2.rectangle(panel, (buffer_bar_x, buffer_bar_y), 
                     (buffer_bar_x + buffer_bar_width, buffer_bar_y + buffer_bar_height), (80, 80, 80), -1)
        
        # Progreso del buffer
        if self.recording:
            buffer_fill_width = int(buffer_progress * buffer_bar_width)
            # Color según completitud
            if buffer_progress < 0.5:
                buffer_color = (0, 150, 255)  # Azul - llenando
            elif buffer_progress < 0.8:
                buffer_color = (0, 255, 150)  # Verde - casi listo
            else:
                buffer_color = (0, 255, 255)  # Amarillo - completo
            
            cv2.rectangle(panel, (buffer_bar_x, buffer_bar_y), 
                         (buffer_bar_x + buffer_fill_width, buffer_bar_y + buffer_bar_height), buffer_color, -1)
        elif len(self.frame_buffer) == 0 and session_count > 0:
            # Mostrar barra verde cuando se completó una secuencia
            cv2.rectangle(panel, (buffer_bar_x, buffer_bar_y), 
                         (buffer_bar_x + buffer_bar_width, buffer_bar_y + buffer_bar_height), (0, 255, 0), -1)
        
        # Texto del buffer
        if self.recording:
            buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        elif len(self.frame_buffer) == 0 and session_count > 0:
            buffer_text = "Secuencia completada - ESPACIO para siguiente"
        else:
            buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        
        cv2.putText(panel, buffer_text, (buffer_bar_x, buffer_bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # === INDICADOR DE CALIDAD EN TIEMPO REAL ===
        if self.recording and len(self.frame_buffer) > 10:
            # Calcular calidad preliminar
            recent_quality = self.calculate_realtime_quality()
            quality_color = (0, 255, 0) if recent_quality > 70 else (0, 255, 255) if recent_quality > 50 else (0, 100, 255)
            
            cv2.putText(panel, f"Calidad: {recent_quality:.0f}%", (width - 120, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
        
        # === INFORMACIÓN DE LATERALIDAD ===
        if self.recording and self.current_frame_handedness:
            hands_detected = self.current_frame_handedness[-1] if self.current_frame_handedness else {'right': False, 'left': False}
            hand_indicators = []
            
            if hands_detected.get('right', False):
                hand_indicators.append("🤚D")  # Derecha
            if hands_detected.get('left', False):
                hand_indicators.append("🤚I")   # Izquierda
            
            if hand_indicators:
                hands_text = " ".join(hand_indicators)
                cv2.putText(panel, hands_text, (width - 150, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # === INSTRUCCIONES RÁPIDAS ===
        instructions = [
            "ESPACIO: Grabar",
            "Q: Salir",
            "R: Reiniciar"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(panel, instruction, (width - 200, 50 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # === ESTADÍSTICAS DE SESIÓN ===
        if session_count > 0:
            session_text = f"Sesión: {session_count} muestras"
            cv2.putText(panel, session_text, (15, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # === INFORMACIÓN DE LOTE ACTUAL ===
        current_batch = (current // 20) + 1 if current > 0 else 1
        samples_in_batch = current % 20
        batch_text = f"Lote {current_batch} - Muestra {samples_in_batch + 1}/20"
        cv2.putText(panel, batch_text, (15, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Combinar panel con frame original
        combined_frame = np.vstack([panel, frame])
        
        return combined_frame

    def calculate_realtime_quality(self):
        """Calcula calidad en tiempo real durante la grabación"""
        if len(self.frame_buffer) < 5:
            return 0.0
        
        try:
            # Tomar últimos 10 frames para análisis rápido
            recent_frames = self.frame_buffer[-10:]
            
            # Factor 1: Consistencia de detección
            valid_frames = sum(1 for frame in recent_frames if np.any(frame))
            consistency = (valid_frames / len(recent_frames)) * 100
            
            # Factor 2: Estabilidad del movimiento
            if len(recent_frames) > 1:
                movements = []
                for i in range(1, len(recent_frames)):
                    if np.any(recent_frames[i]) and np.any(recent_frames[i-1]):
                        movement = np.mean(np.abs(np.array(recent_frames[i]) - np.array(recent_frames[i-1])))
                        movements.append(movement)
                
                if movements:
                    avg_movement = np.mean(movements)
                    # Para señas estáticas: penalizar mucho movimiento
                    # Para señas dinámicas: penalizar poco movimiento
                    if self.sign_type == "Estática":
                        movement_quality = max(0, 100 - (avg_movement * 1000))
                    else:
                        movement_quality = min(100, avg_movement * 500)
                else:
                    movement_quality = 50
            else:
                movement_quality = 50
            
            # Combinar factores
            overall_quality = (consistency * 0.7 + movement_quality * 0.3)
            return max(0, min(100, overall_quality))
            
        except Exception as e:
            return 50.0

    def save_sequence_to_path(self, output_path):
        """Guarda la secuencia en la ruta especificada con análisis de calidad completo"""
        try:
            # Preparar datos de la secuencia
            sequence_data = np.array(self.frame_buffer)
            
            # Análisis completo de calidad
            quality_score = self.calculate_sequence_quality(sequence_data)
            
            # Análisis de lateralidad
            handedness_analysis = self.analyze_sequence_handedness()
            
            # Análisis de movimiento específico por tipo
            movement_quality = self.evaluate_sequence_quality(self.frame_buffer, self.sign_type.lower().replace("ática", "atic").replace("ámica", "amic"))
            
            # Generar nombre de archivo descriptivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quality_str = f"q{int(quality_score)}"
            
            # Determinar lateralidad predominante
            handedness = self.get_handedness_suffix(handedness_analysis)
            
            # Nivel de calidad para el nombre
            quality_level = movement_quality['quality_level'].lower()[:3]  # exc, bue, reg, mal
            
            filename = f"{timestamp}_{quality_str}_{quality_level}_{handedness}.npy"
            filepath = os.path.join(output_path, filename)
            
            # Guardar archivo
            np.save(filepath, sequence_data)
            
            # Actualizar métricas de sesión
            self.sequence_count += 1
            
            # Información detallada para estadísticas
            sequence_info = {
                'sequence': self.sequence_count,
                'quality_score': quality_score,
                'movement_quality': movement_quality,
                'handedness_analysis': handedness_analysis,
                'timestamp': timestamp,
                'sign': self.current_sign,
                'sign_type': self.sign_type,
                'filename': filename,
                'sequence_length': len(sequence_data)
            }
            
            self.session_data['quality_metrics'].append(sequence_info)
            
            if self.current_sign not in self.session_data['collected_signs']:
                self.session_data['collected_signs'][self.current_sign] = 0
            self.session_data['collected_signs'][self.current_sign] += 1
            
            # Mostrar información detallada
            print(f"✅ Secuencia guardada: {filename}")
            print(f"📊 Calidad general: {quality_score:.1f}/100")
            print(f"🎯 Calidad por tipo: {movement_quality['quality_level']} ({movement_quality['score']}/100)")
            print(f"📈 Movimiento promedio: {movement_quality['movement_avg']:.4f}")
            print(f"🤲 Lateralidad: {handedness_analysis['dominant_hand']} " +
                  f"(D:{handedness_analysis['usage_stats']['right_percentage']:.1f}% " +
                  f"I:{handedness_analysis['usage_stats']['left_percentage']:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al guardar secuencia: {e}")
            return False

# Alias para compatibilidad con el script de recolección
DataCollector = EnhancedDataCollector

if __name__ == "__main__":
    collector = EnhancedDataCollector()
    collector.run()
