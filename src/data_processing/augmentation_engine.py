# data_augmentation_engine.py
# Motor de Data Augmentation inteligente para señas dinámicas vs estáticas

import numpy as np
import os
import json
from datetime import datetime
import random
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class SignDataAugmentationEngine:
    def __init__(self, plan_path='plan_mejora_dataset.json', base_data_path='data/sequences'):
        self.plan_path = plan_path
        self.base_data_path = base_data_path
        self.sequence_length = 60  # Matches enhanced_data_collector
        
        # Cargar plan de mejora
        with open(plan_path, 'r', encoding='utf-8') as f:
            self.plan = json.load(f)
        
        # Configuración de augmentation por tipo de seña
        self.augmentation_config = {
            'static': {
                'noise_scale': [0.001, 0.003],
                'drift_scale': [0.002, 0.005],
                'temporal_shifts': [-3, -2, -1, 1, 2, 3],
                'scaling_factors': [0.95, 0.98, 1.02, 1.05],
                'rotation_angles': [-5, -2, 2, 5],  # degrees
                'hand_occlusion_prob': 0.1
            },
            'dynamic': {
                'speed_variations': [0.8, 0.9, 1.1, 1.2],
                'amplitude_variations': [0.85, 0.95, 1.05, 1.15],
                'trajectory_noise': [0.005, 0.01],
                'phase_shifts': [-0.1, -0.05, 0.05, 0.1],
                'smooth_variations': [0.8, 1.2],
                'hand_swap_prob': 0.15
            },
            'phrase': {
                'segment_variations': True,
                'pause_insertions': [0.1, 0.2],
                'rhythm_variations': [0.9, 1.1],
                'emphasis_variations': [0.8, 1.2],
                'natural_noise': [0.003, 0.007]
            }
        }

    def analyze_current_dataset(self):
        """Analiza el dataset actual para identificar déficits"""
        current_counts = {}
        total_sequences = 0
        
        for sign_folder in os.listdir(self.base_data_path):
            sign_path = os.path.join(self.base_data_path, sign_folder)
            if os.path.isdir(sign_path):
                count = len([f for f in os.listdir(sign_path) if f.endswith('.npy')])
                current_counts[sign_folder] = count
                total_sequences += count
        
        print(f"📊 Análisis del dataset actual:")
        print(f"   Total secuencias: {total_sequences}")
        
        # Calcular déficits según el plan
        deficits = {}
        for priority in self.plan['plan_recoleccion']['prioridades']:
            for sign in priority['items']:
                target = priority['objetivo_por_item']
                current = current_counts.get(sign, 0)
                deficit = max(0, target - current)
                if deficit > 0:
                    deficits[sign] = {
                        'current': current,
                        'target': target,
                        'deficit': deficit,
                        'priority': priority['tipo']
                    }
        
        return current_counts, deficits

    def classify_sign_type(self, sign):
        """Clasifica el tipo de seña para aplicar augmentation apropiado"""
        static_signs = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y'}
        
        dynamic_signs = {'J', 'Z', 'Ñ', 'RR', 'LL'}
        
        if sign in static_signs:
            return 'static'
        elif sign in dynamic_signs:
            return 'dynamic'
        else:
            return 'phrase'

    def apply_static_augmentation(self, sequence, variation_type):
        """Aplica augmentation específico para señas estáticas"""
        augmented = sequence.copy()
        config = self.augmentation_config['static']
        
        if variation_type == 'noise':
            # Ruido gaussiano sutil
            noise_scale = random.uniform(*config['noise_scale'])
            noise = np.random.normal(0, noise_scale, augmented.shape)
            augmented += noise
            
        elif variation_type == 'drift':
            # Deriva temporal suave
            drift_scale = random.uniform(*config['drift_scale'])
            t = np.linspace(0, 1, len(augmented))
            drift = np.outer(t, np.random.normal(0, drift_scale, augmented.shape[1]))
            augmented += drift
            
        elif variation_type == 'temporal_shift':
            # Desplazamiento temporal
            shift = random.choice(config['temporal_shifts'])
            if shift > 0:
                augmented[shift:] = augmented[:-shift]
                augmented[:shift] = augmented[shift]  # Repetir primer frame
            else:
                augmented[:shift] = augmented[-shift:]
                augmented[shift:] = augmented[-shift]  # Repetir último frame
                
        elif variation_type == 'scaling':
            # Escalado espacial
            scale = random.choice(config['scaling_factors'])
            augmented *= scale
            
        elif variation_type == 'rotation':
            # Rotación sutil (solo en X-Y)
            angle = np.radians(random.choice(config['rotation_angles']))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Aplicar rotación a coordenadas X-Y de cada landmark
            for i in range(0, augmented.shape[1], 3):
                x_coords = augmented[:, i]
                y_coords = augmented[:, i+1]
                
                augmented[:, i] = x_coords * cos_a - y_coords * sin_a
                augmented[:, i+1] = x_coords * sin_a + y_coords * cos_a
        
        return augmented

    def apply_dynamic_augmentation(self, sequence, variation_type):
        """Aplica augmentation específico para señas dinámicas"""
        augmented = sequence.copy()
        config = self.augmentation_config['dynamic']
        
        if variation_type == 'speed_variation':
            # Variación de velocidad temporal
            speed_factor = random.choice(config['speed_variations'])
            
            # Interpolación temporal
            original_t = np.linspace(0, 1, len(sequence))
            new_t = np.linspace(0, 1/speed_factor, len(sequence))
            
            # Limitar al rango [0, 1]
            new_t = np.clip(new_t, 0, 1)
            
            augmented_list = []
            for col in range(sequence.shape[1]):
                interp_func = interpolate.interp1d(original_t, sequence[:, col], 
                                                 kind='cubic', fill_value='extrapolate')
                augmented_col = interp_func(new_t)
                augmented_list.append(augmented_col)
            
            augmented = np.column_stack(augmented_list)
            
        elif variation_type == 'amplitude_variation':
            # Variación de amplitud del movimiento
            amplitude_factor = random.choice(config['amplitude_variations'])
            
            # Calcular centro de masa del movimiento
            center = np.mean(augmented, axis=0)
            
            # Escalar desde el centro
            augmented = center + (augmented - center) * amplitude_factor
            
        elif variation_type == 'trajectory_noise':
            # Ruido en la trayectoria manteniendo suavidad
            noise_scale = random.uniform(*config['trajectory_noise'])
            
            # Aplicar ruido suavizado
            for col in range(augmented.shape[1]):
                noise = np.random.normal(0, noise_scale, len(augmented))
                smoothed_noise = gaussian_filter1d(noise, sigma=1.5)
                augmented[:, col] += smoothed_noise
                
        elif variation_type == 'phase_shift':
            # Desplazamiento de fase para movimientos periódicos
            phase_shift = random.choice(config['phase_shifts'])
            shift_frames = int(phase_shift * len(augmented))
            
            if shift_frames != 0:
                augmented = np.roll(augmented, shift_frames, axis=0)
                
        elif variation_type == 'hand_swap':
            # Intercambio ocasional de manos (para señas simétricas)
            if random.random() < config['hand_swap_prob']:
                # Intercambiar landmarks de mano derecha (0-62) con izquierda (63-125)
                right_hand = augmented[:, :63].copy()
                left_hand = augmented[:, 63:126].copy()
                augmented[:, :63] = left_hand
                augmented[:, 63:126] = right_hand
        
        return augmented

    def apply_phrase_augmentation(self, sequence, variation_type):
        """Aplica augmentation específico para frases/expresiones"""
        augmented = sequence.copy()
        config = self.augmentation_config['phrase']
        
        if variation_type == 'rhythm_variation':
            # Variación del ritmo de la frase
            rhythm_factor = random.choice(config['rhythm_variations'])
            
            # Aplicar variación no uniforme del tiempo
            t_original = np.linspace(0, 1, len(sequence))
            t_varied = np.power(t_original, rhythm_factor)
            
            augmented_list = []
            for col in range(sequence.shape[1]):
                interp_func = interpolate.interp1d(t_original, sequence[:, col], 
                                                 kind='cubic', fill_value='extrapolate')
                augmented_col = interp_func(t_varied)
                augmented_list.append(augmented_col)
            
            augmented = np.column_stack(augmented_list)
            
        elif variation_type == 'natural_noise':
            # Ruido natural más complejo
            noise_scale = random.uniform(*config['natural_noise'])
            
            # Ruido correlacionado temporalmente
            for col in range(augmented.shape[1]):
                # Generar ruido con correlación temporal
                white_noise = np.random.normal(0, 1, len(augmented) + 10)
                colored_noise = gaussian_filter1d(white_noise, sigma=2)[:len(augmented)]
                augmented[:, col] += colored_noise * noise_scale
        
        return augmented

    def generate_augmented_sequence(self, base_sequence, sign_name, variation_type):
        """Genera una secuencia aumentada basada en una secuencia base"""
        sign_type = self.classify_sign_type(sign_name)
        
        if sign_type == 'static':
            return self.apply_static_augmentation(base_sequence, variation_type)
        elif sign_type == 'dynamic':
            return self.apply_dynamic_augmentation(base_sequence, variation_type)
        else:  # phrase
            return self.apply_phrase_augmentation(base_sequence, variation_type)

    def calculate_augmentation_quality(self, original, augmented, sign_type):
        """Evalúa la calidad de la augmentation"""
        # Similitud estructural
        similarity = np.corrcoef(original.flatten(), augmented.flatten())[0, 1]
        
        # Variación apropiada según tipo
        variation = np.mean(np.abs(original - augmented))
        
        if sign_type == 'static':
            # Para estáticas: alta similitud, baja variación
            quality = similarity * 0.7 + (1.0 - min(variation / 0.01, 1.0)) * 0.3
        elif sign_type == 'dynamic':
            # Para dinámicas: similitud moderada, variación controlada
            optimal_variation = 0.02
            variation_score = 1.0 - abs(variation - optimal_variation) / optimal_variation
            quality = similarity * 0.5 + variation_score * 0.5
        else:  # phrase
            # Para frases: balance entre similitud y naturalidad
            quality = similarity * 0.6 + (1.0 - min(variation / 0.015, 1.0)) * 0.4
        
        return max(0, min(1, quality))

    def augment_dataset_for_plan(self):
        """Ejecuta augmentation para cumplir con el plan de mejora"""
        print("🚀 MOTOR DE DATA AUGMENTATION")
        print("🎯 Generando datos para cumplir plan de mejora")
        print("=" * 50)
        
        current_counts, deficits = self.analyze_current_dataset()
        
        if not deficits:
            print("✅ Dataset ya cumple con el plan de mejora!")
            return
        
        print(f"\n📋 Déficits identificados:")
        for sign, info in deficits.items():
            print(f"   {sign}: {info['current']}/{info['target']} "
                  f"(falta {info['deficit']}, prioridad {info['priority']})")
        
        total_to_generate = sum(info['deficit'] for info in deficits.values())
        print(f"\n🎯 Total a generar: {total_to_generate} secuencias")
        
        generated_count = 0
        augmentation_report = {
            'timestamp': datetime.now().isoformat(),
            'generated_sequences': {},
            'quality_scores': [],
            'total_generated': 0
        }
        
        # Generar augmentations por prioridad
        priorities = ['CRÍTICO', 'ALTO', 'MEDIO', 'BAJO']
        
        for priority in priorities:
            print(f"\n🔥 Procesando prioridad {priority}...")
            
            for sign, info in deficits.items():
                if info['priority'] != priority:
                    continue
                
                sign_path = os.path.join(self.base_data_path, sign)
                if not os.path.exists(sign_path):
                    print(f"⚠️  No hay datos base para {sign}, saltando...")
                    continue
                
                # Cargar secuencias base existentes
                base_files = [f for f in os.listdir(sign_path) if f.endswith('.npy')]
                if not base_files:
                    print(f"⚠️  No hay secuencias .npy para {sign}, saltando...")
                    continue
                
                print(f"   Generando {info['deficit']} secuencias para {sign}...")
                
                sign_type = self.classify_sign_type(sign)
                sign_generated = 0
                sign_quality_scores = []
                
                # Configurar variaciones según tipo
                if sign_type == 'static':
                    variations = ['noise', 'drift', 'temporal_shift', 'scaling', 'rotation']
                elif sign_type == 'dynamic':
                    variations = ['speed_variation', 'amplitude_variation', 'trajectory_noise', 
                                'phase_shift', 'hand_swap']
                else:  # phrase
                    variations = ['rhythm_variation', 'natural_noise']
                
                # Generar hasta completar el déficit
                while sign_generated < info['deficit']:
                    # Seleccionar secuencia base aleatoria
                    base_file = random.choice(base_files)
                    base_sequence = np.load(os.path.join(sign_path, base_file))
                    
                    # Seleccionar tipo de variación
                    variation_type = random.choice(variations)
                    
                    # Generar secuencia aumentada
                    try:
                        augmented_sequence = self.generate_augmented_sequence(
                            base_sequence, sign, variation_type
                        )
                        
                        # Evaluar calidad
                        quality = self.calculate_augmentation_quality(
                            base_sequence, augmented_sequence, sign_type
                        )
                        
                        # Solo guardar si la calidad es aceptable
                        if quality >= 0.6:  # Umbral de calidad
                            # Generar nombre de archivo
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            filename = f"{timestamp}_q{int(quality*100)}_AUG_{variation_type}.npy"
                            filepath = os.path.join(sign_path, filename)
                            
                            # Guardar secuencia aumentada
                            np.save(filepath, augmented_sequence)
                            
                            sign_generated += 1
                            generated_count += 1
                            sign_quality_scores.append(quality)
                            
                            print(f"      ✅ {filename} (calidad: {quality:.3f})")
                        else:
                            print(f"      ❌ Calidad baja ({quality:.3f}), regenerando...")
                            
                    except Exception as e:
                        print(f"      ⚠️  Error generando variación {variation_type}: {e}")
                        continue
                
                # Actualizar reporte
                augmentation_report['generated_sequences'][sign] = {
                    'count': sign_generated,
                    'avg_quality': np.mean(sign_quality_scores) if sign_quality_scores else 0,
                    'variations_used': variations
                }
                augmentation_report['quality_scores'].extend(sign_quality_scores)
        
        # Finalizar reporte
        augmentation_report['total_generated'] = generated_count
        augmentation_report['avg_quality'] = np.mean(augmentation_report['quality_scores'])
        
        # Guardar reporte
        report_filename = f"augmentation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(augmentation_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ AUGMENTATION COMPLETADO!")
        print(f"📊 Total generado: {generated_count} secuencias")
        print(f"📈 Calidad promedio: {augmentation_report['avg_quality']:.3f}")
        print(f"📄 Reporte guardado: {report_filename}")
        
        # Verificación final
        print(f"\n🔍 Verificación final del dataset:")
        final_counts, final_deficits = self.analyze_current_dataset()
        
        if not final_deficits:
            print("🎉 ¡Plan de mejora completado!")
        else:
            print("⚠️  Algunos déficits persisten:")
            for sign, info in final_deficits.items():
                print(f"   {sign}: aún faltan {info['deficit']} secuencias")

if __name__ == "__main__":
    try:
        # Crear motor de augmentation
        augmenter = SignDataAugmentationEngine()
        
        # Ejecutar augmentation para cumplir plan
        augmenter.augment_dataset_for_plan()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que existe el archivo plan_mejora_dataset.json")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
