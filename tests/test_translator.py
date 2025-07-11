# test_translator.py
# Script de prueba para verificar que el traductor funciona correctamente

import numpy as np
from real_time_translator import BidirectionalRealTimeTranslator

def test_translator():
    """Prueba básica del traductor"""
    print("🧪 Probando Traductor Bidireccional...")
    
    try:
        # Crear instancia del traductor
        translator = BidirectionalRealTimeTranslator()
        
        print("✅ Traductor inicializado correctamente")
        print(f"📊 Modelo cargado: {translator.model is not None}")
        print(f"📋 Señas disponibles: {len(translator.signs)} señas")
        print(f"🎯 Señas encontradas: {', '.join(translator.signs)}")
        
        # Probar función de características de movimiento
        dummy_sequence = [np.random.rand(126) for _ in range(50)]
        motion_features = translator._calculate_motion_features(dummy_sequence)
        
        print(f"🔧 Características de movimiento: {len(motion_features)} features")
        print(f"📈 Rango de características: [{motion_features.min():.4f}, {motion_features.max():.4f}]")
        
        # Probar predicción (sin cámara)
        sequence_array = np.expand_dims(np.array(dummy_sequence), axis=0)
        motion_features_array = np.expand_dims(motion_features, axis=0)
        
        predictions = translator.model.predict([sequence_array, motion_features_array], verbose=0)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]
        predicted_sign = translator.signs[predicted_index]
        
        print(f"🎯 Predicción de prueba: {predicted_sign} (confianza: {confidence:.3f})")
        print("✅ Todas las pruebas pasaron correctamente!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_translator()
