import json
from tensorflow.keras.models import load_model

# --- LOAD MODEL ---
def load_symbol_model():
    # Load the pretrained Keras model for symbol recognition
    model = load_model(r"C:\Users\Jimmy\Desktop\final-project\PokemonTCG\models\model04.keras")
    return model

# Load class names from JSON
def load_class_names():
    with open(r'C:\Users\Jimmy\Desktop\final-project\class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}