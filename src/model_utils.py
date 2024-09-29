import json
import gcsfs
from tensorflow.keras.models import load_model

# --- LOAD MODEL ---
def load_symbol_model():
    # Set up Google Cloud Storage FileSystem
    fs = gcsfs.GCSFileSystem()

    # Path to the model in GCS
    gcs_model_path = "gs://jl-final-project/model04.keras"

    # Open the model file from GCS
    with fs.open(gcs_model_path, 'rb') as model_file:
        model = load_model(model_file)

    return model

# Load class names from JSON
def load_class_names():
    url = https://github.com/kingofchardonnaynay/james-liao-final-project/blob/main/data/models/class_indices.json
    response = requests.get(url)
    
    if response.status_code == 200:
        class_indices = response.json()
        return {v: k for k, v in class_indices.items()}
    else:
        raise Exception(f"Error fetching class indices from GitHub: {response.status_code}")
