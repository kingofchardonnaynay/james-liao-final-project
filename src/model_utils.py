import json
import gcsfs
import requests
import os
import streamlit as st
from tensorflow.keras.models import load_model

# --- LOAD MODEL ---
def load_symbol_model():
    # Retrieve Google Cloud credentials from Streamlit secrets
    gcp_credentials = st.secrets["gcp"]
    
    # Write the credentials to a temporary file
    credentials_json_path = "/tmp/gcp_credentials.json"
    with open(credentials_json_path, "w") as f:
        json.dump(gcp_credentials, f)

    # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_json_path
    
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
