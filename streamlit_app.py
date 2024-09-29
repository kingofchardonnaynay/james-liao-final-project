import streamlit as st
from PIL import Image
import cv2
import numpy as np
from src.db_utils import log_user_activity, get_card_info, get_set_and_card_info, get_read_only_connection, get_logging_connection
from src.model_utils import load_symbol_model, load_class_names
from src.image_processing import bounding_box_roi
from src.ocr_utils import extract_xxx_from_ocr
import psycopg2

model = load_symbol_model()
load_class_names()

# Function to handle login
def login():
    st.header("Login")
    username = st.text_input("Enter your name:")
    
    if username:
        log_user_activity(username)
        st.success(f"Welcome, {username}!")
        return username
    
    return None

# Streamlit image upload
def upload_image():
    st.header("Upload an Image")

    uploaded_image = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_rgb = np.array(image)
        # Convert to BGR for OpenCV processing later, but don't display it in BGR
        st.session_state.image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        return image_rgb
    
    return None

# Function to process the image using both set symbol and text
def process_image_with_symbol_and_text(image_rgb, model):
    symbol_prediction, text_prediction, name_prediction = bounding_box_roi(image_rgb, model)


 # Debugging: Check the predictions
    print(f"Symbol Prediction: {symbol_prediction}, Text Prediction: {text_prediction}")


    # Check if both predictions are present
    if not symbol_prediction:
        st.write("Error: Symbol prediction could not be obtained.")
        return

    if not text_prediction:
        st.write("Error: Text prediction could not be obtained.")
        return

    card_info = get_card_info(symbol_prediction, text_prediction)

    # Check if card info was retrieved successfully
    if card_info:
        # Initialize a list to collect missing information messages
        missing_info = []

        set_name = card_info.get('Set')
        card_name = card_info.get('Card')
        collector_number = card_info.get('Collector Number')

        if not set_name:
            missing_info.append("Set Name is missing.")
        if not card_name:
            missing_info.append("Card Name is missing.")
        if not collector_number:
            missing_info.append("Collector Number is missing.")

        # Display the information if present
        if set_name:
            st.write(f"Set Name: {set_name}")
        if card_name:
            st.write(f"Card Name: {card_name}")
        if collector_number:
            st.write(f"Collector Number: {collector_number}")

        # If there are any missing pieces of information, display the messages
        if missing_info:
            for message in missing_info:
                st.write(f"Warning: {message}")

        # Question to confirm if this is the correct card
        is_this_your_card = st.radio("Is this your card?", ("Yes", "No"))

        if is_this_your_card == "Yes":
            st.success("Great! Have a good day!")
        elif is_this_your_card == "No":
            st.write("Proceeding with text-only prediction...")
            process_text_only()
    else:
        st.write("Card information could not be found.")

# Streamlit UI for processing the image
def upload_and_process():
    image_rgb = upload_image()

    if image_rgb is not None:
        if st.button("Process Image"):
            process_image_with_symbol_and_text(image_rgb, model)

# Function to handle text-only processing
def process_text_only():
    # Extract the image from session state
    image_rgb = st.session_state.image_rgb 
    text_prediction, name_result = bounding_box_roi(image_rgb, model)[1:3]

    # Debugging: Check if text_prediction is empty
    if not text_prediction:
        st.write("Error: Text prediction could not be obtained.")
        return

    st.write(f"Text Prediction: {text_prediction}")

    # Use the get_set_and_card_info function to process the text_prediction
    result = get_set_and_card_info(text_prediction)

    # Check for valid results
    if result is None:
        st.write("No information found for the provided text.")
        return

    # Extract relevant data from the result
    possible_ids = result.get("possible_ids", [])
    ocr_result = result.get("OCR_Result")

    # Handle cases for QQQ format or multiple possible sets
    if len(possible_ids) > 1 or (ocr_result.isdigit() and len(ocr_result) == 3):
        if ocr_result.isdigit() and len(ocr_result) == 3:
            # We have QQQ format, compare against name_result
            st.write("Processing QQQ format...")
            longest_name = max(name_result, key=len)
            if longest_name in result.get("Card", ""):
                st.markdown(f"Matched Card: **{longest_name}** in {result['Card']}")
        else:
            st.write("Multiple possible sets found. Processing...")
            matched_names = []
            for set_info in possible_ids:
                set_id, set_name = set_info
                xxx_value = extract_xxx_from_ocr(ocr_result)

                try:
                    # Query for the card in the single set ID
                    conn = get_read_only_connection()
                    cur = conn.cursor()
                    table_name = f'public."{set_id}"'
                    cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (xxx_value,))
                    card_result = cur.fetchone()
                    cur.close()
                    conn.close()

                    if card_result:
                        card_name = card_result[0]
                        st.write(f"Set Name: **{set_name}**, Card Name: **{card_name}**")
                        
                        # Check each name in name_result against card_name
                        for name in name_result:
                            if name in card_name:
                                matched_names.append((name, card_name, set_name))
                        
                except psycopg2.errors.UndefinedTable:
                    st.write(f"Table for Set: {set_name} (ID: {set_id}) cannot be found.")
                    continue
            
            # Select the longest matched name
            if matched_names:
                longest_matched_name_info = max(matched_names, key=lambda x: len(x[0]))
                longest_matched_name, matched_card_name, matched_set_name = longest_matched_name_info
                st.markdown(f"Matched Card: **{longest_matched_name}** in Set: **{matched_set_name}**, Card Name: **{matched_card_name}**")
            else:
                st.write("No matches found among possible cards.")

    else:
        st.write("No matching cards found for the given text.")

    st.info("These are all possible matches. Please upload another card.")

# Ensure that the session state is initialized
if 'image_rgb' not in st.session_state:
    st.session_state.image_rgb = None
if 'symbol_prediction' not in st.session_state:
    st.session_state.symbol_prediction = None
if 'text_prediction' not in st.session_state:
    st.session_state.text_prediction = None

# Function to reset the app to the upload state
def reset_to_upload():
    # Clear session state variables related to the image and any predictions
    for key in list(st.session_state.keys()):
        if key in ["image_rgb", "symbol_prediction", "text_prediction"]:
            del st.session_state[key]

    # Reset any other specific variables as needed
    st.session_state.image_rgb = None

    # Display a message indicating reset
    st.success("The app has been reset. Please upload a new image.")

# Main control function to handle the overall flow
def main():
    st.title("Pokémon Card Identifier App")
    
    # Login
    name = login()

    if name:
        # Image Upload using a form
        with st.form("upload-form", clear_on_submit=True):
            file = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])
            submitted = st.form_submit_button("UPLOAD")

        if submitted and file is not None:
            # Convert uploaded image to RGB
            image = Image.open(file)
            image_rgb = np.array(image)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Store image_rgb in session state
            st.session_state.image_rgb = image_rgb
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Step 3: Process Image using symbol and text predictions immediately after upload
            symbol_prediction, text_prediction, name_prediction = bounding_box_roi(image_rgb, model)

            # Store predictions in session state
            st.session_state.symbol_prediction = symbol_prediction
            st.session_state.text_prediction = text_prediction
            st.session_state.name_prediction = name_prediction
            # Display predictions
            st.header("Set Symbol and OCR Detection")
            st.write(f"Symbol Prediction: {symbol_prediction}")
            st.write(f"Text Prediction: {text_prediction}")
            st.write(f"Name Prediction: {name_prediction}")
            st.write("---")

            # Header for OCR detection only
            st.header("OCR Detection Only")
            process_text_only()

        # Persistent Result Button
        if st.button("Reset to Upload"):
            reset_to_upload()

# Ensure that the session state is initialized
if 'image_rgb' not in st.session_state:
    st.session_state.image_rgb = None

# Run the app
if __name__ == "__main__":
    main()
