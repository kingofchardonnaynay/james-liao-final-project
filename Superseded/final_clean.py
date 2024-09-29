# --- ALL NECESSARY IMPORTS ---
import streamlit as st
import cv2
from PIL import Image
import json
import pandas as pd
import psycopg2
import os
import easyocr
from tensorflow.keras.models import load_model
import numpy as np
import re



# --- LOAD MODEL ---
# Load the pretrained Keras model for symbol recognition
model = load_model(r"C:\Users\Jimmy\Desktop\final-project\PokemonTCG\models\model04.keras")

# Load the class names (saved class indices from JSON file)
with open(r'C:\Users\Jimmy\Desktop\final-project\class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the dictionary to map numeric labels to class names
class_names = {v: k for k, v in class_indices.items()}



# --- LOAD OCR ---
# Initialize EasyOCR reader (use 'en' for English)
reader = easyocr.Reader(['en'])



# --- LOAD POSTGRESQL CONNECTION ---
# Function to create a PostgreSQL connection as the readonly user for querying
def get_read_only_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="pokemontcg",
        user="readonly_user",
        password="D8G*pBDz*koJ"
    )
    return conn

# Function to connect as the logging user for logging, retrieving credentials from environment variables
def get_logging_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="pokemontcg",
        user=os.getenv("PG_LOGGING_USER"),
        password=os.getenv("PG_LOGGING_PASSWORD")
    )
    return conn

# Function to log the user's activity into the restricted_logs.user_logs table
def log_user_activity(username):
    conn = get_logging_connection()
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO restricted_logs.user_logs (username, login_time)
    VALUES (%s, CURRENT_TIMESTAMP);
    """
    cursor.execute(insert_query, (username,))

    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()



# --- CLASSIFY CARDS AS WHITE OR NON-WHITE ---
def classify_card(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    return "white" if mean_brightness > 150 else "non_white"



# --- OCR PREPROCESSING AND ADJUSTMENTS ---
# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=2, beta=25):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Preprocessing to identify all colors that are not black and turn them into white
def ocr_preprocessing(image_roi):
    roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh_black = cv2.threshold(roi_gray, 45, 255, cv2.THRESH_BINARY)
    
    roi_contrast = adjust_brightness_contrast(roi_thresh_black, alpha=1.5, beta=30)
    roi_blur = cv2.GaussianBlur(roi_contrast, (3, 3), 0)
    roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    roi_sharpen = cv2.filter2D(roi_thresh, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    
    debug_show_image(roi_sharpen, title="Preprocessed")
    return roi_sharpen

# OCR processing with EasyOCR
def perform_ocr_easyocr(image_roi):
    result = reader.readtext(image_roi, detail=0)
    print(f"OCR result: {result}")
    return result

# --- POSTPROCESS OCR OUTPUT ---
# Function to process OCR results and select only the valid match
def process_ocr_results(ocr_results):
    # Define the patterns for the different cases
    pattern_letters_digits = r'[a-zA-Z]{2,4}\d{1,3}'  # 2-4 letters followed by 1-3 numbers
    pattern_numbers_slash = r'\d{1,3}/\d{2,3}'  # 1-3 numbers followed by / and 2-3 numbers
    pattern_numbers_only = r'^\d{1,3}$'  # QQQ: 1-3 digits only

    for result in ocr_results:
        # Try to match all patterns
        match_letters_digits = re.search(pattern_letters_digits, result)
        match_numbers_slash = re.search(pattern_numbers_slash, result)
        match_numbers_only = re.search(pattern_numbers_only, result)

        # Return the matched portion if found
        if match_letters_digits:
            return match_letters_digits.group()  # Return the first valid match for letters-digits
        elif match_numbers_slash:
            return match_numbers_slash.group()  # Return the first valid match for numbers-slash
        elif match_numbers_only:
            return match_numbers_only.group()  # Return the first valid match for numbers only

    return None  # Return None if no valid match is found




# --- APPLY TRAINED MODEL TO SET ROIS ---
# Function to convert non white non black to green and crop
def convert_non_black_to_lime_green(roi):
    non_black_mask = np.any(roi > [50, 50, 50], axis=-1)
    processed_roi = roi.copy()
    processed_roi[non_black_mask] = [50, 205, 50]
    return processed_roi

# Crop the ROI to remove green areas
def crop_to_non_green_pixels(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    contours, _ = cv2.findContours(non_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        size = min(max(w, h), min(roi.shape[0], roi.shape[1]) - 1)
        crop_x = max(0, x + w // 2 - size // 2)
        crop_y = max(0, y + h // 2 - size // 2)
        roi_cropped = roi[crop_y:crop_y + size, crop_x:crop_x + size]
        return roi_cropped
    else:
        print("No non-green pixels found, returning original ROI.")
        return roi

# Remove green mask after cropping
def remove_green_mask(roi):
    lime_green_mask = np.all(roi == [50, 205, 50], axis=-1)
    restored_roi = roi.copy()
    restored_roi[lime_green_mask] = [255, 255, 255]
    return restored_roi

# Function to preprocess ROI for model
def preprocess_roi_for_model(roi):
    roi_green = convert_non_black_to_lime_green(roi)
    roi_cropped = crop_to_non_green_pixels(roi_green)
    roi_final = remove_green_mask(roi_cropped)
    return roi_final

# Function to apply the model to all three ROIs and return the top class name
def apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle):
    roi_bottom_left_processed = preprocess_roi_for_model(roi_bottom_left)
    roi_bottom_right_processed = preprocess_roi_for_model(roi_bottom_right)
    roi_middle_processed = preprocess_roi_for_model(roi_middle)

    roi_bottom_left_resized = cv2.resize(roi_bottom_left_processed, (150, 150))
    roi_bottom_right_resized = cv2.resize(roi_bottom_right_processed, (150, 150))
    roi_middle_resized = cv2.resize(roi_middle_processed, (150, 150))

    roi_bottom_left_exp = np.expand_dims(roi_bottom_left_resized, axis=0)
    roi_bottom_right_exp = np.expand_dims(roi_bottom_right_resized, axis=0)
    roi_middle_exp = np.expand_dims(roi_middle_resized, axis=0)

    prediction_bottom_left = model.predict(roi_bottom_left_exp)
    prediction_bottom_right = model.predict(roi_bottom_right_exp)
    prediction_middle = model.predict(roi_middle_exp)

    confidence_bottom_left = np.max(prediction_bottom_left)
    class_bottom_left = np.argmax(prediction_bottom_left)

    confidence_bottom_right = np.max(prediction_bottom_right)
    class_bottom_right = np.argmax(prediction_bottom_right)

    confidence_middle = np.max(prediction_middle)
    class_middle = np.argmax(prediction_middle)

    class_name_bottom_left = class_names.get(class_bottom_left, "Unknown")
    class_name_bottom_right = class_names.get(class_bottom_right, "Unknown")
    class_name_middle = class_names.get(class_middle, "Unknown")

    results = [
        (confidence_bottom_left, class_name_bottom_left),
        (confidence_bottom_right, class_name_bottom_right),
        (confidence_middle, class_name_middle)
    ]

    best_result = max(results, key=lambda x: x[0])
    best_class_name = best_result[1]

    return best_class_name



# --- BOUNDING FUNCTIONS ---
# Draw bounding box for white cards
def draw_bounding_box_white(image_bgr, model):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, "Contour cannot be found."  # Safety net for missing contours

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # --- Text ROIs ---
    # First Text ROI
    text_roi_height = h // 10
    text_roi_width = int(w * 0.5)
    text_roi_y_start = y + h - text_roi_height
    first_text_roi = image_bgr[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

    # Processing First Text ROI for OCR
    processed_first_text_roi = ocr_preprocessing(first_text_roi)
    ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

    # Second Text ROI
    second_text_roi_height = int(text_roi_height * 1.2)
    second_text_roi_x_start = x + w - text_roi_width
    second_text_roi_y_start = y + h - second_text_roi_height
    second_text_roi = image_bgr[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
                                second_text_roi_x_start:second_text_roi_x_start + text_roi_width]

    # Processing Second Text ROI for OCR
    processed_second_text_roi = ocr_preprocessing(second_text_roi)
    ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

    # Keep only one OCR result
    processed_ocr_result_first = process_ocr_results(ocr_results_first)
    processed_ocr_result_second = process_ocr_results(ocr_results_second)
    final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

    # Safety net for missing OCR results
    if not final_ocr_result:
        return None, "Text cannot be found."

    # --- Set Symbol ROIs ---
    roi_bottom_left_coords = (x, y + h - text_roi_height, w // 4, text_roi_height)
    roi_middle_coords = (x + 3 * w // 4, y + h // 2, w // 4, text_roi_height)
    roi_bottom_right_y_adjusted = y + h - 2 * text_roi_height + int(0.15 * text_roi_height)
    roi_bottom_right_coords = (x + w // 2 + w // 4, roi_bottom_right_y_adjusted, w // 4, text_roi_height)

    # Extract ROIs based on the coordinates
    roi_bottom_left = image_bgr[roi_bottom_left_coords[1]:roi_bottom_left_coords[1] + roi_bottom_left_coords[3],
                                roi_bottom_left_coords[0]:roi_bottom_left_coords[0] + roi_bottom_left_coords[2]]
    roi_middle = image_bgr[roi_middle_coords[1]:roi_middle_coords[1] + roi_middle_coords[3],
                        roi_middle_coords[0]:roi_middle_coords[0] + roi_middle_coords[2]]
    roi_bottom_right = image_bgr[roi_bottom_right_coords[1]:roi_bottom_right_coords[1] + roi_bottom_right_coords[3],
                                roi_bottom_right_coords[0]:roi_bottom_right_coords[0] + roi_bottom_right_coords[2]]

    # Get the symbol prediction
    symbol_prediction = apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle)

    # Safety net for missing model prediction
    if not symbol_prediction:
        return None, "Contour cannot be found."

    return symbol_prediction, final_ocr_result


# Function to display image for debugging
def debug_show_image(image, title="Debug Image"):
    """
    Function to display an image in Streamlit for debugging purposes.
    
    Parameters:
        image (numpy.ndarray): The image to display.
        title (str): The title to show above the image.
    """
    # Convert BGR to RGB if necessary (OpenCV uses BGR format)
    if len(image.shape) == 3:  # Check if the image is colored
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image  # Grayscale image, no conversion needed

    st.image(image_rgb, caption=title, use_column_width=True)


# Draw bounding box for non-white cards
def draw_bounding_boxes(image_rgb, threshold_value, model):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # First Text ROI
        text_roi_height = h // 10
        text_roi_width = int(w * 0.5)
        text_roi_y_start = y + h - text_roi_height
        first_text_roi = image_rgb[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

        # Debugging: show the first ROI before preprocessing
        debug_show_image(first_text_roi, title="First Text ROI Before Preprocessing (Non-White)")

        processed_first_text_roi = ocr_preprocessing(first_text_roi)
        ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

        # Second Text ROI
        second_text_roi_height = int(text_roi_height * 1.2)
        second_text_roi_x_start = x + w - text_roi_width
        second_text_roi_y_start = y + h - second_text_roi_height
        second_text_roi = image_rgb[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
                                    second_text_roi_x_start:second_text_roi_x_start + text_roi_width]

        # Debugging: show the second ROI before preprocessing
        debug_show_image(second_text_roi, title="Second Text ROI Before Preprocessing (Non-White)")

        processed_second_text_roi = ocr_preprocessing(second_text_roi)
        ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

        # Keep only one OCR result
        processed_ocr_result_first = process_ocr_results(ocr_results_first)
        processed_ocr_result_second = process_ocr_results(ocr_results_second)

        final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

        # Debugging: Check final OCR result
        print("Final OCR Result:", final_ocr_result)

        # Set Symbol ROIs
        roi_bottom_left_coords = (x, y + h - text_roi_height, w // 4, text_roi_height)
        roi_middle_coords = (x + 3 * w // 4, y + h // 2, w // 4, text_roi_height)
        roi_bottom_right_y_adjusted = y + h - 2 * text_roi_height + int(0.15 * text_roi_height)
        roi_bottom_right_coords = (x + w // 2 + w // 4, roi_bottom_right_y_adjusted, w // 4, text_roi_height)

        # Extract ROIs based on the coordinates
        roi_bottom_left = image_rgb[roi_bottom_left_coords[1]:roi_bottom_left_coords[1] + roi_bottom_left_coords[3],
                                    roi_bottom_left_coords[0]:roi_bottom_left_coords[0] + roi_bottom_left_coords[2]]
        roi_middle = image_rgb[roi_middle_coords[1]:roi_middle_coords[1] + roi_middle_coords[3],
                            roi_middle_coords[0]:roi_middle_coords[0] + roi_middle_coords[2]]
        roi_bottom_right = image_rgb[roi_bottom_right_coords[1]:roi_bottom_right_coords[1] + roi_bottom_right_coords[3],
                                    roi_bottom_right_coords[0]:roi_bottom_right_coords[0] + roi_bottom_right_coords[2]]



        # Get the symbol prediction
        symbol_prediction = apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle)

        # Safety net for missing model prediction
        if not symbol_prediction:
            return None, "Contour cannot be found."

        return symbol_prediction, final_ocr_result

    else:
        print("No contours found.")
        return image_rgb, None



# --- MASTER FUNCTION TO HANDLE BOTH WHITE AND NON-WHITE ---
# Main function to handle both white and non-white cards
def bounding_box_roi(image_rgb, model):
    card_type = classify_card(image_rgb)

    if card_type == "white":
        # Handle white cards without threshold_value
        image_with_bounding_box, ocr_result = draw_bounding_box_white(image_rgb, model)
    else:
        # Handle non-white cards with a fixed threshold_value of 160
        threshold_value = 160
        image_with_bounding_box, ocr_result = draw_bounding_boxes(image_rgb, threshold_value, model)

    return image_with_bounding_box, ocr_result



# --- TEXT EXTRACTION FROM OCR ---
# Extract YYY from OCR result
def extract_yyy_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[1]  # Extracts YYY part
    return None

# Function to extract XXX from OCR result
def extract_xxx_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[0]  # Extracts XXX part
    return ocr_result  # Return the entire string if no slash is present

# Function to handle AAAXXX format with error handling
def handle_aaaxxx_format(ocr_result, conn, cur):
    # Extract letters and digits from AAAXXX format
    letters = ''.join(filter(str.isalpha, ocr_result))
    digits = ''.join(filter(str.isdigit, ocr_result))

    if not letters or not digits:
        print("Invalid AAAXXX format: No letters or digits found")
        return None

    # Determine the table to query based on letters
    table_id = f"{letters.lower()}p"
    if letters.upper() == "HGSS":
        table_id = "hsp"

    try:
        # Query the pokemon_sets table to find the id and name
        cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(id) = %s", (table_id,))
        set_result = cur.fetchone()

        if not set_result:
            print(f"No set found for id = {table_id}")
            return None

        set_id, set_name = set_result

        # Query the table with the same name as the set id to find the card name
        table_name = f'public."{set_id}"'
        cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (ocr_result,))
        card_result = cur.fetchone()

        if not card_result:
            print(f"No card found with card_number = {ocr_result} in set {set_name}")
            return None

        card_name = card_result[0]
        return {"Set": set_name, "Card": card_name, "OCR_Result": ocr_result}

    except psycopg2.errors.UndefinedTable as e:
        # Handle the case where the table doesn't exist
        print(f"Table for Set: {table_id} does not exist: {e}")
        return None

    except psycopg2.Error as e:
        # Catch any other psycopg2 errors
        print(f"Database error occurred: {e}")
        return None

# Function to handle QQQ format
def handle_qqq_format(ocr_result, conn, cur):
    # We expect the format to be purely numeric (1-3 digits)
    if not ocr_result.isdigit():
        print("Invalid QQQ format")
        return None

    # Define the possible sets ('np' and 'svp')
    possible_sets = ["np", "svp"]
    possible_cards = []

    # Loop through both possible sets and query for card matches
    for set_id in possible_sets:
        # First, query pokemon_sets to get the name for the set ID
        cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(id) = %s", (set_id,))
        set_result = cur.fetchone()

        if not set_result:
            print(f"No set found for id = {set_id}")
            continue

        set_id, set_name = set_result

        # Now query the table with the set ID to find the matching card by number (ocr_result)
        table_name = f'public."{set_id}"'
        cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (ocr_result,))
        card_result = cur.fetchone()

        if card_result:
            card_name = card_result[0]
            possible_cards.append({"Set_Name": set_name, "Card": card_name, "Set_ID": set_id, "OCR_Result": ocr_result})

    if possible_cards:
        return possible_cards
    else:
        return "Card cannot be found."  # Return this message if no matches are found\

# Function for finding card using symbol and text
def get_card_info(symbol_prediction, text_prediction):
    try:
        # Ensure there is a symbol prediction
        if not symbol_prediction:
            return None, "Symbol prediction is missing"

        # Establish a connection to the database
        try:
            conn = get_read_only_connection()
        except OperationalError as e:
            return None, "Database connection failed"

        cur = conn.cursor()

        try:
            # 1. Query the pokemon_sets table to find the set ID and name based on the symbol_prediction
            cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(name) = %s", (symbol_prediction.lower(),))
            set_result = cur.fetchone()

            if not set_result:
                return None, "Set cannot be found"

            set_id, set_name = set_result

            # 2. Handle empty or invalid text_prediction
            if not text_prediction or text_prediction.strip() == "":
                return None, "Text prediction is missing or invalid"

            # 3. Handle the text_prediction format
            if "/" in text_prediction:  # If it's in the XXX/YYY format
                card_number = extract_xxx_from_ocr(text_prediction)

            elif re.match(r'^[A-Za-z]{2,4}\d{1,3}$', text_prediction):  # If it's in the AAAAXXX format
                card_number = extract_xxx_from_ocr(text_prediction)

            elif text_prediction.isdigit():  # If it's in the QQQ format (numeric only)
                # For QQQ format, check in both `np` and `svp` sets
                possible_cards = []
                for table_id in ["np", "svp"]:
                    cur.execute(f"SELECT card_name FROM public.\"{table_id}\" WHERE card_number = %s", (text_prediction,))
                    card_result = cur.fetchone()
                    if card_result:
                        card_name = card_result[0]
                        possible_cards.append({"Set": table_id, "Card": card_name})

                if not possible_cards:
                    return None, "Card cannot be found"

                return possible_cards  # Return list of possible cards for Streamlit to handle

            else:
                return None, "Invalid text format"

            # 4. Query the table with the set_id to find the card_name
            try:
                table_name = f'public."{set_id}"'
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (card_number,))
                card_result = cur.fetchone()
            except errors.UndefinedTable:
                return None, f"Set table {set_id} cannot be found"

            # If no matching card is found
            if not card_result:
                return None, "Card cannot be found"

            card_name = card_result[0]
            return {"Set": set_name, "Card": card_name, "Collector Number": card_number, "OCR_Result": text_prediction}

        finally:
            cur.close()
            conn.close()

    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"
    
# Function if set symbol is insufficient
def get_set_and_card_info(ocr_result):
    # Connect to the PostgreSQL database with readonly connection
    conn = get_read_only_connection()
    cur = conn.cursor()
    possible_ids = []  # List to store all possible ids
    try:
        # Handle XXX/YYY format
        if '/' in ocr_result:
            yyy_value = extract_yyy_from_ocr(ocr_result)
            xxx_value = extract_xxx_from_ocr(ocr_result)
            
            if not yyy_value or not xxx_value:
                print("Invalid OCR result format for XXX/YYY.")
                return "Card cannot be found."

            # Query the pokemon_sets table to find all possible ids and names based on printed_total (YYY)
            cur.execute("SELECT id, name FROM public.pokemon_sets WHERE printed_total = %s", (yyy_value,))
            set_results = cur.fetchall()  # Fetch all matching results
            
            if not set_results:
                print(f"No set found with printed_total = {yyy_value}")
                return "Card cannot be found."
            
            # Store all possible matching ids and names
            for set_result in set_results:
                set_id, set_name = set_result
                possible_ids.append((set_id, set_name))  # Store both id and name
            
            # If only one match is found, proceed normally
            if len(possible_ids) == 1:
                set_id, set_name = possible_ids[0]
                # Query the table with the same name as the set id to find the card name
                table_name = f'public."{set_id}"'
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (xxx_value,))
                card_result = cur.fetchone()
                if not card_result:
                    print(f"No card found with card_number = {xxx_value} in set {set_name}")
                    return "Card cannot be found."
                card_name = card_result[0]
                return {"Set_Name": set_name, "Card": card_name, "OCR_Result": ocr_result}
            
            # If multiple matches found, return possible ids and names
            return {"possible_ids": possible_ids, "OCR_Result": ocr_result}
        
        # Handle AAAXXX format or purely numeric string
        elif ocr_result.isdigit() and len(ocr_result) == 3:
            # Handle the QQQ format by querying both "np" and "svp"
            np_table_name = 'public."np"'
            svp_table_name = 'public."svp"'
            
            # Query np table
            cur.execute(f"SELECT card_name FROM {np_table_name} WHERE card_number = %s", (ocr_result,))
            np_card_result = cur.fetchone()
            
            # Query svp table
            cur.execute(f"SELECT card_name FROM {svp_table_name} WHERE card_number = %s", (ocr_result,))
            svp_card_result = cur.fetchone()

            possible_cards = []  # To store possible card results

            if np_card_result:
                possible_cards.append({"Set_Name": "np", "Card": np_card_result[0], "OCR_Result": ocr_result})

            if svp_card_result:
                possible_cards.append({"Set_Name": "svp", "Card": svp_card_result[0], "OCR_Result": ocr_result})

            if possible_cards:
                return {"possible_cards": possible_cards}
            else:
                return "Card cannot be found."

        elif ocr_result.isalnum():
            result = handle_aaaxxx_format(ocr_result, conn, cur)
            if result is None:
                return "Card cannot be found."
            return result
        
        # Handle unknown OCR formats
        else:
            print(f"Unrecognized OCR result format: {ocr_result}")
            return "Card cannot be found."

    except psycopg2.DatabaseError as db_error:
        print(f"Database error: {db_error}")
        return "An error occurred while accessing the database."

    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."

    finally:
        cur.close()
        conn.close()


# --- STREAMLIT FUNCTIONS ---
# Function to handle login
def login():
    st.header("Login")
    username = st.text_input("Enter your name:")
    
    if username:
        log_user_activity(username)  # Log the user's activity
        st.success(f"Welcome, {username}!")
        return username  # Return the username for later use
    
    return None  # Return None if no username is entered

# Streamlit image upload
def upload_image():
    st.header("Upload an Image")

    uploaded_image = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_rgb = np.array(image)  # Keep the image in RGB format
        # Convert to BGR for OpenCV processing later, but don't display it in BGR
        st.session_state.image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Store the BGR version for processing
        
        return image_rgb  # Return the RGB image for display
    
    return None  # Return None if no image is uploaded



# Function to process the image using both set symbol and text
def process_image_with_symbol_and_text(image_rgb, model):
    symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model)


 # Debugging: Check the predictions
    print(f"Symbol Prediction: {symbol_prediction}, Text Prediction: {text_prediction}")  # Debugging


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
            process_text_only()  # Call the function for text-only processing
    else:
        st.write("Card information could not be found.")


# Streamlit UI for processing the image
def upload_and_process():
    image_rgb = upload_image()  # Upload image function

    if image_rgb is not None:
        if st.button("Process Image"):
            process_image_with_symbol_and_text(image_rgb, model)  # Call the processing function

# Function to handle text-only processing
def process_text_only():
    # Extract the image from session state (assuming it's stored there)
    image_rgb = st.session_state.image_rgb 
    text_prediction = bounding_box_roi(image_rgb, model)[1]  # Get only text_prediction

    # Debugging: Check if text_prediction is empty
    if not text_prediction:
        st.write("Error: Text prediction could not be obtained.")
        return

    st.write(f"Text Prediction: {text_prediction}")

    # Check if text_prediction is empty
    if not text_prediction:
        st.write("Error: Text prediction could not be obtained.")
        return

    # Use the get_set_and_card_info function to process the text_prediction
    result = get_set_and_card_info(text_prediction)

    # Check for valid results
    if result is None:
        st.write("No information found for the provided text.")
        return

    # Extract possible_ids
    possible_ids = result.get("possible_ids", [])

    # Handle cases where possible_ids is empty
    if not possible_ids:
        st.write("No matching cards found for the given text.")
        return

    # If multiple possible ids, iterate through them
    if len(possible_ids) > 1:
        st.write(f"Multiple possible sets found. Total: {len(possible_ids)}")
        st.write("Iterating through the possible matches...")

        for set_info in possible_ids:
            set_id, set_name = set_info
            st.write("---")
            try:
                # Query the table with the set_id to find the card name
                conn = get_read_only_connection()
                cur = conn.cursor()
                table_name = f'public."{set_id}"'  # Use set_id for the table name
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (extract_xxx_from_ocr(text_prediction),))
                card_result = cur.fetchone()
                cur.close()
                conn.close()

                if card_result:
                    card_name = card_result[0]
                    st.markdown(f"Set: **{set_name}**")
                    st.markdown(f"Card Name: **{card_name}**")
                else:
                    st.write(f"No card with number **{extract_xxx_from_ocr(text_prediction)}** found for Set: **{set_name}** (ID: {set_id}).")

            except psycopg2.errors.UndefinedTable:
                st.write(f"Table for Set: {set_name} (ID: {set_id}) cannot be found.")
                continue  # Continue to the next set

        st.info("These are all possible matches. Please upload another card.")

    # If only one possible id, continue as normal
    else:
        set_id, set_name = possible_ids[0]  # Unpack the single possible id
        try:
            # Query for the card in the single set ID
            conn = get_read_only_connection()
            cur = conn.cursor()
            table_name = f'public."{set_id}"'
            cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (extract_xxx_from_ocr(text_prediction),))
            card_result = cur.fetchone()
            cur.close()
            conn.close()

            if card_result:
                card_name = card_result[0]
                st.write(f"Set Name: **{set_name}**")
                st.write(f"Card Name: **{card_name}**")
            else:
                st.write(f"No card with number **{extract_xxx_from_ocr(text_prediction)}** found in Set: **{set_name}** (ID: {set_id}).")

        except psycopg2.errors.UndefinedTable:
            st.write(f"Table for Set: {set_name} (ID: {set_id}) cannot be found.")

        st.info("This is our best guess! Please upload another card.")

# Ensure that the session state is initialized
if 'image_rgb' not in st.session_state:
    st.session_state.image_rgb = None  # Initialize as None for uploaded image
if 'symbol_prediction' not in st.session_state:
    st.session_state.symbol_prediction = None  # Initialize for symbol prediction
if 'text_prediction' not in st.session_state:
    st.session_state.text_prediction = None  # Initialize for text prediction

# Function to reset the app to the upload state
def reset_to_upload():
    # Clear session state variables related to the image and any predictions
    for key in list(st.session_state.keys()):
        if key in ["image_rgb", "symbol_prediction", "text_prediction"]:
            del st.session_state[key]

    # Reset any other specific variables as needed
    st.session_state.image_rgb = None  # Clear uploaded image

    # Display a message indicating reset
    st.success("The app has been reset. Please upload a new image.")

# Main control function to handle the overall flow
def main():
    st.title("Pokémon Card Identifier App")
    
    # Step 1: Login
    name = login()  # Call the login function

    if name:
        # Step 2: Image Upload using a form
        with st.form("upload-form", clear_on_submit=True):
            file = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])
            submitted = st.form_submit_button("UPLOAD")

        if submitted and file is not None:
            # Convert uploaded image to RGB
            image = Image.open(file)
            image_rgb = np.array(image)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing
            
            # Store image_rgb in session state
            st.session_state.image_rgb = image_rgb
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Step 3: Process Image using symbol and text predictions immediately after upload
            symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model)  # Use bounding_box_roi

            # Store predictions in session state
            st.session_state.symbol_prediction = symbol_prediction
            st.session_state.text_prediction = text_prediction
            
            # Display predictions
            st.header("Set Symbol and OCR Detection")
            st.write(f"Symbol Prediction: {symbol_prediction}")
            st.write(f"Text Prediction: {text_prediction}")
            st.write("---")  # Line separator

            # Header for OCR detection only
            st.header("OCR Detection Only")
            process_text_only()  # Call the process_text_only function for further OCR handling

        # Persistent Result Button
        if st.button("Reset to Upload"):
            reset_to_upload()

# Ensure that the session state is initialized
if 'image_rgb' not in st.session_state:
    st.session_state.image_rgb = None  # Initialize as None or your default value

# Run the app
if __name__ == "__main__":
    main()

