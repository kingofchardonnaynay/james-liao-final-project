# Import necessary modules
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



# --- LOAD POSTGRESQL CONNECTION
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

    # Insert the user's name and login time into the user_logs table
    insert_query = """
    INSERT INTO restricted_logs.user_logs (username, login_time)
    VALUES (%s, CURRENT_TIMESTAMP);
    """
    cursor.execute(insert_query, (username,))
    
    # Commit the transaction and close the connection
    conn.commit()
    cursor.close()
    conn.close()



# --- CLASSIFY CARDS AS WHITE OR NON-WHITE DUE PROCESSING ISSUES ---
# Function to classify a card as white or non-white
def classify_card(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    return "white" if mean_brightness > 150 else "non_white"



# --- OCR REQUIREMENTS ---
# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=2, beta=25):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# # Function to preprocess OCR
# def ocr_preprocessing(image_roi, threshold_value):
#     roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
#     _, roi_thresh_black = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY)
#     roi_contrast = adjust_brightness_contrast(roi_thresh_black, alpha=1.5, beta=30)
#     roi_blur = cv2.GaussianBlur(roi_contrast, (3, 3), 0)
#     roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                        cv2.THRESH_BINARY_INV, 11, 2)
#     roi_sharpen = cv2.filter2D(roi_thresh, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
#     return roi_sharpen

# Function to preprocess OCR
def ocr_preprocessing(image_roi):
    roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh_black = cv2.threshold(roi_gray, 30, 255, cv2.THRESH_BINARY)  # Use 30 as in the first code
    roi_contrast = adjust_brightness_contrast(roi_thresh_black, alpha=1.5, beta=30)
    roi_blur = cv2.GaussianBlur(roi_contrast, (3, 3), 0)
    roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
    roi_sharpen = cv2.filter2D(roi_thresh, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))

    return roi_sharpen


# OCR processing with EasyOCR
def perform_ocr_easyocr(image_roi):
    result = reader.readtext(image_roi, detail=0)
    return result

# # Function to process OCR results and select only the valid match
# def process_ocr_results(ocr_results):
#     pattern_letters_digits = r'[a-zA-Z]{2,4}\d{1,3}'  # 2-4 letters followed by 1-3 numbers
#     pattern_numbers_slash = r'\d{1,3}/\d{2,3}'  # 1-3 numbers followed by / and 2-3 numbers
#     for result in ocr_results:
#         match_letters_digits = re.search(pattern_letters_digits, result)
#         match_numbers_slash = re.search(pattern_numbers_slash, result)
#         if match_letters_digits:
#             return match_letters_digits.group()  # Return the first valid match for letters-digits
#         elif match_numbers_slash:
#             return match_numbers_slash.group()  # Return the first valid match for numbers-slash
#     return None  # Return None if no valid match is found

# Function to process OCR results and select only the valid match
def process_ocr_results(ocr_results):
    pattern_letters_digits = r'[a-zA-Z]{2,4}\d{1,3}'  # 2-4 letters followed by 1-3 numbers
    pattern_numbers_slash = r'\d{1,3}/\d{2,3}'  # 1-3 numbers followed by / and 2-3 numbers
    for result in ocr_results:
        match_letters_digits = re.search(pattern_letters_digits, result)
        match_numbers_slash = re.search(pattern_numbers_slash, result)
        if match_letters_digits:
            return match_letters_digits.group()  # Return the first valid match for letters-digits
        elif match_numbers_slash:
            return match_numbers_slash.group()  # Return the first valid match for numbers-slash
    return None  # Return None if no valid match is found


# --- APPLY TRAINED MODEL TO SET ROIS ---
# Function to convert non white non black to green and crop
def convert_non_black_to_lime_green(roi):
    # Create a mask for non-black pixels
    non_black_mask = np.any(roi > [50, 50, 50], axis=-1)
    
    # Create a copy of the ROI
    processed_roi = roi.copy()

    # Set non-black pixels to lime green
    processed_roi[non_black_mask] = [50, 205, 50]

    return processed_roi

def crop_to_non_green_pixels(roi):
    # Convert the ROI to HSV color space to help with color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the range for green color)
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([90, 255, 255])

    # Create a mask for green pixels
    green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

    # Invert the mask to get non-green areas
    non_green_mask = cv2.bitwise_not(green_mask)

    # Find contours of non-green areas
    contours, _ = cv2.findContours(non_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ensure the cropping is square and smaller than the original ROI
        size = min(max(w, h), min(roi.shape[0], roi.shape[1]) - 1)
        crop_x = max(0, x + w // 2 - size // 2)
        crop_y = max(0, y + h // 2 - size // 2)

        # Crop the ROI to a square around the largest non-green area
        roi_cropped = roi[crop_y:crop_y + size, crop_x:crop_x + size]

        return roi_cropped
    else:
        print("No non-green pixels found, returning original ROI.")
        return roi

# Function to remove green mask after cropping
def remove_green_mask(roi):
    """
    Removes the lime green pixels and returns the original colors.
    """
    # Create a mask for lime green pixels
    lime_green_mask = np.all(roi == [50, 205, 50], axis=-1)

    # Create a copy of the ROI
    restored_roi = roi.copy()

    # Set lime green pixels back to white
    restored_roi[lime_green_mask] = [255, 255, 255]

    return restored_roi

# # Function to apply the model to all three ROIs and return the top class name
# def apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle):
#     # Crop the ROIs to include only non-green pixels
#     roi_bottom_left_processed = convert_non_black_to_lime_green(roi_bottom_left)
#     roi_bottom_right_processed = convert_non_black_to_lime_green(roi_bottom_right)
#     roi_middle_processed = convert_non_black_to_lime_green(roi_middle)

#     roi_bottom_left_cropped = crop_to_non_green_pixels(roi_bottom_left_processed)
#     roi_bottom_right_cropped = crop_to_non_green_pixels(roi_bottom_right_processed)
#     roi_middle_cropped = crop_to_non_green_pixels(roi_middle_processed)

#     # Remove the green mask after cropping
#     roi_bottom_left_final = remove_green_mask(roi_bottom_left_cropped)
#     roi_bottom_right_final = remove_green_mask(roi_bottom_right_cropped)
#     roi_middle_final = remove_green_mask(roi_middle_cropped)

#     # Resize cropped ROIs to match model input size
#     roi_bottom_left_resized = cv2.resize(roi_bottom_left_final, (150, 150))
#     roi_bottom_right_resized = cv2.resize(roi_bottom_right_final, (150, 150))
#     roi_middle_resized = cv2.resize(roi_middle_final, (150, 150))

#     # Expand dimensions to match the model input shape (batch size, height, width, channels)
#     roi_bottom_left_exp = np.expand_dims(roi_bottom_left_resized, axis=0)
#     roi_bottom_right_exp = np.expand_dims(roi_bottom_right_resized, axis=0)
#     roi_middle_exp = np.expand_dims(roi_middle_resized, axis=0)

#     # Apply the model to each ROI
#     prediction_bottom_left = model.predict(roi_bottom_left_exp)
#     prediction_bottom_right = model.predict(roi_bottom_right_exp)
#     prediction_middle = model.predict(roi_middle_exp)

#     # Get confidence scores and class indices
#     confidence_bottom_left = np.max(prediction_bottom_left)
#     class_bottom_left = np.argmax(prediction_bottom_left)

#     confidence_bottom_right = np.max(prediction_bottom_right)
#     class_bottom_right = np.argmax(prediction_bottom_right)

#     confidence_middle = np.max(prediction_middle)
#     class_middle = np.argmax(prediction_middle)

#     # Map class index to class name
#     class_name_bottom_left = class_names.get(class_bottom_left, "Unknown")
#     class_name_bottom_right = class_names.get(class_bottom_right, "Unknown")
#     class_name_middle = class_names.get(class_middle, "Unknown")

#     # Collect the results with confidence scores
#     results = [
#         (confidence_bottom_left, class_name_bottom_left),
#         (confidence_bottom_right, class_name_bottom_right),
#         (confidence_middle, class_name_middle)
#     ]

#     # Sort the results by confidence score in descending order and return the top class name
#     best_result = max(results, key=lambda x: x[0])
#     best_class_name = best_result[1]

#     return best_class_name

def apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle):
    # Resize ROIs to model input size
    roi_bottom_left_resized = cv2.resize(roi_bottom_left, (150, 150))
    roi_bottom_right_resized = cv2.resize(roi_bottom_right, (150, 150))
    roi_middle_resized = cv2.resize(roi_middle, (150, 150))

    # Expand dimensions to match the model input shape (batch size, height, width, channels)
    roi_bottom_left_exp = np.expand_dims(roi_bottom_left_resized, axis=0)
    roi_bottom_right_exp = np.expand_dims(roi_bottom_right_resized, axis=0)
    roi_middle_exp = np.expand_dims(roi_middle_resized, axis=0)

    # Apply the model to each ROI
    prediction_bottom_left = model.predict(roi_bottom_left_exp)
    prediction_bottom_right = model.predict(roi_bottom_right_exp)
    prediction_middle = model.predict(roi_middle_exp)

    # Get confidence scores and class indices
    confidence_bottom_left = np.max(prediction_bottom_left)
    class_bottom_left = np.argmax(prediction_bottom_left)

    confidence_bottom_right = np.max(prediction_bottom_right)
    class_bottom_right = np.argmax(prediction_bottom_right)

    confidence_middle = np.max(prediction_middle)
    class_middle = np.argmax(prediction_middle)

    # Map class index to class name
    class_name_bottom_left = class_names.get(class_bottom_left, "Unknown")
    class_name_bottom_right = class_names.get(class_bottom_right, "Unknown")
    class_name_middle = class_names.get(class_middle, "Unknown")

    # Collect the results with confidence scores
    results = [
        (confidence_bottom_left, class_name_bottom_left),
        (confidence_bottom_right, class_name_bottom_right),
        (confidence_middle, class_name_middle)
    ]

    # Sort the results by confidence score in descending order and return the top class name
    best_result = max(results, key=lambda x: x[0])
    best_class_name = best_result[1]

    return best_class_name



# --- Combine ROI functions with OCR and model prediction ---
# Function to extract bounding boxes and run OCR on white cards
# def draw_bounding_box_white(image_bgr, model, threshold_value):
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # --- Text ROIs ---
#         # First Text ROI
#         text_roi_height = h // 10
#         text_roi_width = int(w * 0.5)
#         text_roi_y_start = y + h - text_roi_height
#         first_text_roi = image_bgr[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

#         # Processing First Text ROI for OCR using custom threshold
#         processed_first_text_roi = ocr_preprocessing(first_text_roi, threshold_value)
#         ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

#         # Draw the first ROI for debugging
#         cv2.rectangle(image_bgr, (x, text_roi_y_start), (x + text_roi_width, text_roi_y_start + text_roi_height), (255, 0, 0), 2)

#         # Second Text ROI
#         second_text_roi_height = int(text_roi_height * 1.2)
#         second_text_roi_x_start = x + w - text_roi_width
#         second_text_roi_y_start = y + h - second_text_roi_height
#         second_text_roi = image_bgr[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
#                                     second_text_roi_x_start:second_text_roi_x_start + text_roi_width]

#         # Processing Second Text ROI for OCR using custom threshold
#         processed_second_text_roi = ocr_preprocessing(second_text_roi, threshold_value)
#         ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

#          # Draw the second ROI for debugging
#         cv2.rectangle(image_bgr, (second_text_roi_x_start, second_text_roi_y_start), 
#                       (second_text_roi_x_start + text_roi_width, second_text_roi_y_start + second_text_roi_height), (0, 255, 0), 2)

#         # Keep only one OCR result
#         processed_ocr_result_first = process_ocr_results(ocr_results_first)
#         processed_ocr_result_second = process_ocr_results(ocr_results_second)
#         final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

#         # Check if final_ocr_result is blank and return a message if no text is found
#         if not final_ocr_result:
#             return symbol_prediction, "Text cannot be found"
        
#         # Convert the image back to RGB for display in Streamlit
#         image_rgb_debug = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#         # Display the image with ROIs
#         st.image(image_rgb_debug, caption="Image with Text ROIs", use_column_width=True)

#         # --- Set Symbol ROIs ---
#         # Define Set Symbol ROI coordinates relative to bounding box
#         roi_bottom_left_coords = (x, y + h - text_roi_height, w // 4, text_roi_height)
#         roi_middle_coords = (x + 3 * w // 4, y + h // 2, w // 4, text_roi_height)
#         roi_bottom_right_y_adjusted = y + h - 2 * text_roi_height + int(0.15 * text_roi_height)
#         roi_bottom_right_coords = (x + w // 2 + w // 4, roi_bottom_right_y_adjusted, w // 4, text_roi_height)

#         # Extract ROIs based on the coordinates
#         roi_bottom_left = image_bgr[roi_bottom_left_coords[1]:roi_bottom_left_coords[1] + roi_bottom_left_coords[3],
#                                     roi_bottom_left_coords[0]:roi_bottom_left_coords[0] + roi_bottom_left_coords[2]]
#         roi_middle = image_bgr[roi_middle_coords[1]:roi_middle_coords[1] + roi_middle_coords[3],
#                             roi_middle_coords[0]:roi_middle_coords[0] + roi_middle_coords[2]]
#         roi_bottom_right = image_bgr[roi_bottom_right_coords[1]:roi_bottom_right_coords[1] + roi_bottom_right_coords[3],
#                                     roi_bottom_right_coords[0]:roi_bottom_right_coords[0] + roi_bottom_right_coords[2]]

#         # Get the symbol prediction
#         symbol_prediction = apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle)

#         return symbol_prediction, final_ocr_result
#     else:
#         print("No contours found.")
#         return None, None
def draw_bounding_box_white(image_bgr, model, threshold_value):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # --- Text ROIs ---
        # First Text ROI
        text_roi_height = h // 10
        text_roi_width = int(w * 0.5)
        text_roi_y_start = y + h - text_roi_height
        first_text_roi = image_bgr[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

        # Draw the first ROI for debugging
        cv2.rectangle(image_bgr, (x, text_roi_y_start), (x + text_roi_width, text_roi_y_start + text_roi_height), (255, 0, 0), 2)

        # Processing First Text ROI for OCR using custom threshold
        processed_first_text_roi = ocr_preprocessing(first_text_roi)
        ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

        # Second Text ROI
        second_text_roi_height = int(text_roi_height * 1.2)
        second_text_roi_x_start = x + w - text_roi_width
        second_text_roi_y_start = y + h - second_text_roi_height
        second_text_roi = image_bgr[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
                                    second_text_roi_x_start:second_text_roi_x_start + text_roi_width]

        # Draw the second ROI for debugging
        cv2.rectangle(image_bgr, (second_text_roi_x_start, second_text_roi_y_start), 
                      (second_text_roi_x_start + text_roi_width, second_text_roi_y_start + second_text_roi_height), (0, 255, 0), 2)

        # Processing Second Text ROI for OCR using custom threshold
        processed_second_text_roi = ocr_preprocessing(second_text_roi)
        ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

        # Keep only one OCR result
        processed_ocr_result_first = process_ocr_results(ocr_results_first)
        processed_ocr_result_second = process_ocr_results(ocr_results_second)
        final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

        # Check if final_ocr_result is blank and return a message if no text is found
        if not final_ocr_result:
            return None, "Text cannot be found"

        # Convert the image back to RGB for display in Streamlit
        image_rgb_debug = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Display the image with ROIs in Streamlit
        st.image(image_rgb_debug, caption="Image with Text ROIs", use_column_width=True)

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

        return symbol_prediction, final_ocr_result
    else:
        print("No contours found.")
        return None, None

# Function to extract bounding boxes and run OCR on non-white cards
# def draw_bounding_boxes(image_bgr, fixed_threshold_value, model, threshold_value):
#     gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Initialize symbol_prediction as None or a default value
#     symbol_prediction = None

#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # --- Text ROIs ---
#         # First Text ROI
#         text_roi_height = h // 10
#         text_roi_width = int(w * 0.5)
#         text_roi_y_start = y + h - text_roi_height
#         first_text_roi = image_bgr[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

#         # Processing First Text ROI for OCR
#         processed_first_text_roi = ocr_preprocessing(first_text_roi, threshold_value)
#         ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

#         # Draw the first ROI for debugging
#         cv2.rectangle(image_bgr, (x, text_roi_y_start), (x + text_roi_width, text_roi_y_start + text_roi_height), (255, 0, 0), 2)

#         # Second Text ROI
#         second_text_roi_height = int(text_roi_height * 1.2)
#         second_text_roi_x_start = x + w - text_roi_width
#         second_text_roi_y_start = y + h - second_text_roi_height
#         second_text_roi = image_bgr[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
#                                     second_text_roi_x_start:second_text_roi_x_start + text_roi_width]
        
#         # Draw the second ROI for debugging
#         cv2.rectangle(image_bgr, (second_text_roi_x_start, second_text_roi_y_start), 
#                       (second_text_roi_x_start + text_roi_width, second_text_roi_y_start + second_text_roi_height), (0, 255, 0), 2)

#         # Processing Second Text ROI for OCR
#         processed_second_text_roi = ocr_preprocessing(second_text_roi, threshold_value)
#         ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

#         # Keep only one OCR result
#         processed_ocr_result_first = process_ocr_results(ocr_results_first)
#         processed_ocr_result_second = process_ocr_results(ocr_results_second)
#         final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

#         # Convert the image back to RGB for display in Streamlit
#         image_rgb_debug = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#         # Display the image with ROIs in Streamlit
#         st.image(image_rgb_debug, caption="Image with Text ROIs", use_column_width=True)

#         # --- Set Symbol ROIs ---
#         # Define Set Symbol ROI coordinates relative to bounding box
#         roi_bottom_left_coords = (x, y + h - text_roi_height, w // 4, text_roi_height)
#         roi_middle_coords = (x + 3 * w // 4, y + h // 2, w // 4, text_roi_height)
#         roi_bottom_right_y_adjusted = y + h - 2 * text_roi_height + int(0.15 * text_roi_height)
#         roi_bottom_right_coords = (x + w // 2 + w // 4, roi_bottom_right_y_adjusted, w // 4, text_roi_height)

#         # Extract ROIs based on the coordinates
#         roi_bottom_left = image_bgr[roi_bottom_left_coords[1]:roi_bottom_left_coords[1] + roi_bottom_left_coords[3],
#                                     roi_bottom_left_coords[0]:roi_bottom_left_coords[0] + roi_bottom_left_coords[2]]
#         roi_middle = image_bgr[roi_middle_coords[1]:roi_middle_coords[1] + roi_middle_coords[3],
#                             roi_middle_coords[0]:roi_middle_coords[0] + roi_middle_coords[2]]
#         roi_bottom_right = image_bgr[roi_bottom_right_coords[1]:roi_bottom_right_coords[1] + roi_bottom_right_coords[3],
#                                     roi_bottom_right_coords[0]:roi_bottom_right_coords[0] + roi_bottom_right_coords[2]]

#         # Get the symbol prediction
#         symbol_prediction = apply_model_to_rois(model, roi_bottom_left, roi_bottom_right, roi_middle)

#         return symbol_prediction, final_ocr_result
#     else:
#         # If no contours were found, return a message and no symbol prediction
#         return None, "Text cannot be found"
def draw_bounding_boxes(image_bgr, fixed_threshold_value, model, threshold_value):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, fixed_threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize symbol_prediction as None or a default value
    symbol_prediction = None

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # --- Text ROIs ---
        # First Text ROI
        text_roi_height = h // 10
        text_roi_width = int(w * 0.5)
        text_roi_y_start = y + h - text_roi_height
        first_text_roi = image_bgr[text_roi_y_start:text_roi_y_start + text_roi_height, x:x + text_roi_width]

        # Draw the first ROI for debugging
        cv2.rectangle(image_bgr, (x, text_roi_y_start), (x + text_roi_width, text_roi_y_start + text_roi_height), (255, 0, 0), 2)

        # Processing First Text ROI for OCR
        processed_first_text_roi = ocr_preprocessing(first_text_roi)
        ocr_results_first = perform_ocr_easyocr(processed_first_text_roi)

        # Second Text ROI
        second_text_roi_height = int(text_roi_height * 1.2)
        second_text_roi_x_start = x + w - text_roi_width
        second_text_roi_y_start = y + h - second_text_roi_height
        second_text_roi = image_bgr[second_text_roi_y_start:text_roi_y_start + second_text_roi_height,
                                    second_text_roi_x_start:second_text_roi_x_start + text_roi_width]

        # Draw the second ROI for debugging
        cv2.rectangle(image_bgr, (second_text_roi_x_start, second_text_roi_y_start), 
                      (second_text_roi_x_start + text_roi_width, second_text_roi_y_start + second_text_roi_height), (0, 255, 0), 2)

        # Processing Second Text ROI for OCR
        processed_second_text_roi = ocr_preprocessing(second_text_roi)
        ocr_results_second = perform_ocr_easyocr(processed_second_text_roi)

        # Keep only one OCR result
        processed_ocr_result_first = process_ocr_results(ocr_results_first)
        processed_ocr_result_second = process_ocr_results(ocr_results_second)
        final_ocr_result = processed_ocr_result_first if processed_ocr_result_first else processed_ocr_result_second

        # Convert the image back to RGB for display in Streamlit
        image_rgb_debug = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Display the image with ROIs in Streamlit
        st.image(image_rgb_debug, caption="Image with Text ROIs", use_column_width=True)

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

        return symbol_prediction, final_ocr_result
    else:
        # If no contours were found, return a message and no symbol prediction
        return None, "Text cannot be found"
    
# # Main function to handle both white and non-white cards
# def bounding_box_roi(image_rgb, model, threshold_value):
#     card_type = classify_card(image_rgb)

#     if card_type == "white":
#         # Handle white cards with custom threshold_value from Streamlit
#         image_with_bounding_box, ocr_result = draw_bounding_box_white(image_rgb, model, threshold_value)
#     else:
#         # Handle non-white cards with a fixed threshold_value of 160
#         fixed_threshold_value = 160
#         image_with_bounding_box, ocr_result = draw_bounding_boxes(image_rgb, fixed_threshold_value, model, threshold_value)

#     return image_with_bounding_box, ocr_result
# Main function to handle both white and non-white cards
def bounding_box_roi(image_rgb, model, threshold_value):
    card_type = classify_card(image_rgb)

    if card_type == "white":
        # Handle white cards with custom threshold_value from Streamlit
        image_with_bounding_box, ocr_result = draw_bounding_box_white(image_rgb, model, threshold_value)
    else:
        # Handle non-white cards with a fixed threshold_value of 160
        fixed_threshold_value = 160
        image_with_bounding_box, ocr_result = draw_bounding_boxes(image_rgb, fixed_threshold_value, model, threshold_value)

    return image_with_bounding_box, ocr_result


# --- PREPROCESSING FOR DATABASE ---
# Function to extract YYY from OCR result
def extract_yyy_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[1]  # Extracts YYY part
    return None

# Function to extract XXX from OCR result
def extract_xxx_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[0]  # Extracts XXX part
    return ocr_result  # Return the entire string if no slash is present

# Function to handle AAAXXX format
def handle_aaaxxx_format(ocr_result, conn, cur):
    # Extract letters and digits from AAAXXX format
    letters = ''.join(filter(str.isalpha, ocr_result))
    digits = ''.join(filter(str.isdigit, ocr_result))

    if not letters or not digits:
        print("Invalid AAAXXX format")
        return None

    # Determine the table to query based on letters
    table_id = f"{letters.lower()}p"
    if letters.upper() == "HGSS":
        table_id = "hsp"

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

# Main function to get card information using symbol and text predictions
def get_card_info(symbol_prediction, text_prediction):
    # Establish a connection to the database
    conn = get_read_only_connection()
    cur = conn.cursor()
    try:
        # 1. Query the pokemon_sets table to find the set ID and name based on the symbol_prediction
        cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(name) = %s", (symbol_prediction.lower(),))
        set_result = cur.fetchone()
        
        if not set_result:
            return None
        
        set_id, set_name = set_result

        # 2. If the text_prediction contains "/", handle the XXX/YYY format
        if "/" in text_prediction:
            card_number = extract_xxx_from_ocr(text_prediction)
            
            if card_number:
                # Query the table with the same name as the set_id to find the card_name
                table_name = f'public."{set_id}"'
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (card_number,))
                card_result = cur.fetchone()

                if card_result:
                    card_name = card_result[0]
                    return {"Set": set_name, "Card": card_name, "Collector Number": card_number, "OCR_Result": text_prediction}
                else:
                    return None

        # 3. If the text_prediction is in the AAAXXX format, handle using the `handle_aaaxxx_format` function
        elif re.match(r'^[A-Za-z]{3}\d{3}$', text_prediction):
            result = handle_aaaxxx_format(text_prediction, conn, cur)
            if result:
                return result

        # No valid format detected
        st.write(f"Invalid OCR format for text_prediction: {text_prediction}")
        return None

    finally:
        cur.close()
        conn.close()

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
            
            # Query the pokemon_sets table to find all possible ids and names based on printed_total (YYY)
            cur.execute("SELECT id, name FROM public.pokemon_sets WHERE printed_total = %s", (yyy_value,))
            set_results = cur.fetchall()  # Fetch all matching results
            
            if not set_results:
                print(f"No set found with printed_total = {yyy_value}")
                return None
            
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
                    return None
                card_name = card_result[0]
                return {"Set_Name": set_name, "Card": card_name, "OCR_Result": ocr_result}
            
            # If multiple matches found, return possible ids and names
            return {"possible_ids": possible_ids, "OCR_Result": ocr_result}
        
        # Handle AAAXXX format or purely numeric string
        elif ocr_result.isdigit() or ocr_result.isalnum():
            return handle_aaaxxx_format(ocr_result, conn, cur)
        else:
            print(f"Unrecognized OCR result format: {ocr_result}")
            return None
    finally:
        cur.close()
        conn.close()



# --- STREAMLIT FUNCTIONS ---
# Function to handle login
def login_page():
    st.header("Step 1: Login")
    name = st.text_input("Enter your name to log in:")
    if name:
        st.success(f"Welcome, {name}!")
    return name

# Function for uploading an image and setting the threshold
def upload_image_and_set_threshold():
    st.header("Upload an Image and Set OCR Black Threshold")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])

    # Add warning or informative text before the slider
    st.warning("Adjust the black threshold to improve OCR results for white cards.")

    # Custom threshold input for white cards (default set to 70)
    threshold_value = st.slider("Set the black threshold for OCR processing on white cards", 
                                min_value=20, max_value=100, value=30,
                                help="Adjust the threshold based on your image.")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_column_width=True)
        image_rgb = np.array(image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing
        
        return image_rgb, threshold_value
    return None, None

# Processing function that passes the threshold value to ocr_preprocessing1
def process_image_with_custom_threshold(image_rgb, threshold_value, model):
    if image_rgb is not None:
        # Call bounding_box_roi and pass the custom threshold_value for white cards
        image_with_bounding_box, ocr_result = bounding_box_roi(image_rgb, model, threshold_value)

        # Show OCR results and predictions
        st.write(f"OCR Result: {ocr_result if ocr_result else 'Text cannot be found'}")


# Function to process the image and predict the symbol and text
def process_image(image_rgb, model, threshold_value):
    
    # Use your actual model prediction here
    symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model, threshold_value)

    return symbol_prediction, text_prediction

# # Function to handle symbol and text processing
# def process_symbol_and_text_page(threshold_value):
#     st.title("Step 3: Processing the image...")

#     image_rgb = st.session_state.image_rgb
#     symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model, threshold_value)

#     st.write(f"Symbol Prediction: {symbol_prediction}")
#     st.write(f"Text Prediction: {text_prediction}")

#     card_info = get_card_info(symbol_prediction, text_prediction)

#     # Check if card_info is None (symbol is incorrect or card not found)
#     if card_info is None:
#         st.write("Card not found. Proceeding with text-only prediction...")
#         process_text_only_page(threshold_value) 
#         return
    
#     set_name = card_info.get('Set', 'Unknown Set')
#     card_name = card_info.get('Card', 'Unknown Card')
#     collector_number = card_info.get('Collector Number', 'Unknown Collector Number')

#     st.write(f"Set Name: {set_name}")
#     st.write(f"Card Name: {card_name}")
#     st.write(f"Collector Number: {collector_number}")

#     # Adding unique keys to buttons
#     if st.button("Yes", key="confirm_yes"):
#         st.success("Great! You can upload another card.")
#         reset_to_upload()

#     elif st.button("No", key="confirm_no"):
#         st.write("Proceeding with text prediction only...")
#         process_text_only_page(threshold_value)

# Function to handle symbol and text processing
def process_symbol_and_text_page(threshold_value):
    st.title("Step 3: Processing the image...")

    # Check if the image_rgb is in session state before accessing it
    if "image_rgb" not in st.session_state:
        st.error("Image not uploaded. Please go back and upload an image.")
        return

    image_rgb = st.session_state.image_rgb  # Safely access the image
    symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model, threshold_value)

    st.write(f"Symbol Prediction: {symbol_prediction}")
    st.write(f"Text Prediction: {text_prediction}")

    # If text_prediction is None, show an error and skip further processing
    if text_prediction is None:
        st.error("No text prediction found. Please try uploading a different image.")
        reset_to_upload()
        return

    # Proceed to get card info if text_prediction exists
    card_info = get_card_info(symbol_prediction, text_prediction)

    # Check if card_info is None (symbol is incorrect or card not found)
    if card_info is None:
        st.write("Card not found. Proceeding with text-only prediction...")
        process_text_only_page(threshold_value)
        return
    
    set_name = card_info.get('Set', 'Unknown Set')
    card_name = card_info.get('Card', 'Unknown Card')
    collector_number = card_info.get('Collector Number', 'Unknown Collector Number')

    st.write(f"Set Name: {set_name}")
    st.write(f"Card Name: {card_name}")
    st.write(f"Collector Number: {collector_number}")

    if st.button("Yes"):
        st.success("Great! You can upload another card.")
        reset_to_upload()

    elif st.button("No"):
        st.write("Proceeding with text prediction only...")
        process_text_only_page(threshold_value)

# Function to handle text-only processing
def process_text_only_page(threshold_value):
    st.title("Step 4: Processing the image with text only...")

    image_rgb = st.session_state.image_rgb 
    symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model, threshold_value)
    result = get_set_and_card_info(text_prediction)

    st.write(f"Text Prediction: {text_prediction}")
    
    # Extract the XXX value from the OCR text (card number)
    card_number = extract_xxx_from_ocr(text_prediction)
    
    # Check if result is None or "possible_ids" is not in the result
    if result is None or "possible_ids" not in result:
        st.write("Text cannot be found")
        reset_to_upload()
        return

    possible_ids = result["possible_ids"]
    
    # If multiple possible ids, iterate through them
    if len(possible_ids) > 1:
        st.write(f"Multiple possible sets found. Total: {len(possible_ids)}")
        st.write("Iterating through the possible matches...")
        for set_info in possible_ids:
            set_id, set_name = set_info  # Ensure only set_id is used for querying
            st.write("---")
            try:
                # Query the table with the set_id to find the card name
                conn = get_read_only_connection()
                cur = conn.cursor()
                table_name = f'public."{set_id}"'  # Only use set_id for table name
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (card_number,))
                card_result = cur.fetchone()
                cur.close()
                conn.close()

                if card_result:
                    card_name = card_result[0]
                    st.markdown(f"Set: **{set_name}**")
                    st.markdown(f"Card Name: **{card_name}**")
                else:
                    # Inform the user that the card number (XXX) cannot be found in the current set's table
                    st.write(f"No card with number **{card_number}** found for Set: **{set_name}** (ID: {set_id}).")

            except psycopg2.errors.UndefinedTable:
                # If the table does not exist, return the "Text cannot be found" message
                st.write(f"Table for Set: {set_name} (ID: {set_id}) cannot be found.")
                reset_to_upload()
                return

        st.info("These are all possible matches. Please upload another card.")
        reset_to_upload()

    # If only one possible id, continue as normal
    else:
        set_id, set_name = possible_ids[0]  # Unpack correctly here too
        try:
            # Query for the card in the single set ID
            conn = get_read_only_connection()
            cur = conn.cursor()
            table_name = f'public."{set_id}"'
            cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (card_number,))
            card_result = cur.fetchone()
            cur.close()
            conn.close()

            if card_result:
                card_name = card_result[0]
                st.write(f"Set Name: **{set_name}**")
                st.write(f"Card Name: **{card_name}**")
            else:
                # Inform the user that the card number cannot be found
                st.write(f"No card with number **{card_number}** found in Set: **{set_name}** (ID: {set_id}).")

        except psycopg2.errors.UndefinedTable:
            # Handle the case where the table doesn't exist
            st.write(f"Table for Set: {set_name} (ID: {set_id}) cannot be found.")
        
        st.info("This is our best guess! Please upload another card.")
        reset_to_upload()


    
# def process_text_only_page(threshold_value):
#     st.title("Step 4: Processing the image with text only...")

#     image_rgb = st.session_state.image_rgb 
#     symbol_prediction, text_prediction = bounding_box_roi(image_rgb, model, threshold_value)
    
#     # Check if the result is None before accessing possible_ids
#     if text_prediction is None:
#         st.write("Text cannot be found")
#         return

#     result = get_set_and_card_info(text_prediction)

#     # If the result is None or if "possible_ids" is not in the result, return a message
#     if result is None or "possible_ids" not in result:
#         st.write("Text cannot be found")
#         return

#     possible_ids = result["possible_ids"]

#     if len(possible_ids) > 1:
#         st.write(f"Multiple possible sets found. Total: {len(possible_ids)}")
#         st.write("Iterating through the possible matches...")
#         for set_id in possible_ids:
#             st.write(f"Checking set with id: {set_id}")
#             conn = get_read_only_connection()
#             cur = conn.cursor()
#             table_name = f'public."{set_id}"'
#             cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (extract_xxx_from_ocr(text_prediction),))
#             card_result = cur.fetchone()
#             cur.close()
#             conn.close()

#             if card_result:
#                 card_name = card_result[0]
#                 st.write(f"Set ID: {set_id}, Card Name: {card_name}")
#             else:
#                 st.write(f"No card found for Set ID: {set_id}")

#         st.info("These are all possible matches. Please upload another card.")
#         reset_to_upload()
#     # If only one possible id, continue as normal
#     else:
#         set_id, set_name = possible_ids[0]
#         card_name = result.get("Card", "Unknown Card")
#         st.write(f"Set Name: {set_name}")
#         st.write(f"Card Name: {card_name}")
#         st.info("This is our best guess! Please upload another card.")
#         reset_to_upload()

# Function to reset the app to upload image state
def reset_to_upload():
    st.session_state.page = 'upload_image'


# # Main function to control the flow
# def main():
#     st.title("Pokémon Card Identifier App")
    
#     # Step 1: Login
#     name = login_page()
    
#     if name:
#         # Step 2: Image Upload
#         image_rgb, threshold_value= upload_image_and_set_threshold()

#         if image_rgb is not None:
#             # Step 3: Process Image (use your real model here)
#             st.button("Process Image", on_click=process_image_with_custom_threshold, args=(image_rgb, threshold_value, model))

#             # Step 4: Confirm Card with the option to process text-only
#             process_symbol_and_text_page(threshold_value)

# # Run the app
# if __name__ == '__main__':
#     main()

# Function for uploading an image and setting the threshold
def upload_image_and_set_threshold():
    st.header("Upload an Image and Set OCR Black Threshold")

    # Image upload
    uploaded_image = st.file_uploader("Upload an image of the card", type=["jpg", "jpeg", "png", "webp"])

    # Add warning or informative text before the slider
    st.warning("Adjust the black threshold to improve OCR results for white cards.")

    # Custom threshold input for white cards (default set to 70)
    threshold_value = st.slider("Set the black threshold for OCR processing on white cards", 
                            min_value=20, max_value=100, value=30,  # Default value set to 30
                            help="Adjust the threshold based on your image.")

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_column_width=True)
        image_rgb = np.array(image)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing
        
        # Save the image to session state
        st.session_state["image_rgb"] = image_rgb
        st.session_state["threshold_value"] = threshold_value

        return image_rgb, threshold_value
    return None, None


# Main function to control the flow
def main():
    st.title("Pokémon Card Identifier App")
    
    # Step 1: Login
    name = login_page()
    
    if name:
        # Step 2: Image Upload
        image_rgb, threshold_value = upload_image_and_set_threshold()

        if image_rgb is not None:
            # Step 3: Process Image (use your real model here)
            if st.button("Process Image"):
                process_symbol_and_text_page(threshold_value)

# Run the app
if __name__ == '__main__':
    main()
