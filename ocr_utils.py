import cv2
import easyocr
import re
import numpy as np
import streamlit as st
import psycopg2


# Function to display image for debugging
def debug_show_image(image, title="Debug Image"):
    # Convert BGR to RGB if necessary
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    st.image(image_rgb, caption=title, use_column_width=True)

# --- LOAD OCR ---
reader = easyocr.Reader(['en'])

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
    pattern_letters_digits = r'[a-zA-Z]{2,4}\d{1,3}'
    pattern_numbers_slash = r'\d{1,3}/\d{2,3}'
    pattern_numbers_only = r'^\d{1,3}$'

    for result in ocr_results:
        # Try to match all patterns
        match_letters_digits = re.search(pattern_letters_digits, result)
        match_numbers_slash = re.search(pattern_numbers_slash, result)
        match_numbers_only = re.search(pattern_numbers_only, result)

        # Return the matched portion if found
        if match_letters_digits:
            return match_letters_digits.group()
        elif match_numbers_slash:
            return match_numbers_slash.group()
        elif match_numbers_only:
            return match_numbers_only.group()

    return None

# --- TEXT EXTRACTION FROM OCR ---
# Extract YYY from OCR result
def extract_yyy_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[1]
    return None

# Function to extract XXX from OCR result
def extract_xxx_from_ocr(ocr_result):
    if '/' in ocr_result:
        return ocr_result.split("/")[0]
    return ocr_result

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
    # We expect the format to be purely numeric
    if not ocr_result.isdigit():
        print("Invalid QQQ format")
        return None

    # Define the possible sets ('np' and 'svp')
    possible_sets = ["np", "svp"]
    possible_cards = []

    # Loop through both possible sets and query for card matches
    for set_id in possible_sets:
        # First, query pokemon_sets to get the name for the set id
        cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(id) = %s", (set_id,))
        set_result = cur.fetchone()

        if not set_result:
            print(f"No set found for id = {set_id}")
            continue

        set_id, set_name = set_result

        # Now query the table with the set ID to find the matching card by number
        table_name = f'public."{set_id}"'
        cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (ocr_result,))
        card_result = cur.fetchone()

        if card_result:
            card_name = card_result[0]
            possible_cards.append({"Set_Name": set_name, "Card": card_name, "Set_ID": set_id, "OCR_Result": ocr_result})

    if possible_cards:
        return possible_cards
    else:
        return "Card cannot be found."