import cv2
import numpy as np
from src.ocr_utils import ocr_preprocessing, perform_ocr_easyocr, process_ocr_results
from src.model_utils import load_class_names, load_symbol_model
import streamlit as st

model = load_symbol_model()
class_names = load_class_names()

# --- CLASSIFY CARDS AS WHITE OR NON-WHITE ---
# Function to classify cards based on brightness due to processing
def classify_card(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    return "white" if mean_brightness > 150 else "non_white"

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
        return None, "Contour cannot be found.", None

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
        return None, "Text cannot be found.", None
    
    # --- Name ROI ---
    name_roi_height = text_roi_height
    name_roi_coords = (x, y, (2 * w) // 3, name_roi_height * 2)
    name_roi = image_bgr[name_roi_coords[1]:name_roi_coords[1] + name_roi_coords[3],
                         name_roi_coords[0]:name_roi_coords[0] + name_roi_coords[2]]
    
    # Processing Name ROI for OCR
    processed_name_roi = ocr_preprocessing(name_roi)
    name_ocr_result = perform_ocr_easyocr(processed_name_roi)

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
        return None, "Contour cannot be found.", None

    return symbol_prediction, final_ocr_result, name_ocr_result


# Function to display image for debugging
def debug_show_image(image, title="Debug Image"):
    # Convert BGR to RGB if necessary
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

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

        # --- Name ROI ---
        name_roi_height = text_roi_height
        name_roi_coords = (x, y, (2 * w) // 3, name_roi_height * 2)
        name_roi = image_rgb[name_roi_coords[1]:name_roi_coords[1] + name_roi_coords[3],
                            name_roi_coords[0]:name_roi_coords[0] + name_roi_coords[2]]
        
        # Processing Second Text ROI for OCR
        processed_name_roi = ocr_preprocessing(name_roi)
        name_ocr_result = perform_ocr_easyocr(processed_name_roi)

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
            return None, "Contour cannot be found.", None

        return symbol_prediction, final_ocr_result, name_ocr_result

    else:
        print("No contours found.")
        return image_rgb, None, None



# --- MASTER FUNCTION TO HANDLE BOTH WHITE AND NON-WHITE ---
# Main function to handle both white and non-white cards
def bounding_box_roi(image_rgb, model):
    card_type = classify_card(image_rgb)

    if card_type == "white":
        # Handle white cards without threshold_value
        image_with_bounding_box, ocr_result, name_result = draw_bounding_box_white(image_rgb, model)
    else:
        # Handle non-white cards with a fixed threshold_value of 160
        threshold_value = 160
        image_with_bounding_box, ocr_result, name_result = draw_bounding_boxes(image_rgb, threshold_value, model)

    return image_with_bounding_box, ocr_result, name_result
