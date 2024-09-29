import psycopg2
import os
from psycopg2 import OperationalError
from ocr_utils import extract_xxx_from_ocr, extract_yyy_from_ocr, handle_aaaxxx_format, handle_qqq_format
import re

# --- LOAD POSTGRESQL CONNECTION ---
# Function to create a PostgreSQL connection as the readonly user
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
            # Query the pokemon_sets table to find the set ID and name based on the symbol_prediction
            cur.execute("SELECT id, name FROM public.pokemon_sets WHERE LOWER(name) = %s", (symbol_prediction.lower(),))
            set_result = cur.fetchone()

            if not set_result:
                return None, "Set cannot be found"

            set_id, set_name = set_result

            # Handle empty or invalid text_prediction
            if not text_prediction or text_prediction.strip() == "":
                return None, "Text prediction is missing or invalid"

            # Handle the text_prediction format
            if "/" in text_prediction:
                card_number = extract_xxx_from_ocr(text_prediction)

            elif re.match(r'^[A-Za-z]{2,4}\d{1,3}$', text_prediction):
                card_number = extract_xxx_from_ocr(text_prediction)

            elif text_prediction.isdigit():
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

                return possible_cards

            else:
                return None, "Invalid text format"

            # Query the table with the set_id to find the card_name
            try:
                table_name = f'public."{set_id}"'
                cur.execute(f"SELECT card_name FROM {table_name} WHERE card_number = %s", (card_number,))
                card_result = cur.fetchone()
            except psycopg2.errors.UndefinedTable:
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
    possible_ids = []
    try:
        # Handle XXX/YYY format
        if '/' in ocr_result:
            yyy_value = extract_yyy_from_ocr(ocr_result)
            xxx_value = extract_xxx_from_ocr(ocr_result)
            
            if not yyy_value or not xxx_value:
                print("Invalid OCR result format for XXX/YYY.")
                return "Card cannot be found."

            # Query the pokemon_sets table to find all possible ids and names based on printed_total
            cur.execute("SELECT id, name FROM public.pokemon_sets WHERE printed_total = %s", (yyy_value,))
            set_results = cur.fetchall()
            
            if not set_results:
                print(f"No set found with printed_total = {yyy_value}")
                return "Card cannot be found."
            
            # Store all possible matching ids and names
            for set_result in set_results:
                set_id, set_name = set_result
                possible_ids.append((set_id, set_name))
            
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

            possible_cards = []

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
                result = handle_qqq_format(ocr_result, conn, cur)
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