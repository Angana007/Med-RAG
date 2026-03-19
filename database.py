import sqlite3
import json
import os

"""
Module: database.py
This module sets up a permanent SQLite database to store organized patient information. 
It allows the system to quickly verify a patient and pull their entire medical history 
before the AI starts answering questions.
"""
DB_NAME = "clinical_data.db"

#Initialize the SQLite database and create the patient metadata table
def init_db():
    """
    Initializes the SQLite database and defines the clinical metadata schema.
    Includes an index on mrd_number to optimize query performance during 
    real-time API requests.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
    """CREATE TABLE IF NOT EXISTS patient_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,   -- A hidden unique ID for every single row
        mrd_number TEXT,                        -- Now allowed to repeat for multiple visits
        patient_name TEXT,
        gender TEXT,
        doctor_name TEXT,
        dschg_date TEXT,
        document_type TEXT,
        visit_id TEXT,
        UNIQUE(mrd_number, visit_id)            -- Prevents the EXACT same visit being added twice
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mrd ON patient_records(mrd_number)")
    conn.commit()
    conn.close()

#Function to parse local JSON files and insert records safely.   
def populate_db():
    """
    Parses local JSON clinical records and synchronizes them with the database.
    Uses 'INSERT OR IGNORE' to prevent duplicate data, making it safe to run the script multiple times as you add new files.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    for file_name in json_files:
        with open(file_name, 'r') as f:
            try:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
                
                for record in records:
                    cursor.execute(
                        '''INSERT OR IGNORE INTO patient_records
                        (mrd_number, patient_name, gender, doctor_name, dschg_date, document_type, visit_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)                        
                        ''', (
                        str(record.get('mrd_number')),
                        record.get('patient_name'),
                        record.get('gender'),
                        record.get('doctor_name'),
                        record.get('dschg_date'),      # Mapped for "past-dated events"
                        record.get('document_type'),   # Added for context filtering
                        record.get('visit_id')         # Unique identifier per record
                        ))
            except json.JSONDecodeError:
                print(f"Error skipping invalid JSON: {file_name}")
                
    conn.commit()
    conn.close()
    print(f"Successfully populated {DB_NAME} from local JSON files.")

#Function to retrieve all clinical metadata associated with a specific MRD number.
def get_patient_metadata(mrd_number):
    """
    Retrieves the full clinical history for a specific patient.

    Args:
        mrd_number: The unique Medical Record Number to query.

    Returns:
        A list of dictionaries containing all historical visit metadata 
        to be used as grounding context for the LLM.
    """
    
    conn =sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row                #Enables accessing columns by patient name
    cursor = conn.cursor()
    
    try:
        query = "SELECT * FROM patient_records WHERE mrd_number = ?"
        cursor.execute(query, (str(mrd_number),))
        #Ensures the chatbot sees the entire medical history, not just the first file the database found
        rows = cursor.fetchall()
        
        # Convert sqlite3.Row objects into a list of standard Python dictionaries
        results = [dict(row) for row in rows]
        return results
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()
    