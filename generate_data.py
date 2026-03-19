import json
import random
from faker import Faker
from datetime import datetime

fake = Faker()

def generate_clinical_description(diagnosis):
    """
    Optional: You can replace this with an Ollama API call 
    to get high-quality clinical text from Phi-3.
    """
    # Simplified placeholder for the demo
    return f"Patient presents with symptoms of {diagnosis}. Vital signs stable. " \
           f"Liver and Spleen appear normal in USG. Recommended follow-up in 2 weeks."

def create_synthetic_record(patient_id, mrd, name, dob, gender):
    diagnosis = random.choice(["Chronic Gastritis", "Type 2 Diabetes", "Hypertension"])
    
    return {
        "patient_id": patient_id,
        "mrd_number": str(mrd),
        "patient_name": name,
        "dob": dob.strftime("%Y-%m-%d 00:00:00"),
        "visit_id": str(fake.random_number(digits=7)),
        "visit_type": "OP",
        "visit_code": f"OP{fake.random_number(digits=4)}",
        "adm_date": None,
        "dschg_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "document_type": "OP_CON Reports",
        "form_name": "OPD_Progress_Notes",
        "number": 1,
        "description": generate_clinical_description(diagnosis),
        "gender": gender,
        "doctor_name": f"Dr. {fake.last_name()}",
        "doctor_speciality": random.choice(["ONCOSURGERY", "GENERAL MEDICINE", "CARDIOLOGY"]),
        "patient_category": "GNL"
    }

# Generate 5 records for 1 synthetic patient
synthetic_data = []
p_id = 20001
p_name = fake.name().upper()
p_dob = fake.date_of_birth(minimum_age=30, maximum_age=80)
p_gender = random.choice(["Male", "Female"])

for _ in range(5):
    synthetic_data.append(create_synthetic_record(p_id, p_id, p_name, p_dob, p_gender))

with open("synthetic_patient_records.json", "w") as f:
    json.dump(synthetic_data, f, indent=4)

print("Synthetic data generated successfully!")