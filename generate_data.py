"""
Module: generate_data.py

Synthetic patient-record generator. Run standalone to (re)create
synthetic_patient_records.json, the fake clinical dataset that
database.py and embeddings.py ingest, and that the whole demo pipeline
runs against:
    python generate_data.py

[UPGRADED] Fixes a structural gap in the original version: diagnosis was
re-rolled independently on every visit, so the same patient could flip
between unrelated conditions across their own record history, and the
clinical note text + doctor specialty were decoupled from the diagnosis
entirely (e.g. a diabetes visit handled by ONCOSURGERY).

Fix: diagnosis is now assigned ONCE per patient (a chronic condition,
consistent with how real longitudinal records work), and every other
field (note content, specialty, vitals) is derived FROM that diagnosis —
so a single patient's 5 visits read as one coherent history, not 5
independently randomized snapshots.

Identifiers (name, MRD, DOB) remain fully Faker-random — this script does
NOT make those more "realistic." Only the CLINICAL CONTENT is made
internally consistent. See note at bottom on why that distinction matters.
"""

import json
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# ── Diagnosis Profile Table ─────────────────────────────────────────────────
# Maps each diagnosis to the fields that should realistically follow from it.
# This is the key structural change: diagnosis drives doctor_speciality and
# the note content, instead of all three being randomized independently.
DIAGNOSIS_PROFILES = {
    "Chronic Gastritis": {
        "speciality": "GENERAL MEDICINE",
        "vitals_note": "Vital signs stable. Mild epigastric tenderness on palpation.",
        "findings": "Liver and Spleen appear normal in USG. No signs of ulceration on review.",
        "plan": "Continue PPI therapy. Recommended follow-up in 2 weeks.",
        "progression": ["Initial presentation with epigastric discomfort.",
                         "Symptoms improved on PPI therapy, mild residual discomfort.",
                         "Asymptomatic on current regimen, continuing maintenance dose."],
    },
    "Type 2 Diabetes": {
        "speciality": "GENERAL MEDICINE",
        "vitals_note": "Vital signs stable. Blood glucose monitored.",
        "findings": "HbA1c reviewed against prior visit. No acute complications noted.",
        "plan": "Continue current antidiabetic regimen. Recommended follow-up in 4 weeks.",
        "progression": ["Newly diagnosed, counselled on diet and lifestyle modification.",
                         "Blood glucose trending toward target range on current medication.",
                         "Stable glycemic control, no medication adjustment needed this visit."],
    },
    "Hypertension": {
        "speciality": "CARDIOLOGY",
        "vitals_note": "Blood pressure measured in-clinic, vitals otherwise stable.",
        "findings": "No acute cardiac findings. ECG within normal limits.",
        "plan": "Continue antihypertensive medication. Recommended follow-up in 4 weeks.",
        "progression": ["Initial hypertension diagnosis, antihypertensive started.",
                         "Blood pressure trending downward on current dose.",
                         "Blood pressure within target range, maintaining current regimen."],
    },
}


def generate_clinical_description(diagnosis: str, visit_index: int) -> str:
    """
    Builds a clinical note from the diagnosis profile, varying the
    progression line by visit_index so a patient's notes read as an
    evolving history rather than identical text repeated 5 times.

    NOTE: This is a template-based placeholder, same as the original.
    For higher-quality free text, swap this for an Ollama/Phi-3 call —
    that's an orthogonal upgrade to the consistency fix made here.
    """
    profile = DIAGNOSIS_PROFILES[diagnosis]
    progression_line = profile["progression"][min(visit_index, len(profile["progression"]) - 1)]

    return (
        f"Patient presents with symptoms consistent with {diagnosis}. "
        f"{progression_line} "
        f"{profile['vitals_note']} {profile['findings']} {profile['plan']}"
    )


def create_synthetic_record(patient_id, mrd, name, dob, gender, diagnosis, visit_index, visit_date):
    """
    Builds one visit record. diagnosis, doctor_speciality, and the note
    content are no longer independently randomized — they're all derived
    from the SAME diagnosis, passed in by the caller, so they agree with
    each other within a visit AND across a patient's visit history.
    """
    profile = DIAGNOSIS_PROFILES[diagnosis]

    return {
        "patient_id": patient_id,
        "mrd_number": str(mrd),
        "patient_name": name,
        "dob": dob.strftime("%Y-%m-%d 00:00:00"),
        "visit_id": str(fake.random_number(digits=7)),
        "visit_type": "OP",
        "visit_code": f"OP{fake.random_number(digits=4)}",
        "adm_date": None,
        "dschg_date": visit_date.strftime("%Y-%m-%d %H:%M:%S"),
        "document_type": "OP_CON Reports",
        "form_name": "OPD_Progress_Notes",
        "number": visit_index + 1,
        "description": generate_clinical_description(diagnosis, visit_index),
        "gender": gender,
        "doctor_name": f"Dr. {fake.last_name()}",
        "doctor_speciality": profile["speciality"],   # ← derived from diagnosis, not random
        "patient_category": "GNL",
        "diagnosis": diagnosis,    # surfaced explicitly — useful for eval test cases later
    }


def create_patient_history(patient_id: int, num_visits: int = 5) -> list:
    """
    Generates ONE patient's full visit history. Diagnosis is chosen ONCE
    here — at the patient level — and every visit in the returned list
    shares it. Visit dates are spaced out chronologically (not all
    timestamped "now"), so the record reads as a real history rather
    than 5 simultaneous snapshots.
    """
    name = fake.name().upper()
    dob = fake.date_of_birth(minimum_age=30, maximum_age=80)
    gender = random.choice(["Male", "Female"])
    diagnosis = random.choice(list(DIAGNOSIS_PROFILES.keys()))   # chosen ONCE per patient

    records = []
    # Space visits out — e.g. every 3-5 weeks going backward from today,
    # so dschg_date isn't identical across all 5 visits.
    visit_date = datetime.now()
    for i in range(num_visits):
        records.append(
            create_synthetic_record(
                patient_id=patient_id,
                mrd=patient_id,
                name=name,
                dob=dob,
                gender=gender,
                diagnosis=diagnosis,
                visit_index=i,
                visit_date=visit_date,
            )
        )
        visit_date -= timedelta(weeks=random.randint(3, 5))

    # Earliest visit first, matching how a real chart reads top-to-bottom over time.
    records = list(reversed(records))
    for i, record in enumerate(records):
        record["number"] = i + 1   # renumber so visit 1 = earliest, matching the reversed order
    return records


if __name__ == "__main__":
    synthetic_data = create_patient_history(patient_id=20001, num_visits=5)

    with open("synthetic_patient_records.json", "w") as f:
        json.dump(synthetic_data, f, indent=4)

    print(f"Generated {len(synthetic_data)} coherent visit records for 1 synthetic patient.")
    print(f"Diagnosis: {synthetic_data[0]['diagnosis']} (consistent across all visits)")

    # ── Why identifiers stay random but content doesn't ───────────────────────
    # Names/MRD/DOB are intentionally left as plain Faker output — making
    # those MORE realistic doesn't help your eval story, and creeping closer
    # to real-looking PHI patterns is a risk to avoid, not a goal to chase.
    # The improvement that actually matters for a believable clinical RAG
    # demo is INTERNAL CONSISTENCY of clinical content — which is what this
    # rewrite adds. See: diagnosis fixed per-patient, note text and
    # specialty both derived from it, visit dates spaced chronologically.
