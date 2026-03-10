from fastapi import FastAPI
import requests
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
multi_model = joblib.load("multi_output_model.pkl")

READ_API_KEY = "VP4QU22XYV802EIN"
WRITE_API_KEY = "YOUR_WRITE_API_KEY"
CHANNEL_ID = "3291923"

@app.get("/predict")
def predict_from_thingspeak():
    # Fetch latest data
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
    response = requests.get(url).json()
    latest_entry = response["feeds"][-1]

    temp = float(latest_entry["field1"])
    force = float(latest_entry["field2"])
    spo2 = float(latest_entry["field4"]) if latest_entry["field4"] not in ["", "-999"] else 0.0
    patient_id = latest_entry.get("field5", "Unknown")
    # Predict
    X_new = pd.DataFrame([[temp, spo2, force]], columns=["Temp","SPO2","Pressure"])
    prediction = multi_model.predict(X_new)
    Temp_Status, SPO2_Status, Compression_Status = prediction[0]

    # Send predictions back to ThingSpeak
    update_url = f"https://api.thingspeak.com/update?api_key={WRITE_API_KEY}"
    payload = {
        "field6": Temp_Status,
        "field7": SPO2_Status,
        "field8": Compression_Status
    }
    requests.post(update_url, params=payload)

    return {
        "Patient_ID": patient_id,
        "Temp": temp,
        "SPO2": spo2,
        "Force": force,
        "Temp_Status": Temp_Status,
        "SPO2_Status": SPO2_Status,
        "Compression_Status": Compression_Status
    }