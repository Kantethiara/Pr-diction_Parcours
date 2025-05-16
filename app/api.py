from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# === CHARGER LE MODELE ET ENCODEUR ===
MODEL_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/RandomForest_pipeline.joblib"
ENCODER_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/label_encoder.joblib"
ENCODFREQ_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/lieu_mapping.joblib"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
lieu_mapping = joblib.load(ENCODFREQ_PATH)

# === INITIALISATION FASTAPI ===
app = FastAPI(
    title="API de Prédiction Parcours",
    description="Prédit la décision semestrielle à partir du profil étudiant",
    version="1.0"
)

# === DONNÉES D’ENTRÉE (correspondent aux colonnes sans la cible) ===
class StudentData(BaseModel):
    Informatique: float
    Electronique_Automatique: float
    Biologie_Biophysique: float
    Mécanique: float
    Communication: float
    Semestre: str
    Gestion_Risques_HQSE: float
    Biomédical: float
    Maintenance_Systèmes: float
    Sciences_fondamentales: float
    lieu_naissance: str
    Moyenne_generale: float
    age: float

@app.post("/predict")

@app.post("/predict")
def predict(data: StudentData):
    input_dict = data.dict()

    # Vérifier que la ville est connue
    ville = input_dict["lieu_naissance"].upper()
    if ville not in lieu_mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Ville inconnue : '{ville}'. Veuillez utiliser une des villes connues."
        )

    # Encoder la ville
    input_dict["lieu_naissance"] = lieu_mapping[ville]

    # Convertir en DataFrame
    input_df = pd.DataFrame([input_dict])

    # Prédiction
    pred_encoded = model.predict(input_df)
    pred_label = label_encoder.inverse_transform(pred_encoded)

    # Probabilité associée
    proba = model.predict_proba(input_df).max()  # probabilité de la classe prédite

    return {
        "prediction": pred_label[0],
        "probabilite": round(float(proba), 4)  # pour un affichage lisible
    }