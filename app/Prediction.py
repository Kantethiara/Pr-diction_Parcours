from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
import io


# === CHEMINS VERS LES MODELES ===
MODEL_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/RandomForest_pipeline.joblib"
ENCODER_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/label_encoder.joblib"
# ENCODFREQ_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/lieu_mapping.joblib"

# === CHARGEMENT DES MODELES ===
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
# lieu_mapping = joblib.load(ENCODFREQ_PATH)

# === UNITE DE REMPLACEMENT ===
unite = {
    "Sciences_fondamentales": [
        "Licence-GBM-111 Sciences fondamentales I",
        "Licence-GBM-121 Sciences fondamentales II",
        "Licence-GBM-112 Sciences chimiques"
    ],
    "Biologie_Biophysique": [
        "Licence-GBM-113 Sciences biologiques I",
        "Licence-GBM-124 Sciences biologiques II",
        "Licence-GBM-231 Sciences biologiques III",
        "Licence-GBM-236 Biophysique I",
        "Licence-GBM-242 Biophysique II",
        "Licence-GBM-356 Techniques d'imagerie"
    ],
    "Informatique": [
        "Licence-GBM-122 Informatique I",
        "Licence-GBM-245 Informatique II",
        "Licence-GBM-234 Automatismes & Informatique industrielle"
    ],
    "Electronique_Automatique": [
        "Licence-GBM-114 Electricité-Electronique I",
        "Licence-GBM-123 Electricité-Electronique II",
        "Licence-GBM-233 Electricité-Electronique III",
        "Licence-GBM-244 Automatique - Système embarqué",
        "Licence-GBM-354 Traitement des signaux"
    ],
    "Maintenance_Systèmes": [
        "Licence-GBM-241 Organisation et méthodes de maintenance I",
        "Licence-GBM-352 Organisation et méthodes de maintenance II",
        "Licence-GBM-353 Maintenance des systèmes",
        "Licence-GBM-355 Maintenance biomédicale"
    ],
    "Biomédical": [
        "Licence-GBM-235 Technologies biomédicales",
        "Licence-GBM-246 Instrumentation biomédicale I",
        "Licence-GBM-351 Instrumentation biomédicale II"
    ],
    "Communication": [
        "Licence-GBM-116 Communication I",
        "Licence-GBM-126 Communication II",
        "Licence-GBM-362 Communication III",
        "Licence-GBM-361 Développement personnel"
    ],
    "Mécanique": [
        "Licence-GBM-115 Mécanique I",
        "Licence-GBM-125 Mécanique II"
    ],
    "Gestion_Risques_HQSE": [
        "Licence-GBM-232 HQSE",
        "Licence-GBM-243 Gestion des risques"
    ],
    "Imagerie": [
        "Licence-GBM-356 Techniques d'imagerie"
    ],
    "Moyenne_generale": [
        "Moyenne générale"
    ]
    
}

# === MAPPING SEMESTRE ===
semester_mapping = {
    'L1GBM': {'S1': 'S1', 'S2': 'S2'},
    'L2GBM': {'S1': 'S3', 'S2': 'S4'},
    'L3GBM': {'S1': 'S5', 'S2': 'S6'},
}


# === FONCTIONS DE TRAITEMENT ===
def replace_col_names(df, unite):
    new_cols = []
    ue_count = {}

    # Étape 1 : Renommer les colonnes selon le mapping
    for col in df.columns:
        found = False
        for ue, col_names in unite.items():
            if col.strip() in col_names:
                ue_count[ue] = ue_count.get(ue, 0) + 1
                new_col = f"{ue}_{ue_count[ue]}" if ue_count[ue] > 1 else ue
                new_cols.append(new_col)
                found = True
                break
        if not found:
            new_cols.append(col.strip())

    df.columns = new_cols

    # Étape 2 : Ajouter les colonnes manquantes avec des zéros
    for ue in unite.keys():
        if ue not in df.columns:
            df[ue] = 0

    return df

def extract_semester_from_filename(filename):
    filename_only = filename.split("/")[-1].replace(".xls", "").replace(".xlsx", "")
    parts = filename_only.split("_")
    if len(parts) >= 3:
        niveau = parts[1]
        semestre_brut = parts[2]
        if niveau in semester_mapping and semestre_brut in semester_mapping[niveau]:
            return semester_mapping[niveau][semestre_brut]
    return "Inconnu"

def process_file(file_bytes, filename, unite):
    df = pd.read_excel(io.BytesIO(file_bytes)).dropna(axis=0, how='all')
    df = replace_col_names(df, unite)
    categories = list(unite.keys())
    colonnes_a_garder = ['N°', 'Prénom(s)', 'Nom'] + categories
    df = df.loc[:, df.columns.intersection(colonnes_a_garder)]
    df["Semestre"] = extract_semester_from_filename(filename)
    df.fillna(0, inplace=True)

    # Ajout des colonnes manquantes avec 0
    colonnes_model = [
        "Informatique", "Electronique_Automatique", "Biologie_Biophysique",
        "Mécanique", "Communication", "Semestre", "Gestion_Risques_HQSE",
        "Biomédical", "Maintenance_Systèmes", "Sciences_fondamentales", 
        "Moyenne_generale", "lieu_naissance", "age"  # Ajout ici
    ]
    for col in colonnes_model:
        if col not in df.columns:
            df[col] = 0 if col in ["age"] else "INCONNU" if col == "lieu_naissance" else 0

    return df


# === INITIALISATION FASTAPI ===
app = FastAPI(
    title="API de Prédiction Parcours",
    description="Prédit la décision semestrielle à partir du profil étudiant",
    version="1.0"
)



# === ENDPOINT PREDICTION ===

@app.post("/predict-file/")
async def predict_from_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename

    try:
        # Traitement du fichier
        df = process_file(content, filename, unite)

        colonnes_model = [
            "Informatique", "Electronique_Automatique", "Biologie_Biophysique",
            "Mécanique", "Communication", "Semestre", "Gestion_Risques_HQSE",
            "Biomédical", "Maintenance_Systèmes", "Sciences_fondamentales", 
            "Moyenne_generale", "lieu_naissance", "age"
        ]

        for col in colonnes_model:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Colonne manquante : {col}")

                # Prédictions
        pred_encoded = model.predict(df[colonnes_model])
        pred_label = label_encoder.inverse_transform(pred_encoded)
        proba = model.predict_proba(df[colonnes_model])[:, 1]

        # Reconstitution du fichier original avec les deux colonnes ajoutées
        df_original = pd.read_excel(io.BytesIO(content)).dropna(axis=0, how='all')
        df_original["prediction"] = pred_label
        df_original["probabilite de valide"] = proba.round(4)

        # Création du fichier Excel à renvoyer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_original.to_excel(writer, index=False)
        output.seek(0)


        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=predictions_{filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier : {str(e)}")
