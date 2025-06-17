import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
import io
from src.components.preprocessing import build_preprocessor  # adapte le chemin si besoin
from fastapi.responses import StreamingResponse


# === CHEMINS VERS LES MODELES ===
MODEL_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/DecisionTree_pipeline.joblib"
ENCODER_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/label_encoder.joblib"
Model_seul = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/DecisionTree_model.joblib"

# === CHARGEMENT DES MODELES ===
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
model_seul = joblib.load(Model_seul)
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




@app.post("/shap-explanation/")
async def shap_explanation(
    file: UploadFile = File(...), 
    nom: str = None,
    prenom: str = None,
    numero_etudiant: str = None

):
    content = await file.read()
    filename = file.filename

    try:
        # 1. Traitement du fichier avec TOUTES les variables du modèle
        df = process_file(content, filename, unite)
        df.columns = df.columns.str.strip()  # Nettoyage des noms de colonnes
        # print("Colonnes du DataFrame après traitement :", df.columns.tolist())

        # 2. Recherche de l'étudiant
        
        
        if numero_etudiant:
            # Recherche par numéro étudiant si fourni
            if numero_etudiant not in df["N°"].values:
                raise HTTPException(status_code=404, detail="Numéro étudiant non trouvé.")
            ligne = df[df["N°"] == numero_etudiant].copy()
        elif nom or prenom:
            # Recherche par nom et/ou prénom (en utilisant les noms de colonnes après process_file)
            query = []
            if nom:
                # Recherche insensible à la casse et partielle pour le nom
                query.append(df["Nom"].str.lower().str.contains(nom.lower(), na=False))
            if prenom:
                # Recherche insensible à la casse et partielle pour le prénom
                query.append(df["Prénom(s)"].str.lower().str.contains(prenom.lower(), na=False))
            
            # Combiner les conditions
            if query:
                mask = query[0]
                for q in query[1:]:
                    mask = mask & q
                
                matching_students = df[mask]
                
                if len(matching_students) == 0:
                    raise HTTPException(status_code=404, detail="Aucun étudiant trouvé avec ces critères.")
                elif len(matching_students) > 1:
                    # Retourner la liste des étudiants correspondants pour affinage
                    students_list = matching_students[["N°", "Nom", "Prénom(s)"]].to_dict(orient="records")
                    return {
                        "message": "Plusieurs étudiants correspondent à la recherche",
                        "etudiants": students_list,
                        "count": len(matching_students)
                    }
                else:
                    ligne = matching_students.copy()
        else:
            raise HTTPException(status_code=400, detail="Vous devez fournir soit un numéro étudiant, soit un nom/prénom")

        colonnes_model = [
            "Informatique", "Electronique_Automatique", "Biologie_Biophysique",
            "Mécanique", "Communication", "Semestre", "Gestion_Risques_HQSE",
            "Biomédical", "Maintenance_Systèmes", "Sciences_fondamentales", 
            "Moyenne_generale", "lieu_naissance", "age"
        ]

        # 2. Vérification des colonnes manquantes
        missing_cols = [col for col in colonnes_model if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Colonnes manquantes : {missing_cols}")

        # 3. Préparation des données
        ligne_X = ligne[colonnes_model]

        # 4. Préprocessing avec vérification
        preprocessor = build_preprocessor(df[colonnes_model])
        preprocessor.fit(df[colonnes_model])
        ligne_transformed = preprocessor.transform(ligne_X)

        # Vérification cruciale des features
        feature_names_out = preprocessor.get_feature_names_out()
        print("Features après prétraitement:", feature_names_out)  # Debug

        # 5. Alignement avec le modèle
        if hasattr(model_seul, 'feature_names_in_'):
            expected_features = model_seul.feature_names_in_
        else:
            expected_features = feature_names_out[:model_seul.n_features_in_]

        # Création d'un masque d'alignement
        feature_mask = [f in expected_features for f in feature_names_out]
        ligne_transformed_aligned = ligne_transformed[:, feature_mask]

        # 6. Calcul SHAP avec vérification
        try:
            explainer = shap.TreeExplainer(model_seul)
            shap_values = explainer(ligne_transformed_aligned)
            shap_values_pos = shap_values.values[0, :, 1]
            
            # 7. Préparation des résultats
            df_shap = pd.DataFrame({
                'feature': expected_features,  # Utiliser les noms attendus par le modèle
                'shap_value': shap_values_pos
            })
            
            # Filtrage (adapté aux features alignées)
            df_shap = df_shap[~df_shap['feature'].str.contains('Semestre|lieu_naissance|Moyenne_generale', regex=True)]
            df_shap = df_shap[df_shap['shap_value'] != 0].copy()    
            df_neg = df_shap[df_shap['shap_value'] < 0].copy()
            df_neg['abs_impact'] = df_neg['shap_value'].abs()
            df_neg = df_neg.sort_values('abs_impact', ascending=False)
            
            resultats = []
            for i, (_, row) in enumerate(df_neg.iterrows(), 1):
                clean_feature = row['feature'].split('__')[-1] if '__' in row['feature'] else row['feature']
                
                resultats.append({
                    "rang": i,
                    "thematique": clean_feature,
                    "impact_negatif": float(row['shap_value']),
                    "magnitude_impact": float(row['abs_impact']),
                    "interpretation": f"La thématique {clean_feature} réduit la probabilité de validation."
                })
            prob_validation = float(model_seul.predict_proba(ligne_transformed_aligned)[0][1])  # Classe 1 = réussite
            prob_non_validation = 1 - prob_validation  # Classe 0 = échec

            if not resultats:
                resultats.append({
                    "message": "Aucune thématique n'a d'impact négatif significatif sur la décision pour cet étudiant."
                })
                                # Calcul de la probabilité
              
            return {
                "etudiant": ligne["N°"].values[0],
                "nom": ligne["Nom"].values[0],
                "prenom": ligne["Prénom(s)"].values[0],
                "Probabilité de validation": float(model_seul.predict_proba(ligne_transformed_aligned)[0][1]),  # Correction ici
                # "Décision": "Validé" if prob_validation >= 0.5 else "Non validé" , # Interprétation claire,
                "impacts_negatifs": resultats
            }

        except HTTPException:
            raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur SHAP: {str(e)}"
        )
        
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
          # Prédiction
        X = df[colonnes_model]
        X.loc[:, "Semestre"] = X.loc[:, "Semestre"].astype(str)
  # S'assurer que 'Semestre' est bien traité
        
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
