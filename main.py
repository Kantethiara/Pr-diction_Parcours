import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from src.components.modelling import (
    train_and_compare_models, 
    save_artifacts, 
    show_feature_importance
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Chargement des données
        data_path = Path(__file__).resolve().parent / "src" / "data" / "shift_Data1.xlsx"        
        logger.info(f"Chargement des données depuis {data_path}")

        df = pd.read_excel(data_path)
        df = df.rename(columns={"Moyenne generale": "Moyenne_generale"})
        
        # 2. Entraînement
        logger.info("Lancement de l'entraînement...")
        start = datetime.now()
        
        pipeline, model, label_encoder, model_name = train_and_compare_models(df)
        
        logger.info(f"Temps d'entraînement: {(datetime.now()-start).total_seconds():.2f}s")
        logger.info(f"Modèle sélectionné: {model_name}")

        # 3. Sauvegarde
        save_artifacts(pipeline, label_encoder, model_name)
        
        # 4. Analyse
        X = df.drop(columns=["decision_semestrielle"])
        show_feature_importance(pipeline, X)
        
        logger.info("Processus terminé avec succès")
        
    except Exception as e:
        logger.error("Échec du processus", exc_info=True)
        raise

if __name__ == "__main__":
    main()