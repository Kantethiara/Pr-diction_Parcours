import pandas as pd

from src.components.modelling import (
    train_and_compare_models, save_artifacts, show_feature_importance
)

df = pd.read_excel("/Users/thiarakante/Documents/Databeez/prediction_parcours/src/data/shift_data.xlsx")
df = df.rename(columns={"Moyenne generale": "Moyenne_generale"})

pipeline, model, label_encoder, model_name, lieu_mapping = train_and_compare_models(df)

save_artifacts(pipeline, label_encoder, lieu_mapping, name=model_name)

# (facultatif) Affichage des features importantes
X = df.drop(columns=["decision_semestrielle"])

show_feature_importance(pipeline, X)

