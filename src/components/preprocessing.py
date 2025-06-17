import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Définir l'ordre strict des semestres
    semester_order = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    
    # Forcer le type string et vérifier les valeurs
    X["Semestre"] = X["Semestre"].astype(str)
    invalid_semesters = set(X["Semestre"].unique()) - set(semester_order)
    if invalid_semesters:
        raise ValueError(f"Valeurs de semestre invalides : {invalid_semesters}")

    ordinal_cols = ["Semestre"]
    ordinal_categories = [semester_order]

    # Colonnes numériques (exclure 'Semestre')
    num_cols = [col for col in X.select_dtypes(include=["int64", "float64"]).columns 
               if col != "Semestre"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
            ("num", "passthrough", num_cols)
        ],
        remainder="drop"  # Ignorer les autres colonnes
    )
    return preprocessor


def frequency_encode(df: pd.DataFrame, column: str):
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df[column] = df[column].map(freq_map).fillna(0)
    return df, freq_map

