import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    ordinal_cols = ["Semestre"]
    ordinal_categories = [["S1", "S2", "S3", "S4", "S5"]]

    # Colonnes numériques (inclura lieu_naissance déjà encodé)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    return preprocessor


def frequency_encode(df: pd.DataFrame, column: str):
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df[column] = df[column].map(freq_map).fillna(0)
    return df, freq_map

