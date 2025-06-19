from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class SemestreTransformer(BaseEstimator, TransformerMixin):
    """Transformateur personnalisé pour les semestres avec noms de features"""
    def __init__(self):
        self.ordre = {'S1':1, 'S2':2, 'S3':3, 'S4':4, 'S5':5, 'S6':6}
        self.feature_name_ = 'Semestre'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X['Semestre'].map(self.ordre).fillna(0).to_numpy().reshape(-1, 1)
    
    def get_feature_names_out(self, input_features=None):
        return np.array([self.feature_name_])

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Préprocesseur avec gestion parfaite des noms de features"""
    # 1. Colonnes à exclure
    excluded = ["numero etudiant", "Moyenne_generale", "decision_semestrielle"]
    features = [col for col in X.columns if col not in excluded]
    
    # 2. Détection automatique des types
    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(X[col]) and col != 'Semestre']
    categorical_features = [col for col in features if col not in numeric_features and col != 'Semestre']
    
    # 3. Configuration des transformers
    transformers = [
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('sem', SemestreTransformer(), ['Semestre'])
    ]
    
    # 4. Création du préprocesseur avec noms de features propres
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False  # Désactive les préfixes automatiques
    )
    
    return preprocessor

def get_feature_names(preprocessor, categorical_features) -> list:
    """Obtient les noms propres des features après transformation"""
    feature_names = []
    
    for name, transformer, original_features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(original_features)
        elif name == 'sem':
            feature_names.extend(transformer.get_feature_names_out())
        elif name == 'cat':
            # Pour OneHotEncoder
            for i, col in enumerate(original_features):
                categories = transformer.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    return feature_names

def frequency_encode(df: pd.DataFrame, column: str):
    """Encodage fréquentiel robuste avec vérifications"""
    if column not in df.columns:
        raise ValueError(f"Colonne {column} introuvable")
    
    freq_map = df[column].value_counts(normalize=True)
    if freq_map.isna().any():
        raise ValueError("Valeurs manquantes détectées dans la colonne")
    
    df[column] = df[column].map(freq_map).fillna(0)
    return df, freq_map.to_dict()