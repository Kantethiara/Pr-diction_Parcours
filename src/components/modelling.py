import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from .preprocessing import build_preprocessor, get_feature_names
from sklearn.model_selection import cross_val_score, StratifiedKFold

from pathlib import Path

# Définir un chemin relatif basé sur le répertoire actuel
SAVE_PATH = Path(__file__).resolve().parent / "artifacts"
def train_and_compare_models(df: pd.DataFrame, target_col: str = "decision_semestrielle"):
    # 1. Préparation des données
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Encodage de la target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 2. Split des données (identique à Colab)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # 3. Construction du préprocesseur
    preprocessor = build_preprocessor(X_train)
    
    # 4. Liste des modèles à tester
    base_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=3,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced_subsample'
        )
    }
    
    # 5. Évaluation des modèles
    best_score = 0
    best_model = None
    best_name = ""
    
    print("\n🔍 Évaluation des modèles :")
    for name, model in base_models.items():
        try:
            # Création du pipeline
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Entraînement et évaluation
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"✅ {name}: Accuracy = {acc:.4f}")
            
            if acc > best_score:
                best_score = acc
                best_model = pipe
                best_name = name
        except Exception as e:
            print(f"❌ Erreur avec {name}: {str(e)}")
            continue
    
    print(f"\n🏆 Meilleur modèle: {best_name} (Accuracy: {best_score:.4f})")
    return best_model, base_models[best_name], label_encoder, best_name

def save_artifacts(pipeline: Pipeline, label_encoder: LabelEncoder, name: str):
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Sauvegarde du pipeline complet
    pipeline_path = os.path.join(SAVE_PATH, f"{name}_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    
    # Sauvegarde du préprocesseur séparément (utile pour l'exploration)
    preprocessor = pipeline.named_steps['preprocessor']  # <-- Changement clé
    preprocessor_path = os.path.join(SAVE_PATH, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Sauvegarde du label encoder
    encoder_path = os.path.join(SAVE_PATH, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    
    # Sauvegarde du modèle seul (pour SHAP ou autres analyses)
    model_only = pipeline.named_steps["classifier"]
    model_path = os.path.join(SAVE_PATH, f"{name}_model.joblib")
    joblib.dump(model_only, model_path)

    print(f"\n💾 Artefacts sauvegardés dans {SAVE_PATH}:")
    print(f"- Pipeline complet: {pipeline_path}")
    print(f"- Préprocesseur: {preprocessor_path}")
    print(f"- Encodage cible: {encoder_path}")
    print(f"- Modèle seul: {model_path}")

def show_feature_importance(pipeline: Pipeline, original_features: list):
    """Affiche les importances avec les vrais noms de colonnes"""
    try:
        # 1. Récupération des composants
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # 2. Obtention des noms de features
        cat_features = [col for col in original_features 
                       if not pd.api.types.is_numeric_dtype(original_features[col])]
        feature_names = get_feature_names(preprocessor, cat_features)
        
        # 3. Vérification de cohérence
        if len(feature_names) != len(model.feature_importances_):
            raise ValueError("Incohérence entre les features et les importances")
        
        # 4. Affichage propre
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n📊 Feature Importances:")
        print(importance_df.to_string(index=False))
        
        return importance_df
        
    except Exception as e:
        print(f"\n⚠️ Erreur d'affichage: {str(e)}")
        return None
        
 
    
def load_artifacts(model_name: str):
    pipeline_path = os.path.join(SAVE_PATH, f"{model_name}_pipeline.joblib")
    encoder_path = os.path.join(SAVE_PATH, "label_encoder.joblib")
    
    if not os.path.exists(pipeline_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Artefacts non trouvés. Avez-vous bien entraîné et sauvegardé le modèle?")
    
    pipeline = joblib.load(pipeline_path)
    encoder = joblib.load(encoder_path)
    
    return pipeline, encoder