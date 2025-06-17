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
from .preprocessing import build_preprocessor, frequency_encode

SAVE_PATH = "/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/"




def train_and_compare_models(df: pd.DataFrame, target_col: str = "decision_semestrielle"):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # # Calculer frequency encoding + mapping pour lieu_naissance
    # freq_map = X["lieu_naissance"].value_counts(normalize=True).to_dict()
    # X["lieu_naissance"] = X["lieu_naissance"].map(freq_map).fillna(0)
    # lieu_mapping = freq_map  # On garde la map pour la réutiliser plus tard

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    preprocessor = build_preprocessor(X_train)

    base_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    best_score = 0
    best_pipeline = None
    best_model_name = ""

    print("\n🔍 Évaluation des modèles avant optimisation :")

    for name, model in base_models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name}: accuracy = {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_pipeline = pipe
            best_model_name = name

    print(f"\n🏆 Meilleur modèle avant optimisation: {best_model_name} avec accuracy = {best_score:.4f}")

    # Définition de la grille d'hyperparamètres pour optimisation
    if best_model_name == "RandomForest":
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__bootstrap": [True, False]
        }
    elif best_model_name == "DecisionTree":
        param_grid = {
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        }
    elif best_model_name == "LogisticRegression":
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"]
        }

    opt_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", base_models[best_model_name])
    ])

    search = RandomizedSearchCV(
        opt_pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="accuracy",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, y_pred)

    print(f"\n🔧 Meilleur modèle après optimisation ({best_model_name}) : accuracy = {best_acc:.4f}")
    print("🔧 Meilleurs hyperparamètres:", search.best_params_)

    # On retourne bien 5 objets
    return best_model, base_models[best_model_name], label_encoder, best_model_name


def save_artifacts(pipeline: Pipeline, label_encoder: LabelEncoder, name: str):
    os.makedirs(SAVE_PATH, exist_ok=True)

    # # Sauvegarde du mapping de lieu
    # mapping_path = os.path.join(SAVE_PATH, "lieu_mapping.joblib")
    # joblib.dump(lieu_mapping, mapping_path)

    # Sauvegarde du pipeline et de l'encodeur
    model_path = os.path.join(SAVE_PATH, f"{name}_pipeline.joblib")
    encoder_path = os.path.join(SAVE_PATH, "label_encoder.joblib")

    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)
    
        # Extraire le modèle seul (classifier) pour SHAP
    model_only = pipeline.named_steps["classifier"]
    model_only_path = os.path.join(SAVE_PATH, f"{name}_model.joblib")
    joblib.dump(model_only, model_only_path)

    print(f"💾 Modèle seul sauvegardé pour SHAP : {model_only_path}")


    print(f"💾 Pipeline sauvegardé : {model_path}")
    print(f"💾 Encodage cible sauvegardé : {encoder_path}")
    # print(f"💾 Mapping des lieux sauvegardé : {mapping_path}")


def show_feature_importance(pipeline: Pipeline, X: pd.DataFrame):
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessing"]

    if hasattr(model, "feature_importances_"):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = X.columns  # fallback si pas de noms
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        print("\n📊 Feature importances :")
        for name, imp in feat_imp:
            print(f"{name}: {imp:.4f}")
    else:
        print("\nℹ️ Ce modèle ne fournit pas de feature importance.")
