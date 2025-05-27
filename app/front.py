import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import requests
import base64
import io
import dash_ag_grid as dag
import matplotlib.pyplot as plt
import shap

# Initialisation de l'application Dash
app = dash.Dash(__name__, external_stylesheets=[
    "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
])
server = app.server


# URL de l'API
API_URL = "http://127.0.0.1:8000/predict-file"

def generate_shap_waterfall_image(pipeline, X_test, index=0):
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessing"]

    # Transformer les données
    X_transformed = preprocessor.transform(X_test)

    # Créer l'explainer
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    # Construire l'explication locale
    shap_local = shap.Explanation(
        values=shap_values.values[index, :, 1],
        base_values=shap_values.base_values[index, 1],
        data=shap_values.data[index],
        feature_names=shap_values.feature_names
    )

    # Générer le plot
    plt.figure()
    shap.plots.waterfall(shap_local, show=False)

    # Sauvegarder l’image en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return f"data:image/png;base64,{encoded}"

# Définition du layout de l'application
app.layout = html.Div([
    html.Div([
        html.H1("🎓 Interface de Prédiction du Parcours Étudiant", className="text-center text-primary mb-4"),
        html.H2("📁 Prédiction par fichier CSV", className="text-secondary mt-4"),

        dcc.Upload(
            id="upload-data",
            children=html.Div([
                "Glissez et déposez un fichier CSV ici ou ",
                html.A("cliquez pour sélectionner", className="text-primary")
            ]),
            style={
                "width": "100%", "height": "60px", "lineHeight": "60px",
                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                "textAlign": "center", "margin": "10px"
            },
            multiple=False
        ),

        html.Div(id="file-predictions"),
        html.Div(id="download-link", className="mt-3")
    ], className="container mt-4")
])


@app.callback(
    [Output("file-predictions", "children"),
     Output("download-link", "children")],
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def predict_from_file(contents, filename):
    if contents is None:
        return "", ""

    try:
        # Décodage du contenu base64 du fichier uploadé
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Envoi du fichier décodé à l'API via POST
        files = {'file': (filename, io.BytesIO(decoded), 'application/octet-stream')}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            try:
                # Lecture du fichier Excel retourné par l'API en DataFrame
                result_df = pd.read_excel(io.BytesIO(response.content))
                result_df["prediction"] = result_df["prediction"].astype(str).str.strip().str.capitalize()

                print("Valeurs uniques dans 'prediction' :", result_df["prediction"].unique())

            except Exception as e:
                return html.Div(f"⚠️ Erreur lors de la lecture du fichier Excel retourné : {str(e)}", className="text-danger"), ""

            # Préparation du CSV pour le téléchargement
            download_csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(download_csv.encode()).decode()

            # Affichage dans Dash AG Grid
            table = dag.AgGrid(
                id="ag-grid",
                rowData=result_df.to_dict("records"),
                columnDefs=[
                    {"headerName": col, "field": col, "filter": True, "sortable": True}
                    for col in result_df.columns
                ],
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                    "minWidth": 100,
                    "flex": 1,
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 10
                },
                className="ag-theme-alpine",
                style={"height": "500px", "width": "100%"},
            )

            # Lien de téléchargement
            download_link = html.A(
                "📥 Télécharger le fichier avec les prédictions",
                id="download-link-anchor",  # ID unique ici
                href=f"data:text/csv;base64,{b64}",
                download=f"predictions_{filename}.csv",
                className="btn btn-success"
            )

            return table, download_link

        else:
            return html.Div(f"⚠️ Erreur lors de la prédiction : {response.text}", className="text-danger"), ""

    except Exception as e:
        return html.Div(f"⚠️ Erreur lors du traitement du fichier : {str(e)}", className="text-danger"), ""


if __name__ == '__main__':
    app.run(debug=True)
