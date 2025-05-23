import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import requests
import json
import base64
import io

app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])
server = app.server  # Pour le déploiement

API_URL = "http://127.0.0.1:8000/predict"  # URL de l'API FastAPI

features = [
    "Informatique", "Electronique_Automatique", "Biologie_Biophysique", "Mécanique", "Communication",
    "Semestre", "Gestion_Risques_HQSE", "Biomédical", "Maintenance_Systèmes", "Sciences_fondamentales",
    "lieu_naissance", "Moyenne_generale", "age"
]

app.layout = html.Div([
    html.Div([
        html.H1("🎓 Interface de Prédiction du Parcours Étudiant", className="text-center text-primary mb-4"),

        html.H2("📋 Prédiction individuelle", className="text-secondary"),
        html.Div([
            html.Div([
                html.Label(f, className="form-label"),
                dcc.Input(id=f"input-{f}", type="text", className="form-control mb-2")
            ]) for f in features
        ], className="mb-3"),
        html.Button("Prédire", id="predict-button", className="btn btn-primary mb-3"),
        html.Div(id="prediction-output", className="alert alert-info"),

        html.Hr(),

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
        html.Div(id="file-predictions")
    ], className="container mt-4")
])

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    [State(f"input-{f}", "value") for f in features]
)
def predict_single(n_clicks, *values):
    if n_clicks is None:
        return ""

    data = dict(zip(features, values))
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            return html.Div([
                html.P(f"✅ Prédiction : {result['prediction']}", className="fw-bold"),
                html.P(f"🎯 Probabilité : {result['probabilite'] * 100:.2f}%")
            ])
        else:
            return html.Div(["Erreur : ", response.text], className="text-danger")
    except Exception as e:
        return html.Div(["Exception : ", str(e)], className="text-danger")

@app.callback(
    Output("file-predictions", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def predict_from_file(contents, filename):
    if contents is None:
        return ""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    if not set(features).issubset(df.columns):
        return html.Div("Colonnes manquantes dans le fichier CSV.", className="text-danger")

    results = []
    for _, row in df.iterrows():
        data = row[features].to_dict()
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                res = response.json()
                results.append({**data, **res})
            else:
                results.append({**data, "prediction": "Erreur", "probabilite": None})
        except:
            results.append({**data, "prediction": "Erreur", "probabilite": None})

    result_df = pd.DataFrame(results)
    return dash_table.DataTable(
        data=result_df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in result_df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
        style_header={"backgroundColor": "#007BFF", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {
                'if': {'filter_query': '{prediction} eq "non valide"'},
                'backgroundColor': '#f8d7da',
                'color': 'black'
            },
            {
                'if': {'filter_query': '{prediction} eq "valide"'},
                'backgroundColor': '#d4edda',
                'color': 'black'
            }
        ]
    )

if __name__ == '__main__':
    app.run(debug=True)
