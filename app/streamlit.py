import streamlit as st
import pandas as pd
import requests
import io
import joblib
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Analyse Parcours Étudiant",
    page_icon="🎓",
    layout="wide"
)

def get_modules_from_uploaded_file(df):
    """Extrait la liste des modules présents dans le fichier chargé qui commencent par Licence-GBM"""
    # Liste des colonnes à exclure (informations étudiant, moyenne, etc.)
    excluded_columns = ['N°', 'Nom', 'Prénom(s)', 'Moyenne générale']
    # Filtrer les colonnes qui commencent par "Licence-GBM"
    modules = [col for col in df.columns 
               if col not in excluded_columns 
               and str(col).startswith('Licence-GBM')]
    # print(f"Modules extraits du fichier : {modules}")  
    return modules


unite = {
    "Sciences_fondamentales": [
         "Licence-GBM-111 Sciences fondamentales I",
         "Licence-GBM-121 Sciences fondamentales II",
         "licence-GBM-115 Mécanique I",
         "Licence-GBM-125 Mécanique II"
     ],
     "Sciences_chimiques_biologiques": [
        "Licence-GBM-112 Sciences chimiques",
         "Licence-GBM-113 Sciences biologiques I",
         "Licence-GBM-124 Sciences biologiques II",
         "Licence-GBM-231 Sciences biologiques III"
     ],
     "Biophysique_imagerie_medicale": [
         "Licence-GBM-236 Biophysique I",
         "Licence-GBM-242 Biophysique II",
         "Licence-GBM-354 Traitement des signaux",
         "Licence-GBM-356 Techniques d'imagerie"
     ],
     "Electronique_Automatique_Informatique": [
        "Licence-GBM-114 Electricité-Electronique I",
         "Licence-GBM-123 Electricité-Electronique II",
         "Licence-GBM-233 Electricité-Electronique III",
         "Licence-GBM-122 Informatique I",
         "Licence-GBM-245 Informatique II",
         "Licence-GBM-244 Automatique - Système embarqué",
         "Licence-GBM-234 Automatismes & Informatique industrielle"
     ],
     "Maintenance_Biomedicale": [
         "Licence-GBM-241 Organisation et méthodes de maintenance I",
         "Licence-GBM-352 Organisation et méthodes de maintenance II",
         "Licence-GBM-353 Maintenance des systèmes",
         "Licence-GBM-355 Maintenance biomédicale"
     ],
     "Technologies_Biomédicales": [
         "Licence-GBM-235 Technologies biomédicales",
         "Licence-GBM-246 Instrumentation biomédicale I",
         "Licence-GBM-351 Instrumentation biomédicale II"
     ],
     "Competences_douces": [
         "Licence-GBM-116 Communication I",
         "Licence-GBM-126 Communication II",
         "Licence-GBM-362 Communication III",
         "Licence-GBM-361 Développement personnel"
     ],
     "QHSE": [
         "Licence-GBM-232 HQSE",
         "Licence-GBM-243 Gestion des risques"
     ]
 }
 

st.markdown("""
    <style>
    .student-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    .student-card h3 {
        color: #000000;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
        margin-top: 0;
    }
    .student-info {
        margin-top: 15px;
    }
    .student-info p {
        color: #000000;
        margin: 8px 0;
    }
    .student-info strong {
        color: #333333;
    }
    .probability-badge {
        font-weight: bold;
        padding: 3px 10px;
        border-radius: 15px;
        display: inline-block;
        margin-left: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 5px;
    }
    .stFileUploader>div>div>div>button {
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .student-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .impact-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f8f9fa;
    }
    .negative-impact {
        border-left-color: black;
    }
    </style>
    """, unsafe_allow_html=True)

# URL de l'API
API_URL = "http://127.0.0.1:8000/predict-file"
API_SHAP_URL = "http://127.0.0.1:8000/shap-explanation/"

# Chargement du modèle (cache pour performance)
@st.cache_resource
def load_model():
    return joblib.load("/Users/thiarakante/Documents/Databeez/prediction_parcours/src/components/artifacts/RandomForest_pipeline.joblib")

pipeline = load_model()
model = pipeline.named_steps["classifier"]

# Interface utilisateur
st.title("🎓 Prédiction du Parcours Étudiant")
st.markdown("""
    <div style='margin-bottom:2rem;'>
    Chargez un fichier Excel contenant les données étudiant pour obtenir :
    <ul>
        <li>Des prédictions de validation</li>
        <li>Des explications SHAP individuelles</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Section Upload
with st.expander("📤 Téléverser un fichier Excel", expanded=True):
    uploaded_file = st.file_uploader(
        "Glissez-déposez un fichier Excel ou cliquez pour parcourir",
        type=["xlsx", "xls"],
    )

if uploaded_file is not None:
    try:
        # Charger le fichier dans un dataframe
        df_etudiants = pd.read_excel(uploaded_file)
        
        # Afficher l'aperçu du fichier
        with st.expander("👀 Aperçu du fichier chargé", expanded=False):
            st.dataframe(df_etudiants.head(3))

        # Envoi à l'API pour prédiction globale
        with st.spinner("🔮 Calcul des prédictions en cours..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                df_results = pd.read_excel(io.BytesIO(response.content))
                
                # Affichage des résultats globaux
                st.success("✅ Prédictions effectuées avec succès !")
                with st.expander("📊 Résultats complets", expanded=True):
                    st.dataframe(df_results.style.highlight_max(axis=0, color='black'))

        # Section SHAP - Interface de recherche améliorée
        st.subheader("🔍 Analyse individuelle")
        
        # Interface de recherche à 3 colonnes avec autocomplétion
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Autocomplétion pour le nom
            noms_uniques = sorted(df_etudiants['Nom'].dropna().astype(str).unique().tolist())
            search_nom = st.selectbox(
                "Nom",
                options=noms_uniques,
                index=None,
                placeholder="Sélectionnez un nom...",
                help="Les noms proviennent du fichier chargé"
            )
        
        with col2:
            # Autocomplétion pour le prénom (filtré si un nom est sélectionné)
            if search_nom:
                prenoms_filtres = sorted(df_etudiants[df_etudiants['Nom'] == search_nom]['Prénom(s)']
                                      .dropna().astype(str).unique().tolist())
            else:
                prenoms_filtres = sorted(df_etudiants['Prénom(s)'].dropna().astype(str).unique().tolist())
            
            search_prenom = st.selectbox(
                "Prénom",
                options=prenoms_filtres,
                index=None,
                placeholder="Sélectionnez un prénom...",
                help="Filtré par nom si sélectionné"
            )
        
        with col3:
            # Autocomplétion pour le numéro étudiant (filtré si nom/prénom sélectionnés)
            if search_nom and search_prenom:
                numeros_filtres = sorted(df_etudiants[
                    (df_etudiants['Nom'] == search_nom) & 
                    (df_etudiants['Prénom(s)'] == search_prenom)
                ]['N°'].dropna().astype(str).unique().tolist())
            elif search_nom:
                numeros_filtres = sorted(df_etudiants[df_etudiants['Nom'] == search_nom]['N°']
                                     .dropna().astype(str).unique().tolist())
            else:
                numeros_filtres = sorted(df_etudiants['N°'].dropna().astype(str).unique().tolist())
            
            search_numero = st.selectbox(
                "N° étudiant",
                options=numeros_filtres,
                index=None,
                placeholder="Sélectionnez un numéro...",
                help="Filtré par nom/prénom"
            )
            
            search_btn = st.button("Analyser cet étudiant", type="primary", 
                                 help="Cliquez pour voir l'analyse détaillée")

        if search_btn:
            if not any([search_numero, search_nom, search_prenom]):
                st.warning("⚠️ Veuillez sélectionner au moins un critère de recherche")
            else:
                with st.spinner("🔍 Analyse SHAP en cours..."):
                    try:
                        # Préparation des paramètres pour l'API
                        params = {
                            "numero_etudiant": search_numero if search_numero else None,
                            "nom": search_nom if search_nom else None,
                            "prenom": search_prenom if search_prenom else None
                        }
                        
                        # Appel à l'API SHAP
                        response = requests.post(
                            API_SHAP_URL,
                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                            params={k: v for k, v in params.items() if v is not None},
                            timeout=15
                        )

                        if response.status_code == 200:
                            try:
                                # Vérifier si la réponse est bien un JSON
                                data = response.json()
                                if isinstance(data, str):
                                    # Si la réponse est une string, essayer de la convertir en dict
                                    import json
                                    data = json.loads(data)
                                
                                # Vérifier que data est bien un dictionnaire
                                if not isinstance(data, dict):
                                    raise ValueError("La réponse n'est pas un dictionnaire JSON valide")
                                
                                                            # Affichage des résultats
                               # Puis dans votre affichage des résultats :
                                st.markdown(f"""
                                    <div class="student-card">
                                        <h3>📝 Fiche étudiante</h3>
                                        <div class="student-info">
                                            <p><strong>N°:</strong> {data.get('etudiant', 'N/A')}</p>
                                            <p><strong>Nom:</strong> {data.get('nom', 'N/A')}</p>
                                            <p><strong>Prénom:</strong> {data.get('prenom', 'N/A')}</p>
                                            <p>
                                                <strong>Probabilité de validation:</strong> 
                                                <span class="probability-badge" style="
                                                    color: {'#e74c3c' if data.get('Probabilité de validation', 0) < 0.5 else '#2ecc71'};
                                                    background-color: {'#fde8e8' if data.get('Probabilité de validation', 0) < 0.5 else '#e8f8f0'};
                                                ">
                                                    {data.get('Probabilité de validation', 0):.1%}
                                                </span>
                                            </p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Affichage des impacts négatifs# Modifiez la section des impacts négatifs comme suit :
                                if data.get("impacts_negatifs"):
                                    st.subheader("📉 Facteurs à améliorer")
                                    
                                    # Récupérer les modules du fichier chargé
                                    current_semester_modules = get_modules_from_uploaded_file(df_etudiants)
                                    
                                    # Nouveau : Filtrer d'abord les impacts qui ont des modules dans le semestre
                                    filtered_impacts = []
                                    for impact in data["impacts_negatifs"]:
                                        if isinstance(impact, dict):
                                            thematique = impact.get('thematique', 'Inconnue')
                                            all_modules = unite.get(thematique, [])
                                            
                                            # Vérifier si au moins un module de la thématique est présent
                                            has_module_in_semester = any(
                                                any(module.replace(" ", "") in file_module.replace(" ", "") 
                                                    for file_module in current_semester_modules)
                                                for module in all_modules
                                            )
                                            
                                            if has_module_in_semester:
                                                filtered_impacts.append(impact)
                                    
                                    # Afficher seulement les impacts filtrés
                                    if not filtered_impacts:
                                        st.info("Aucun impact négatif significatif pour les modules de ce semestre")
                                    else:
                                        for impact in filtered_impacts:
                                            thematique = impact.get('thematique', 'Inconnue')
                                            all_modules = unite.get(thematique, [])
                                            
                                            # Trouver les modules correspondants
                                            relevant_modules = []
                                            for module in all_modules:
                                                for file_module in current_semester_modules:
                                                    if module.replace(" ", "") in file_module.replace(" ", ""):
                                                        relevant_modules.append(file_module)
                                                        break
                                            
                                            with st.expander(f"Impact #{impact.get('rang', '?')} - {thematique}", expanded=False):
                                                col1, col2 = st.columns([1, 2])
                                                
                                                with col1:
                                                    st.markdown(f"""
                                                    **Impact:** {impact.get('impact_negatif', 0):.4f}  
                                                    **Magnitude:** {impact.get('magnitude_impact', 0):.4f}  
                                                    **Interprétation:** {impact.get('interpretation', 'Non disponible')}
                                                    """)
                                                
                                                with col2:
                                                    st.markdown("**Modules concernés ce semestre:**")
                                                    for module in relevant_modules:
                                                        st.markdown(f"- {module}")
                                # Affichage de toutes les thématiques
                                with st.expander("📚 Modules par thématique (ce semestre)", expanded=False):
                                    current_semester_modules = get_modules_from_uploaded_file(df_etudiants)
                                    has_any_module = False
                                    
                                    for thematique, modules in unite.items():
                                        relevant_modules = []
                                        for module in modules:
                                            for file_module in current_semester_modules:
                                                if module.replace(" ", "") in file_module.replace(" ", ""):
                                                    relevant_modules.append(file_module)
                                                    break
                                        
                                        if relevant_modules:
                                            has_any_module = True
                                            st.markdown(f"**{thematique}**")
                                            for module in relevant_modules:
                                                st.markdown(f"- {module}")
                                            st.markdown("---")
                                    
                                    if not has_any_module:
                                        st.warning("Aucun module thématique trouvé dans le fichier chargé")
                                
                            except json.JSONDecodeError:
                                st.error("La réponse n'est pas au format JSON valide")
                                st.text(f"Réponse brute: {response.text[:500]}...")
                            except Exception as e:
                                st.error(f"Erreur de traitement: {str(e)}")
                                st.text(f"Réponse partielle: {str(data)[:500]}...")

                            # Ajoutez cette partie après l'affichage des impacts négatifs filtrés
                            if filtered_impacts:  # Utiliser filtered_impacts plutôt que data['impacts_negatifs']
                                st.subheader("📊 Visualisation des impacts par thématique")
                                
                                # Préparation des données pour le graphique
                                impacts_df = pd.DataFrame(filtered_impacts)
                                
                              # Ajouter le nombre de modules par thématique pour l'info-bulle
                                current_semester_modules = get_modules_from_uploaded_file(df_etudiants)
                                impacts_df['nb_modules'] = impacts_df['thematique'].apply(
                                    lambda x: len([m for m in unite.get(x, []) 
                                            if any(m.replace(" ", "") in fm.replace(" ", "") 
                                                    for fm in current_semester_modules)]))
                                                                
                                # Trier par magnitude d'impact
                                impacts_df = impacts_df.sort_values('magnitude_impact', ascending=False)
                                                                
                                # Création du graphique avec Plotly
                                fig = px.bar(
                                    impacts_df,
                                    x='thematique',
                                    y='magnitude_impact',
                                    color='magnitude_impact',
                                    color_continuous_scale='reds',
                                    title="Magnitude d'impact des thématiques défavorables",
                                    labels={
                                        'thematique': 'Thématique',
                                        'magnitude_impact': 'Impact négatif',
                                        'nb_modules': 'Nombre de modules'
                                    },
                                    hover_data=['nb_modules', 'interpretation'],
                                    text='magnitude_impact'
                                )
                                                                
                                # Personnalisation avancée du graphique
                                fig.update_traces(
                                    texttemplate='%{text:.3f}',
                                    textposition='outside',
                                    marker_line_color='rgb(150,0,0)',
                                    marker_line_width=1.5,
                                    hovertemplate=(
                                        "<b>%{x}</b><br>"
                                        "Impact: %{y:.4f}<br>"
                                        "Modules: %{customdata[0]}<br>"
                                        "%{customdata[1]}"
                                        "<extra></extra>"
                                    )
                                )
                                                                
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0.8)',  # Fond noir pour meilleur contraste
                                    paper_bgcolor='rgba(0,0,0,0.8)',
                                    xaxis=dict(
                                        tickangle=-45,
                                        title_font=dict(size=14, color='white'),
                                        tickfont=dict(size=12, color='white'),
                                        categoryorder='total descending'
                                    ),
                                    yaxis=dict(
                                        title_font=dict(size=14, color='white'),
                                        tickfont=dict(size=12, color='white'),
                                        gridcolor='rgba(255,255,255,0.1)'
                                    ),
                                    coloraxis_showscale=False,
                                    hoverlabel=dict(
                                        bgcolor='black',
                                        font_size=12,
                                        font_family='Arial',
                                        font_color='white'
                                    ),
                                    height=600,
                                    margin=dict(t=60, b=150),
                                    uniformtext_minsize=8,
                                    uniformtext_mode='hide',
                                    title_font=dict(color='white')  # Titre en blanc
                                )
                                                                
                                # Ajout de lignes de référence
                                mean_impact = impacts_df['magnitude_impact'].mean()
                                fig.add_hline(
                                    y=mean_impact,
                                    line_dash='dot',
                                    line_color='grey',
                                    annotation_text=f'Moyenne: {mean_impact:.3f}',
                                    annotation_position='top right',
                                    annotation_font=dict(color='white')
                                )
                                                                
                                # Affichage du graphique
                                st.plotly_chart(fig, use_container_width=True)
                                                                
                                # Analyse des résultats
                                max_impact = impacts_df.iloc[0]
                                st.markdown(f"""
                                <div style="
                                    background-color: black;
                                    padding: 15px;
                                    border-radius: 10px;
                                    margin-top: 20px;
                                    color: white;
                                ">
                                    <h4 style="color: #e74c3c;">Analyse des résultats</h4>
                                    <p><strong>Thématique la plus impactante:</strong> {max_impact['thematique']} (Impact: {max_impact['magnitude_impact']:.4f})</p>
                                    <p><strong>Nombre de modules concernés:</strong> {max_impact['nb_modules']}</p>
                                    <p><strong>Interprétation:</strong> {max_impact['interpretation']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error(f"Erreur API (HTTP {response.status_code}): {response.text[:500]}...")
                            
                    except requests.exceptions.Timeout:
                        st.error("⏳ Délai d'attente dépassé - Veuillez réessayer")
    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")

# Sidebar (inchangé)
with st.sidebar:
    st.image("/Users/thiarakante/Documents/Databeez/prediction_parcours/app/logo1.png", use_container_width=True)
    st.markdown("""
        ### 📚 Aide
        - **Format fichier** : Doit être un fichier Excel (.xlsx ou .xls)
        - **Recherche** : Par numéro, nom, prénom ou combinaison
        - **Impacts négatifs** : Thématiques qui réduisent la probabilité de validation
        """)
    
    st.markdown("---")
    st.markdown("""
        <small>
        Développé par <strong>DATABEEZ</strong> | Rue 4 X A, Point E - Immeuble Arcanes<br>
        +221 77 838 78 87 | <a href="mailto:info@data-beez.com">info@data-beez.com</a><br>
        Version 1.1.0
        </small>
        """, unsafe_allow_html=True)                                                              