import pandas as pd


def replace_col_names(df, unite):
    new_cols = []
    ue_count = {}
    for col in df.columns:
        found = False
        for ue, col_names in unite.items():
            if col in col_names:
                ue_count[ue] = ue_count.get(ue, 0) + 1
                new_col = f"{ue}_{ue_count[ue]}" if ue_count[ue] > 1 else ue
                new_cols.append(new_col)
                found = True
                break
        if not found:
            new_cols.append(col)
    df.columns = new_cols
    return df


semester_mapping = {
    'L1GBM': {'S1': 'S1', 'S2': 'S2'},
    'L2GBM': {'S1': 'S3', 'S2': 'S4'},
    'L3GBM': {'S1': 'S5', 'S2': 'S6'},
}

def extract_semester_from_filename(filename):
    filename_only = filename.split("/")[-1].replace(".xls", "").replace(".xlsx", "")
    parts = filename_only.split("_")
    if len(parts) >= 3:
        niveau = parts[1]
        semestre_brut = parts[2]
        if niveau in semester_mapping and semestre_brut in semester_mapping[niveau]:
            return semester_mapping[niveau][semestre_brut]
    return "Inconnu"

def process_file(file, filename, unite):
    df = pd.read_excel(file).dropna(axis=0, how='all')

    # Renommer les colonnes
    df = replace_col_names(df, unite)

    # Garder les colonnes importantes
    categories = list(unite.keys())
    colonnes_a_garder = ['N°', 'Prénom(s)', 'Nom'] + categories
    df = df.loc[:, df.columns.intersection(colonnes_a_garder)]

    # Ajouter le semestre
    df["Semestre"] = extract_semester_from_filename(filename)

    df.fillna(0, inplace=True)
    
    return df
