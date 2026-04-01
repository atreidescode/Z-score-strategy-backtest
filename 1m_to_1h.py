import pandas as pd

# 1. Remplacez ces noms par les noms EXACTS de vos fichiers HistData décompressés
fichiers = [
    "DAT_ASCII_SPXUSD_M1_2023.csv", 
    "DAT_ASCII_SPXUSD_M1_2024.csv", 
    "DAT_ASCII_SPXUSD_M1_2025.csv"
]

liste_df = []

print("Lecture et assemblage des fichiers en cours...")
# 2. Boucle pour lire et combiner les 3 années
for fichier in fichiers:
    df_temp = pd.read_csv(fichier, sep=';', header=None, 
                          names=['Date_Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    liste_df.append(df_temp)

# Fusionner en un seul grand tableau (Dataframe)
df_complet = pd.concat(liste_df, ignore_index=True)

print("Formatage des dates (cela peut prendre quelques secondes)...")
# 3. Convertir le texte en format Date compréhensible par Python
df_complet['Date_Time'] = pd.to_datetime(df_complet['Date_Time'], format='%Y%m%d %H%M%S')
df_complet.set_index('Date_Time', inplace=True)

# Trier chronologiquement par sécurité
df_complet.sort_index(inplace=True)

print("Conversion des bougies 1 Minute en 1 Heure...")
# 4. Resampling (M1 -> H1)
df_h1 = df_complet.resample('1h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna() # On enlève les heures vides (ex: week-ends)

print("\n--- OPÉRATION TERMINÉE ---")
print(f"Vous avez maintenant {len(df_h1)} bougies horaires (H1) sur 3 ans.")
print("\nAperçu des 5 premières lignes :")
print(df_h1.head())

# 5. Sauvegarde du fichier final propre
nom_fichier_final = "historique_3ans_H1.csv"
df_h1.to_csv(nom_fichier_final)
print(f"\nFichier '{nom_fichier_final}' sauvegardé avec succès et prêt pour le backtest !")