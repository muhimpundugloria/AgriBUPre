# AgriPredict Burundi

Application de prédiction des récoltes au Burundi construite avec Streamlit et scikit-learn.

## Structure du projet

- `app/app.py` : application Streamlit interactive.
- `data/agriculture_burundi.csv` : dataset agricole.
- `train.py` : pipeline de prétraitement, entraînement et sauvegarde des modèles.
- `models/` : modèles entraînés et métadonnées (`decision_tree.pkl`, `random_forest.pkl`, `logistic_regression.pkl`, `scaler.pkl`, `metadata.json`).
- `requirements.txt` : environnement Python requis.

## Environnement

1. Activez le venv :
   ```powershell
   .\venv\Scripts\Activate
   ```
2. Installez les dépendances si nécessaire :
   ```powershell
   pip install -r requirements.txt
   ```

## Entraîner les modèles

Exécutez :
```powershell
python train.py
```

Cela génère les fichiers suivants dans `models/` :
- `decision_tree.pkl`
- `random_forest.pkl`
- `logistic_regression.pkl`
- `scaler.pkl`
- `metadata.json`

## Lancer l'application Streamlit

Exécutez :
```powershell
streamlit run app/app.py
```

Ensuite, ouvrez l'URL fournie par Streamlit dans votre navigateur.

## Notes

- Le pipeline remplace les valeurs manquantes par des imputations adaptées (`pluviométrie` par médiane, `utilisation_engrais` par mode).
- Les colonnes `rendement_t_ha` et `production_totale_t` sont exclues des features pour éviter les fuites de données.
- La variable cible est `bonne_recolte`.

## Interface

- L’application Streamlit (`app/app.py`) a été refondue (nouvelle présentation, nouveau design et catégorisation de la probabilité) tout en conservant la compatibilité avec `metadata.json` et le format des features.

