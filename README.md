
# Détection de Données Out-Of-Distribution (OOD)

## Introduction
Ce projet vise à explorer et à évaluer l'efficacité de différentes méthodes de détection de données out-of-distribution (OOD) dans les réseaux de neurones. En particulier, nous nous concentrons sur l'utilisation de l'Analyse en Composantes Principales (PCA) combinée avec la distance de Mahalanobis, ainsi que la Factorisation Matricielle Non-négative (NMF) pour identifier les entrées OOD par rapport à des données in-distribution (ID), near OOD, et far OOD. Le projet teste ces méthodes sur plusieurs jeux de données, offrant un cadre de benchmarking complet pour évaluer leur performance.

## Installation

Pour installer les dépendances requises pour ce projet, veuillez exécuter la commande suivante :

```bash
pip install -r requirements.txt
```

## Utilisation

Pour exécuter les scripts de benchmarking sur les jeux de données inclus, suivez les instructions ci-dessous :

1. Placez vos ensembles de données dans le dossier approprié sous `datasets/`.
2. Pour exécuter la méthode PCA + Mahalanobis :

```bash
python src/pca_mahalanobis/run_benchmark.py
```

3. Pour exécuter la méthode NMF + Mahalanobis :

```bash
python src/nmf_mahalanobis/run_benchmark.py
```

Les résultats seront stockés dans le dossier `results/` correspondant.

## Structure du Projet

Le projet est organisé comme suit :

- `datasets/` : Contient les ensembles de données organisés par type (ID, near OOD, far OOD).
- `models/` : Stocke les modèles pré-entraînés sur chaque ensemble de données ID.
- `src/` : Contient le code source pour les méthodes de détection OOD et les utilitaires.
- `results/` : Dossier pour stocker les résultats des benchmarks.
- `notebooks/` : Jupyter notebooks pour l'analyse exploratoire et les visualisations.

## Contribution

Nous encourageons les contributions à ce projet ! Si vous avez des suggestions d'amélioration ou si vous souhaitez ajouter de nouvelles méthodes de détection OOD, veuillez ouvrir une issue ou soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
```

N'oubliez pas d'ajuster le contenu selon les spécificités de votre projet, comme les détails des scripts d'exécution ou les instructions spécifiques pour l'ajout de nouveaux jeux de données. Ce modèle de `README.md` fournit une base pour une documentation claire et complète de votre projet.