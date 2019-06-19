# TODO :

## Prépation des données

- Séparer dans 2 dossiers les images (noisy, not noisy) 
  - Par scène
  - Par zone
  - Par métrique [scene, zone]
  
- Transformer chaque image comme souhaitée (ici reconstruction SV avec 110 composantes faibles)
- Pour chaque image ajouter sa forme sous 4 rotations (augmentation du nombre de données)

## Chargement des données
- Chargement de l'ensemble des images (association : "path", "label")
- Mettre en place un équilibre de classes
- Mélange de l'ensemble des données
- Séparation des données (train, validation, test)

## Conception du modèle
- Mise en place d'un modèle CNN
- Utilisation BatchNormalization / Dropout


## Si non fonctionnel
- Utilisation d'une approche transfer learning