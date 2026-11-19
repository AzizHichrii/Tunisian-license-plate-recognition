# Tunisian License Plate Recognition (TLPR)

## Cahier de développement détaillé --- Deep Learning, Computer Vision et MLOps

**Version :** 1.0\
**Type de projet :** AI Engineering / Deep Learning / MLOps\
**Objectif :** construire une application capable de détecter les
plaques d'immatriculation tunisiennes dans une image ou une vidéo, d'en
extraire la région, puis de reconnaître les caractères et de retourner
le résultat sous forme de texte.

------------------------------------------------------------------------

# 1. Vision et périmètre du projet

## 1.1 Objectif fonctionnel

Le système doit suivre cette chaîne :

1.  Recevoir une image ou une frame vidéo.
2.  Détecter une ou plusieurs plaques d'immatriculation.
3.  Extraire chaque plaque (crop).
4.  Prétraiter l'image de la plaque si nécessaire.
5.  Reconnaître les caractères avec un modèle OCR.
6.  Normaliser et valider le texte selon les formats de plaques retenus.
7.  Retourner le texte, les coordonnées, les scores et le temps de
    traitement.
8.  Enregistrer les métriques et, si la politique de conservation
    l'autorise, les métadonnées du résultat.

Exemple illustratif de sortie : `123 TUN 4567`. Le format réel à
reconnaître doit être défini à partir de références tunisiennes
vérifiées et des catégories de plaques incluses dans le périmètre.

## 1.2 Périmètre recommandé pour la première version

Commencer par une version maîtrisable :

-   Une image à la fois.
-   Plaques standard tunisiennes clairement visibles.
-   Une plaque peut être absente ou plusieurs plaques peuvent être
    présentes.
-   Détection par YOLO.
-   OCR baseline avec EasyOCR ou un moteur OCR comparable.
-   Pipeline complet fonctionnant localement.
-   API HTTP pour exposer l'inférence.

Ajouter ensuite, après validation :

-   Reconnaissance CRNN entraînée sur les données du projet.
-   Vidéo et caméra en direct.
-   Plaques inclinées, éloignées, floues ou prises de nuit.
-   Détection de catégories de plaques supplémentaires.
-   Déploiement distant, monitoring avancé et processus de
    réentraînement.

## 1.3 Critères de réussite

Définir les objectifs chiffrés après la constitution d'un premier
dataset et l'établissement d'une baseline. Ne pas fixer arbitrairement
une accuracy cible avant d'avoir mesuré la difficulté des données.

Les résultats à mesurer :

-   Qualité de détection : Precision, Recall, mAP@50, mAP@50-95.
-   Qualité OCR : Character Error Rate (CER), Word/Sequence Exact Match
    Accuracy.
-   Qualité end-to-end : proportion des plaques réelles correctement
    détectées et entièrement reconnues.
-   Performance système : latence moyenne et percentiles (p50, p95),
    débit, mémoire et taux d'erreur.
-   Robustesse : résultats par condition (jour, nuit, pluie, angle,
    distance, flou).

------------------------------------------------------------------------

# 2. Architecture cible

## 2.1 Pipeline d'inférence

``` text
Image / Video Frame
        |
        v
Input validation
        |
        v
YOLO License Plate Detector
        |
        v
Bounding boxes + detection confidence
        |
        v
Plate crop + perspective correction (if needed)
        |
        v
Image preprocessing
        |
        v
OCR engine (EasyOCR baseline / CRNN)
        |
        v
Text normalization
        |
        v
Tunisian plate format validation
        |
        v
Structured response + logs + metrics
```

## 2.2 Architecture MLOps

``` text
Data sources
    |
    v
Raw data + metadata
    |
    v
Data validation and annotation
    |
    v
DVC dataset version
    |
    v
Preprocessing pipeline
    |
    v
Train / validation / test split
    |
    v
YOLO training ---------------- OCR training
    |                               |
    v                               v
MLflow experiments             MLflow experiments
    |                               |
    +---------------+---------------+
                    |
                    v
             Model evaluation
                    |
                    v
       Model registry / approved artifacts
                    |
                    v
          Inference pipeline + FastAPI
                    |
                    v
             Docker deployment
                    |
                    v
       Prometheus metrics + Grafana
                    |
                    v
     Feedback / reviewed new labeled data
                    |
                    v
       DVC version + controlled retraining
```

## 2.3 Stack technique recommandée

  -----------------------------------------------------------------------
  Domaine                 Outil                   Utilisation
  ----------------------- ----------------------- -----------------------
  Langage                 Python                  Code data, entraînement
                                                  et backend

  Environnement           venv ou conda           Isolation des
                                                  dépendances

  Computer Vision         OpenCV                  Lecture, crop,
                                                  transformations et
                                                  vidéo

  Deep Learning           PyTorch                 Entraînement et
                                                  expérimentation OCR

  Détection               Ultralytics YOLO        Localisation des
                          (version choisie après  plaques
                          benchmark)              

  OCR baseline            EasyOCR                 Première référence OCR

  OCR avancé              CRNN + CNN + BiLSTM +   Modèle OCR entraînable
                          CTC                     

  Annotation              CVAT ou Roboflow        Bounding boxes et
                                                  labels texte

  Analyse                 pandas, NumPy,          Analyse et
                          Matplotlib              visualisation des
                                                  données

  Data validation         scripts Python /        Contrôle des fichiers
                          Pydantic                et annotations

  Data versioning         DVC + Git               Versions des datasets
                                                  et pipelines

  Experiment tracking     MLflow                  Paramètres, métriques
                                                  et artefacts

  Backend                 FastAPI + Pydantic      API d'inférence

  Base de données         PostgreSQL              Métadonnées et
                                                  historique si
                                                  nécessaire

  Tests                   pytest                  Tests unitaires et
                                                  intégration

  Qualité du code         Ruff                    Linting et formatage

  Conteneurs              Docker + Docker Compose Environnements
                                                  reproductibles

  CI                      GitHub Actions          Tests et build
                                                  automatisés

  Monitoring              Prometheus + Grafana    Métriques techniques

  Documentation API       Swagger / OpenAPI via   Documentation des
                          FastAPI                 endpoints
  -----------------------------------------------------------------------

**Règle de simplicité :** ne pas installer et configurer tous les outils
dès le premier jour. Ajouter chaque composant lorsqu'il devient utile
dans le workflow.

------------------------------------------------------------------------

# 3. Phase 0 --- Cadrage, spécifications et préparation

## Objectif

Définir exactement ce que le système doit reconnaître, dans quelles
conditions et avec quelles contraintes.

## Tâches détaillées

### 0.1 Étudier les plaques visées

-   Identifier les formats tunisiens concernés à partir de sources
    officielles ou fiables.
-   Répertorier les catégories incluses dans la première version.
-   Documenter l'ordre des chiffres, lettres ou mots, les séparateurs et
    les variations visuelles.
-   Définir les caractères autorisés et les règles de normalisation.
-   Prévoir explicitement les cas qui ne seront pas pris en charge au
    début.
-   Ne pas supposer qu'un seul format couvre toutes les plaques
    tunisiennes.

### 0.2 Définir les scénarios d'utilisation

-   Photo prise de face.
-   Photo prise en angle.
-   Plusieurs véhicules dans la même image.
-   Plaque partiellement masquée.
-   Plaque éloignée ou de petite taille.
-   Image de jour et de nuit.
-   Reflets, pluie, flou de mouvement et faible résolution.
-   Image sans plaque.

### 0.3 Définir les entrées et sorties

Entrées : - Fichier image JPG, JPEG ou PNG. - Plus tard : flux vidéo ou
caméra.

Sorties par plaque : - Texte reconnu. - Bounding box. - Score de
détection. - Score OCR si disponible. - Statut de validation du
format. - Temps de traitement. - Message d'erreur ou statut si la
reconnaissance échoue.

Ne pas présenter un score OCR et un score YOLO comme s'il s'agissait de
la même mesure. Un score combiné ne doit être utilisé que si une méthode
de combinaison est définie et documentée.

### 0.4 Définir les contraintes

-   Matériel disponible (CPU, GPU, VRAM).
-   Budget d'entraînement et de stockage.
-   Sources de données autorisées.
-   Politique de confidentialité et durée de conservation.
-   Temps de réponse attendu.
-   Démonstration locale ou déploiement distant.

### 0.5 Écrire les spécifications

Créer : - `docs/requirements.md` - `docs/data_policy.md` -
`docs/architecture.md` - `docs/acceptance_criteria.md`

## Livrables

-   Périmètre fonctionnel validé.
-   Formats de plaques documentés.
-   Scénarios de test définis.
-   Contraintes matérielles et données connues.

## Critère de passage

Ne pas commencer l'entraînement tant que les formats ciblés, les données
nécessaires et les critères d'évaluation ne sont pas définis.

------------------------------------------------------------------------

# 4. Phase 1 --- Initialisation du dépôt et de l'environnement

## Objectif

Créer un projet propre, reproductible et organisé.

## Tâches détaillées

### 1.1 Créer le dépôt Git

-   Créer un dépôt GitHub privé au début si les données ou les
    annotations ne sont pas publiables.
-   Ajouter un README initial.
-   Créer `.gitignore`.
-   Définir une convention de branches simple : `main`, `develop` si
    nécessaire, et branches de fonctionnalités.
-   Faire des commits petits et descriptifs.

### 1.2 Créer l'environnement Python

-   Vérifier la version de Python compatible avec les bibliothèques
    retenues.
-   Créer un environnement virtuel.
-   Installer progressivement les dépendances.
-   Enregistrer les versions dans `requirements.txt` ou utiliser un
    gestionnaire de dépendances verrouillé.
-   Vérifier l'installation de PyTorch et la disponibilité du GPU.

Exemple :

``` bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas opencv-python matplotlib scikit-learn
pip install torch torchvision
pip install ultralytics
pip install fastapi uvicorn pydantic
pip install mlflow dvc pytest ruff
```

Adapter l'installation de PyTorch à la version CUDA et au système
réellement utilisés. Ne pas copier une commande CUDA sans vérifier la
compatibilité du GPU et du pilote.

### 1.3 Vérifier le matériel

-   Relever le modèle du GPU et la VRAM.
-   Vérifier le pilote NVIDIA et CUDA si GPU NVIDIA.
-   Lancer un petit test PyTorch.
-   Mesurer la mémoire disponible.
-   Décider si l'entraînement sera local ou sur un GPU cloud.

### 1.4 Créer la structure de dossiers

``` text
tunisian-lpr/
├── configs/
│   ├── detection.yaml
│   └── ocr.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── docs/
├── notebooks/
├── src/
│   ├── data/
│   ├── detection/
│   ├── ocr/
│   ├── pipeline/
│   ├── api/
│   └── utils/
├── models/
│   ├── detection/
│   └── ocr/
├── tests/
├── scripts/
├── monitoring/
├── .github/
│   └── workflows/
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Les grands fichiers de données et les poids des modèles ne doivent pas
être commités directement dans Git. Les gérer avec DVC ou un stockage
d'artefacts adapté.

### 1.5 Ajouter les premières règles de qualité

-   Configurer Ruff.
-   Configurer pytest.
-   Ajouter un fichier de configuration central.
-   Définir les chemins par variables d'environnement ou configuration.
-   Ne jamais committer de secrets, tokens ou mots de passe.

## Livrables

-   Dépôt initialisé.
-   Environnement Python fonctionnel.
-   GPU vérifié ou stratégie cloud définie.
-   Arborescence créée.
-   Dépendances documentées.

## Critère de passage

Un script Python simple doit pouvoir importer les bibliothèques
principales et détecter le matériel disponible sans erreur.

------------------------------------------------------------------------

# 5. Phase 2 --- Recherche, collecte et gestion des données

## Objectif

Construire un dataset représentatif de plaques tunisiennes, avec des
droits d'utilisation vérifiés.

## Tâches détaillées

### 2.1 Rechercher des sources

Explorer : - Roboflow Universe. - Kaggle. - Datasets académiques de
détection de plaques. - Sources institutionnelles ou partenaires
autorisés. - Collecte personnelle avec autorisation appropriée.

Pour chaque source, noter : - URL ou origine. - Auteur / organisation. -
Licence. - Conditions d'utilisation. - Date de collecte. - Pays et type
de plaques. - Présence des annotations. - Résolution et format. -
Limites connues.

Ne pas télécharger ou réutiliser automatiquement des images simplement
parce qu'elles sont accessibles sur Internet. Vérifier les droits et les
conditions d'utilisation.

### 2.2 Constituer un dataset adapté à la Tunisie

-   Vérifier que les formats sont réellement tunisiens.
-   Identifier les catégories de plaques.
-   Collecter des conditions de prise de vue variées.
-   Éviter un dataset composé uniquement d'images frontales et nettes.
-   Documenter les sources et les catégories.
-   Préserver les images originales dans `data/raw/`.

Un dataset étranger peut servir à expérimenter ou préentraîner, mais les
données tunisiennes sont indispensables pour évaluer l'objectif
tunisien.

### 2.3 Créer deux ensembles de données logiques

**Dataset Detection :** - Image complète. - Bounding box de chaque
plaque. - Classe de l'objet, initialement `license_plate`. - Identifiant
de l'image et source.

**Dataset OCR :** - Image de plaque recadrée. - Transcription exacte. -
Identifiant de l'image source. - Catégorie de plaque. - Informations de
qualité facultatives.

### 2.4 Définir les métadonnées

Créer un fichier CSV ou Parquet avec des champs tels que :

``` text
image_id
file_path
source
license
capture_condition
plate_category
vehicle_group_id
annotation_status
split
```

Ne pas enregistrer de données personnelles non nécessaires.

### 2.5 Détecter les doublons et quasi-doublons

-   Calculer les hashes des fichiers.
-   Utiliser un perceptual hash pour repérer les images presque
    identiques.
-   Examiner les doublons manuellement.
-   Grouper les images provenant de la même séquence vidéo ou du même
    véhicule.
-   Éviter qu'une même scène soit répartie entre train et test.

### 2.6 Contrôler la qualité des données

Vérifier : - Fichiers ouvrables. - Extensions et formats. -
Dimensions. - Images vides ou corrompues. - Images sans plaque. -
Plaques illisibles. - Doublons. - Labels manquants ou incohérents. -
Distribution des catégories et conditions de prise de vue.

### 2.7 Définir une taille de dataset progressive

Commencer avec un petit échantillon pilote, puis augmenter selon les
résultats.

Objectif de collecte initial à étudier : - Environ 2 000 à 3 000 images
complètes pour la détection. - Environ 3 000 à 5 000 crops de plaques
pour l'OCR.

Ce sont des objectifs de planification, pas des garanties de
performance. La quantité utile dépend de la diversité, de la qualité et
de l'exactitude des labels.

## Outils

-   Python, pathlib, pandas, Pillow, OpenCV.
-   Roboflow ou CVAT pour l'annotation.
-   DVC pour versionner les données.
-   Git pour versionner les scripts et métadonnées.

## Livrables

-   Registre des sources et licences.
-   Dataset brut.
-   Rapport de qualité.
-   Inventaire des catégories.
-   Détection des doublons.

## Critère de passage

Les données doivent être légalement utilisables, lisibles, traçables et
suffisamment représentatives pour préparer une première expérimentation.

------------------------------------------------------------------------

# 6. Phase 3 --- Annotation et préparation des données

## Objectif

Créer des annotations fiables et des jeux train, validation et test sans
fuite de données.

## Tâches détaillées

### 3.1 Annoter la détection

Avec CVAT ou Roboflow :

1.  Créer un projet Object Detection.
2.  Définir la classe `license_plate`.
3.  Importer les images.
4.  Tracer une bounding box serrée autour de chaque plaque visible.
5.  Annoter toutes les plaques pertinentes dans l'image.
6.  Marquer les cas ambigus selon une règle documentée.
7.  Faire contrôler un échantillon par une deuxième personne si
    possible.
8.  Exporter dans le format YOLO.
9.  Vérifier les fichiers labels exportés.

Structure YOLO typique :

``` text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Chaque ligne de label YOLO contient :

``` text
class_id x_center y_center width height
```

Les coordonnées sont normalisées entre 0 et 1.

### 3.2 Valider les annotations YOLO

Créer un script qui vérifie : - Un fichier label par image lorsque
nécessaire. - Les valeurs numériques. - Les coordonnées comprises dans
les limites autorisées. - Les classes connues. - Les boîtes de largeur
et hauteur positives. - Les fichiers manquants. - Les images sans objet,
si elles sont volontairement conservées comme négatifs.

Générer des images de contrôle avec les bounding boxes dessinées.

### 3.3 Préparer les labels OCR

Créer un CSV :

``` csv
image_path,text,source_image_id,plate_category
data/ocr/plate_001.jpg,"TRANSCRIPTION_EXACTE",image_001,standard
```

Les transcriptions doivent être vérifiées caractère par caractère.

Définir : - L'alphabet supporté. - Le traitement des espaces. - Les
chiffres et caractères arabes ou latins selon les formats retenus. - La
normalisation Unicode si l'arabe est inclus. - La représentation
canonique du texte. - Les règles de comparaison pour l'évaluation.

Ne pas transformer une transcription de façon à masquer une erreur
réelle du modèle.

### 3.4 Créer les crops OCR

Deux possibilités : - Utiliser les bounding boxes annotées pour générer
les crops. - Utiliser les crops validés manuellement.

Conserver le lien entre chaque crop et l'image d'origine.

### 3.5 Créer train / validation / test

Répartition initiale possible : - 70 % train. - 15 % validation. - 15 %
test.

La séparation doit être faite par groupe de véhicule, séquence ou source
lorsque cela est pertinent, et non uniquement image par image.

Règles : - Train : apprentissage. - Validation : choix des
hyperparamètres et du modèle. - Test : évaluation finale, utilisé le
moins possible pendant le développement. - Ne pas entraîner sur les
images de test. - Documenter la graine aléatoire et la méthode de split.

### 3.6 Prétraitement et augmentation

Pour la détection : - Redimensionnement géré par la configuration
YOLO. - Augmentations géométriques et photométriques raisonnables. -
Mosaic ou autres augmentations disponibles, si elles sont adaptées. -
Contrôle visuel des transformations.

Pour OCR : - Redimensionnement avec conservation du ratio. - Padding. -
Ajustement du contraste. - Denoising si bénéfique. - Perspective
correction si nécessaire. - Augmentation légère de luminosité,
contraste, flou et bruit.

Éviter de déformer les caractères ou de créer des exemples irréalistes.

### 3.7 Analyse exploratoire

Créer des visualisations : - Nombre d'images par split. - Nombre de
plaques par image. - Dimensions des images. - Distribution des tailles
de bounding boxes. - Répartition des catégories. - Répartition jour/nuit
et autres conditions. - Longueur des transcriptions OCR. - Fréquence des
caractères.

## Livrables

-   Dataset YOLO annoté.
-   Dataset OCR annoté.
-   Scripts de validation.
-   Splits documentés.
-   Rapport exploratoire.
-   Exemples visuels contrôlés.

## Critère de passage

Les annotations doivent être valides et les données
train/validation/test doivent être séparées sans fuite identifiable.

------------------------------------------------------------------------

# 7. Phase 4 --- Versioning des données avec DVC

## Objectif

Pouvoir retrouver précisément les données utilisées pour chaque
expérimentation.

## Tâches détaillées

### 4.1 Initialiser DVC

``` bash
pip install dvc
dvc init
```

### 4.2 Ajouter les données au suivi DVC

``` bash
dvc add data/raw
dvc add data/processed
```

DVC crée des fichiers `.dvc` suivis par Git, tandis que les données sont
gérées dans son cache.

### 4.3 Configurer un remote

Choisir un stockage compatible avec DVC : - Stockage cloud objet
compatible. - Stockage distant d'équipe. - Stockage local partagé pour
un environnement de développement.

Configurer les identifiants via un mécanisme sécurisé, jamais dans le
dépôt.

### 4.4 Versionner les changements

À chaque version importante : - Mettre à jour les données. - Relancer
les contrôles qualité. - Mettre à jour les fichiers DVC. - Commiter les
métadonnées dans Git. - Envoyer les données vers le remote. - Ajouter un
tag Git pour les versions importantes.

### 4.5 Tracer les datasets

Créer un identifiant de version de dataset et l'associer à chaque run
MLflow.

## Livrables

-   DVC initialisé.
-   Remote configuré.
-   Versions de datasets récupérables.
-   Documentation de restauration.

## Critère de passage

Une autre copie du projet doit pouvoir récupérer exactement la version
de dataset associée à une expérimentation.

------------------------------------------------------------------------

# 8. Phase 5 --- Modèle de détection YOLO

## Objectif

Entraîner un détecteur qui localise les plaques dans une image complète.

## Tâches détaillées

### 5.1 Établir une baseline

-   Choisir une première version YOLO supportée.
-   Commencer avec un poids pré-entraîné adapté.
-   Vérifier le format du dataset.
-   Lancer une expérimentation courte.
-   Vérifier que les loss diminuent et que les prédictions sont
    cohérentes.
-   Visualiser les résultats sur des images non utilisées pour
    l'entraînement.

### 5.2 Configurer l'entraînement

Paramètres à expérimenter : - Architecture et taille du modèle. - Image
size. - Batch size. - Nombre d'epochs. - Learning rate. - Optimizer. -
Weight decay. - Early stopping / patience. - Seed. - Device (CPU ou
GPU).

Ne pas changer tous les paramètres simultanément. Procéder par
expériences contrôlées.

### 5.3 Comparer les versions et tailles

Comparer, selon la compatibilité et le budget : - YOLOv8n / YOLOv8s. -
YOLO11n / YOLO11s. - Éventuellement une génération plus récente.

Comparer les tailles de modèles de façon équitable avec : - Même dataset
et même split. - Résolution comparable. - Budget d'entraînement
documenté. - Même protocole d'évaluation.

### 5.4 Mesurer les performances

Enregistrer : - Precision. - Recall. - mAP@50. - mAP@50-95. - Confusion
matrix si pertinente. - Temps d'entraînement. - Latence d'inférence. -
Taille des poids. - Consommation mémoire.

### 5.5 Analyser les erreurs

Inspecter : - Faux positifs. - Faux négatifs. - Plaques petites. -
Plaques inclinées. - Faible luminosité. - Reflets. - Occlusions. -
Images contenant plusieurs plaques.

### 5.6 Sélectionner le détecteur

Choisir selon le compromis entre qualité, vitesse, taille et robustesse,
pas uniquement selon une métrique.

### 5.7 Exporter le modèle

-   Sauvegarder les poids sélectionnés.
-   Enregistrer la configuration.
-   Documenter la version du code et du dataset.
-   Tester le chargement dans un script d'inférence indépendant.

## Outils

-   PyTorch.
-   Ultralytics YOLO.
-   OpenCV.
-   MLflow pour le suivi.
-   DVC pour les données.
-   Matplotlib pour les graphiques.

## Livrables

-   Modèle YOLO baseline.
-   Plusieurs runs comparables.
-   Rapport de métriques.
-   Analyse des erreurs.
-   Artefact du modèle sélectionné.

## Critère de passage

Le détecteur doit localiser correctement les plaques sur le jeu de
validation et produire des prédictions exploitables par l'étape OCR.

------------------------------------------------------------------------

# 9. Phase 6 --- Développement OCR

## Objectif

Lire correctement les caractères présents dans les plaques recadrées.

## Tâches détaillées

### 6.1 Construire une baseline OCR

-   Installer EasyOCR.
-   Définir les langues et caractères compatibles avec les plaques
    retenues.
-   Tester sur les crops annotés.
-   Sauvegarder les prédictions.
-   Comparer les textes prédits aux transcriptions de référence.
-   Identifier les erreurs fréquentes.

Vérifier la prise en charge réelle de l'écriture et des caractères du
format choisi. Une configuration linguistique inadéquate peut dégrader
fortement les résultats.

### 6.2 Préparer un benchmark OCR

-   Utiliser les mêmes crops de test pour chaque moteur.
-   Conserver les labels de référence.
-   Définir une normalisation avant comparaison.
-   Mesurer le temps d'inférence.
-   Segmenter les résultats par qualité d'image et catégorie de plaque.

### 6.3 Développer CRNN si les données le permettent

Architecture indicative :

``` text
Plate image
    |
    v
CNN feature extractor
    |
    v
Sequence feature map
    |
    v
BiLSTM / recurrent layers
    |
    v
Character probabilities
    |
    v
CTC loss / decoding
    |
    v
Recognized sequence
```

Tâches : - Définir l'alphabet. - Encoder les caractères. - Construire le
Dataset et DataLoader PyTorch. - Définir l'architecture CNN + BiLSTM. -
Implémenter CTC Loss. - Implémenter le décodage greedy, puis beam search
si utile. - Entraîner sur les crops annotés. - Sauvegarder checkpoints
et configuration. - Évaluer sur validation. - Évaluer une seule fois le
modèle final sur test.

### 6.4 Mesurer les performances OCR

-   CER : taux d'erreur au niveau caractère.
-   Exact sequence match : séquence entière identique.
-   Character accuracy.
-   Temps d'inférence.
-   Résultats par condition de capture.

### 6.5 Comparer EasyOCR et CRNN

Comparer : - Qualité. - Temps. - Facilité de déploiement. -
Robustesse. - Taille. - Besoin en données. - Capacité à reconnaître les
caractères retenus.

Ne pas présumer que CRNN sera nécessairement meilleur : la comparaison
doit décider.

## Livrables

-   Baseline EasyOCR.
-   Modèle CRNN si développé.
-   Rapport comparatif.
-   Poids et configuration OCR retenus.

## Critère de passage

L'OCR doit produire des séquences évaluables et ses limites doivent être
documentées.

------------------------------------------------------------------------

# 10. Phase 7 --- Pipeline complet et post-traitement

## Objectif

Relier détection, crop, OCR et validation dans un service cohérent.

## Tâches détaillées

### 7.1 Développer des modules séparés

-   `detector.py` : charge YOLO et retourne les bounding boxes.
-   `cropper.py` : extrait les plaques.
-   `preprocessor.py` : transforme les crops.
-   `recognizer.py` : exécute l'OCR.
-   `validator.py` : normalise et valide le texte.
-   `inference.py` : orchestre les modules.
-   `schemas.py` : définit les structures de sortie.

### 7.2 Gérer les bounding boxes

-   Vérifier les coordonnées.
-   Les limiter aux dimensions de l'image.
-   Éviter les crops vides.
-   Ajouter une marge contrôlée si nécessaire.
-   Gérer les plaques proches des bords.
-   Conserver les coordonnées originales.

### 7.3 Améliorer le crop

Tester : - Crop direct. - Padding autour de la plaque. - Correction de
perspective. - Rotation / deskew. - Amélioration de contraste. -
Upscaling prudent.

Comparer chaque variante sur un même jeu de validation OCR.

### 7.4 Normaliser le texte

-   Supprimer les espaces parasites.
-   Normaliser les caractères Unicode si nécessaire.
-   Appliquer les règles de format documentées.
-   Conserver la sortie brute OCR pour audit.
-   Conserver séparément la sortie normalisée.
-   Ne pas remplacer automatiquement un caractère ambigu sans
    enregistrer cette correction.

### 7.5 Valider le format

Créer un validateur basé sur les formats réellement inclus dans le
périmètre.

Le validateur doit distinguer : - Texte conforme. - Texte non
conforme. - Texte incomplet. - Texte incertain.

Une validation de format n'est pas une preuve que le texte OCR est
exact.

### 7.6 Définir la réponse complète

Pour chaque plaque : - Bounding box. - Detection confidence. - Texte
brut. - Texte normalisé. - OCR confidence si disponible. - Format
validity. - Temps de traitement. - Statut final.

### 7.7 Tester les cas d'échec

-   Aucune plaque.
-   Plusieurs plaques.
-   Image corrompue.
-   Plaque illisible.
-   OCR vide.
-   OCR non conforme.
-   Erreur de chargement de modèle.

## Livrables

-   Pipeline end-to-end local.
-   Tests unitaires des composants.
-   Tests sur images réelles.
-   Exemples de résultats et d'échecs.

## Critère de passage

Le pipeline doit traiter correctement une image de bout en bout et
retourner une réponse structurée, y compris en cas d'échec.

------------------------------------------------------------------------

# 11. Phase 8 --- Évaluation end-to-end et tests de robustesse

## Objectif

Mesurer le système complet et comprendre les causes d'échec.

## Tâches détaillées

### 8.1 Évaluation de détection

Mesurer Precision, Recall, mAP@50 et mAP@50-95 sur le test de détection.

### 8.2 Évaluation OCR

Mesurer CER et Exact Match sur des crops de référence.

### 8.3 Évaluation end-to-end

Définir une plaque comme correctement reconnue seulement si : - Elle est
détectée selon le seuil de correspondance défini. - Son texte final
correspond exactement à la transcription de référence après
normalisation documentée.

Calculer :

``` text
End-to-End Exact Recognition Rate
= Nombre de plaques entièrement reconnues correctement
  / Nombre total de plaques réelles dans le test
```

Documenter séparément les plaques non détectées, les mauvaises
détections et les erreurs OCR.

### 8.4 Tests par conditions

Créer des sous-ensembles : - Jour. - Nuit. - Faible luminosité. - Pluie
/ reflets. - Flou de mouvement. - Différents angles. - Plaques
éloignées. - Occlusion. - Différentes catégories de plaques.

### 8.5 Tests de performance

Mesurer : - Latence par étape. - Latence end-to-end. - p50 et p95. -
Images par seconde en vidéo. - RAM et VRAM. - Taille des modèles.

### 8.6 Analyser les erreurs

Créer un fichier d'erreurs avec : - Image ID. - Ground truth. -
Prediction. - Erreur de détection ou OCR. - Condition de capture. -
Commentaire. - Action corrective envisagée.

## Livrables

-   Rapport de test final.
-   Tableau comparatif.
-   Galerie d'erreurs.
-   Limites connues.
-   Décision de validation du modèle.

------------------------------------------------------------------------

# 12. Phase 9 --- Tracking des expérimentations avec MLflow

## Objectif

Enregistrer les expériences et retrouver le modèle associé à chaque
résultat.

## Tâches détaillées

### 9.1 Installer et démarrer MLflow

``` bash
pip install mlflow
mlflow server --host 127.0.0.1 --port 5000
```

Configurer le tracking URI côté code.

### 9.2 Définir les expériences

Créer au minimum : - `TLPR_Detection` - `TLPR_OCR` - `TLPR_End_To_End`

### 9.3 Logger chaque entraînement

Paramètres : - Architecture. - Version du code. - Version DVC du
dataset. - Seed. - Hyperparamètres. - Taille des images. - Batch size. -
Epochs. - Matériel.

Métriques : - Détection : Precision, Recall, mAP. - OCR : CER, Exact
Match. - Pipeline : end-to-end exact recognition rate, latency. -
Ressources : durée, mémoire si disponible.

Artefacts : - Poids du modèle. - Configuration. - Courbes. - Matrice de
confusion si pertinente. - Rapport d'évaluation. - Exemples de
prédictions. - Fichier de métadonnées.

### 9.4 Créer une comparaison d'expériences

-   Comparer les runs sur les mêmes données.
-   Filtrer les runs invalides.
-   Identifier le meilleur compromis.
-   Documenter pourquoi un modèle est sélectionné.

### 9.5 Model Registry

Utiliser le registre MLflow si adapté à l'environnement : - Enregistrer
les modèles validés. - Associer les versions et descriptions. - Ajouter
les résultats d'évaluation. - Promouvoir un modèle seulement après
validation. - Garder la possibilité de revenir à une version antérieure.

Si les modèles YOLO et OCR sont gérés séparément, enregistrer leurs
versions indépendamment et créer un manifeste qui identifie la
combinaison utilisée par le pipeline.

## Livrables

-   Serveur MLflow.
-   Expériences de détection et OCR.
-   Historique des runs.
-   Modèles versionnés.
-   Manifeste de la pipeline validée.

------------------------------------------------------------------------

# 13. Phase 10 --- Backend FastAPI

## Objectif

Exposer le pipeline à travers une API utilisable par le frontend ou un
autre client.

## Tâches détaillées

### 10.1 Créer l'application

Organiser : - `main.py` - `routes/` - `services/` - `schemas/` -
`core/` - `dependencies/`

### 10.2 Créer les endpoints

  Endpoint           Méthode   Rôle
  ------------------ --------- -------------------------------
  `/health`          GET       Vérifier l'état de l'API
  `/predict`         POST      Traiter une image
  `/predict/batch`   POST      Traiter plusieurs images
  `/history`         GET       Lire l'historique si activé
  `/models`          GET       Afficher les versions actives

### 10.3 Définir les schémas de réponse

Exemple indicatif :

``` json
{
  "success": true,
  "detections": [
    {
      "plate_text_raw": "OCR_RAW_TEXT",
      "plate_text_normalized": "NORMALIZED_TEXT",
      "detection_confidence": 0.94,
      "ocr_confidence": 0.91,
      "bbox": [120, 200, 350, 270],
      "format_valid": true
    }
  ],
  "processing_time_ms": 180
}
```

Les scores sont illustratifs. Le champ OCR confidence ne doit être
exposé que si le moteur fournit un score dont la signification est
comprise.

### 10.4 Ajouter les validations de sécurité

-   Limiter la taille des uploads.
-   Vérifier le type réel du fichier.
-   Générer des noms de fichiers internes.
-   Ne pas faire confiance au nom ou à l'extension fournis par
    l'utilisateur.
-   Gérer les erreurs sans exposer les chemins internes.
-   Appliquer des limites de requêtes si nécessaire.
-   Ne pas journaliser le contenu des images ou les plaques en clair
    sans besoin et politique explicites.

### 10.5 Ajouter PostgreSQL si nécessaire

Tables possibles : - `inference_requests` - `plate_detections` -
`model_versions`

Stocker uniquement les données nécessaires. Prévoir une politique de
suppression et de rétention.

### 10.6 Tester l'API

-   Tests des endpoints.
-   Tests des fichiers invalides.
-   Tests sans plaque.
-   Tests avec plusieurs plaques.
-   Tests d'erreurs du modèle.
-   Tests de charge simples.
-   Documentation OpenAPI.

## Livrables

-   API FastAPI.
-   Schémas validés.
-   Tests automatisés.
-   Documentation Swagger.
-   Connexion PostgreSQL si activée.

------------------------------------------------------------------------

# 14. Phase 11 --- Frontend React

## Objectif

Créer une interface de démonstration et d'utilisation.

## Tâches détaillées

### 11.1 Écran d'accueil

-   Présentation du projet.
-   Bouton d'import d'image.
-   Zone de dépôt de fichier.
-   Bouton d'analyse.

### 11.2 Écran de résultat

Afficher : - Image originale. - Bounding boxes. - Plaques recadrées. -
Texte OCR. - Scores disponibles. - Statut de validation. - Temps de
traitement.

### 11.3 Gestion des états

Prévoir : - Chargement. - Résultat. - Aucun objet détecté. - Erreur de
fichier. - Erreur serveur. - Reconnaissance incertaine.

### 11.4 Historique et statistiques

Si la conservation est activée : - Historique paginé. - Recherche et
filtres. - Statistiques calculées depuis les données réelles. -
Suppression selon les droits et la politique de conservation.

### 11.5 Caméra en direct --- extension

Après validation du mode image : - Accès caméra avec permission
utilisateur. - Capture de frames à intervalle défini. - Appels à l'API
ou traitement vidéo dédié. - Affichage des résultats. - Limitation du
débit pour éviter la surcharge.

## Livrables

-   Interface React.
-   Connexion API.
-   Écrans de résultats et d'erreurs.
-   Démonstration fonctionnelle.

------------------------------------------------------------------------

# 15. Phase 12 --- Docker et reproductibilité du déploiement

## Objectif

Faire fonctionner l'application de manière cohérente dans un
environnement différent de la machine de développement.

## Tâches détaillées

### 12.1 Dockeriser le backend

-   Choisir une image Python adaptée.
-   Installer les dépendances.
-   Copier le code.
-   Définir les variables d'environnement.
-   Charger les modèles depuis un chemin configurable.
-   Définir la commande de démarrage.
-   Ajouter un healthcheck si pertinent.

### 12.2 Dockeriser le frontend

-   Installer les dépendances Node.
-   Construire les fichiers de production.
-   Servir les fichiers avec un serveur adapté.
-   Configurer l'URL de l'API.

### 12.3 Créer Docker Compose

Services possibles : - `backend` - `frontend` - `postgres` - `mlflow` -
`prometheus` - `grafana`

Ne lancer que les services nécessaires à l'environnement. Le stockage
MLflow et les données PostgreSQL doivent être persistants.

### 12.4 Tester le déploiement local

-   Build des images.
-   Démarrage des services.
-   Vérification des healthchecks.
-   Test de l'API.
-   Test du frontend.
-   Vérification de la persistance.
-   Test après redémarrage.

### 12.5 Gestion GPU

Si l'inférence GPU est nécessaire : - Vérifier le support GPU du runtime
Docker. - Configurer l'accès au GPU. - Vérifier les versions
CUDA/PyTorch. - Prévoir un mode CPU de secours si possible.

## Livrables

-   Dockerfiles.
-   Docker Compose.
-   Configuration documentée.
-   Application démarrée de manière reproductible.

------------------------------------------------------------------------

# 16. Phase 13 --- CI/CD avec GitHub Actions

## Objectif

Automatiser les contrôles qualité à chaque modification du code.

## Tâches détaillées

### 13.1 Workflow CI

À chaque Pull Request : 1. Checkout du dépôt. 2. Installation de Python.
3. Installation des dépendances. 4. Ruff. 5. Tests unitaires. 6. Tests
d'intégration légers. 7. Vérification du build Docker. 8. Rapport de
statut.

### 13.2 Tests du pipeline

Tester avec quelques images de test légères et autorisées : - Chargement
du détecteur. - Chargement de l'OCR. - Format de la réponse. - Cas sans
détection. - Cas d'erreur.

Ne pas lancer un entraînement complet à chaque commit.

### 13.3 Workflow de livraison

Après validation : - Construire l'image Docker. - La taguer avec un
numéro de version ou commit. - La publier dans un registre si
nécessaire. - Déployer dans un environnement de staging. - Exécuter des
tests de santé. - Promouvoir en production après approbation.

### 13.4 Workflow d'entraînement séparé

Créer un workflow manuel ou déclenché par une nouvelle version de
données : - Récupérer le dataset DVC. - Exécuter les validations. -
Lancer l'entraînement sur un runner disposant des ressources
nécessaires. - Logger dans MLflow. - Évaluer le candidat. - Ne pas
promouvoir automatiquement sans seuils et validation définis.

## Livrables

-   Workflow CI.
-   Build automatisé.
-   Workflow de livraison contrôlé.
-   Workflow d'entraînement distinct.

------------------------------------------------------------------------

# 17. Phase 14 --- Monitoring avec Prometheus et Grafana

## Objectif

Observer la santé technique de l'application après déploiement.

## Tâches détaillées

### 14.1 Exposer des métriques depuis FastAPI

Métriques possibles : - Nombre de requêtes. - Requêtes par endpoint. -
Latence. - Nombre d'erreurs. - Taille des uploads. - Durée de chaque
étape du pipeline. - Nombre de plaques détectées. - Nombre de
prédictions OCR vides ou rejetées.

Éviter d'utiliser le texte des plaques comme label de métrique : cela
crée des cardinalités élevées et expose des données sensibles.

### 14.2 Configurer Prometheus

-   Ajouter la cible de scraping.
-   Définir l'intervalle.
-   Vérifier que les métriques sont collectées.
-   Définir des règles d'alerte utiles.

### 14.3 Configurer Grafana

Dashboards : - Requêtes par minute. - Latence p50 / p95. - Taux
d'erreurs. - CPU / RAM. - Utilisation GPU si disponible. - Durée par
étape. - Volume de prédictions.

### 14.4 Distinguer monitoring technique et qualité ML

Prometheus peut observer le service, mais il ne sait pas si une plaque
prédite est correcte sans vérité terrain.

Pour suivre la qualité ML : - Collecter des échantillons de façon
contrôlée. - Faire annoter les échantillons. - Comparer prédictions et
labels. - Calculer les métriques de qualité périodiquement. - Protéger
les données et limiter leur conservation.

### 14.5 Détecter les changements de données

Suivre des indicateurs comme : - Distribution de luminosité. -
Résolution. - Flou. - Distribution des scores. - Proportion de résultats
rejetés. - Conditions de capture.

Un changement de distribution est un signal à investiguer, pas une
preuve automatique de baisse de précision.

## Livrables

-   Endpoint de métriques.
-   Configuration Prometheus.
-   Dashboards Grafana.
-   Alertes pertinentes.
-   Rapport de surveillance.

------------------------------------------------------------------------

# 18. Phase 15 --- Feedback et retraining contrôlé

## Objectif

Améliorer le modèle à partir de nouvelles données correctement annotées.

## Tâches détaillées

### 15.1 Collecter les cas difficiles

Identifier : - Plaques non détectées. - OCR incertain. - Faible
confiance. - Conditions non représentées. - Erreurs remontées par un
utilisateur autorisé.

### 15.2 Faire valider les nouvelles données

-   Vérifier les droits de collecte.
-   Supprimer les données inutiles.
-   Anonymiser ou protéger les données lorsque nécessaire.
-   Faire annoter les images par une personne.
-   Contrôler les annotations.
-   Ne jamais traiter une prédiction du modèle comme une vérité terrain.

### 15.3 Versionner le nouveau dataset

-   Ajouter les nouveaux échantillons.
-   Vérifier les doublons.
-   Prévenir la fuite entre train et test.
-   Mettre à jour DVC.
-   Créer une nouvelle version documentée.

### 15.4 Réentraîner

-   Utiliser la configuration versionnée.
-   Logger le run MLflow.
-   Comparer au modèle actuellement approuvé.
-   Mesurer les performances globales et par catégorie.
-   Vérifier qu'aucune régression importante n'est introduite.

### 15.5 Promouvoir ou rejeter

Promouvoir uniquement si : - Les tests sont réussis. - Les métriques
atteignent les critères définis. - Les sous-groupes importants ne
régressent pas de manière inacceptable. - Le modèle est reproductible. -
La validation humaine ou technique est terminée.

Garder l'ancien modèle disponible pour rollback.

## Livrables

-   Processus de collecte et validation.
-   Nouvelle version DVC.
-   Runs MLflow.
-   Rapport de comparaison.
-   Procédure de promotion et rollback.

------------------------------------------------------------------------

# 19. Phase 16 --- Sécurité, confidentialité et gouvernance

Les images de véhicules et les plaques peuvent être des données
sensibles ou permettre d'identifier des personnes selon le contexte.

À prévoir : - Collecter uniquement les données nécessaires et
autorisées. - Vérifier les licences et autorisations. - Limiter les
accès aux images et résultats. - Chiffrer les transferts et stockages
lorsque nécessaire. - Définir une durée de conservation. - Prévoir une
suppression effective des données. - Ne pas publier de plaques réelles
dans les captures, logs ou dépôts publics sans autorisation. - Protéger
les secrets dans les variables sécurisées de l'environnement ou de la
CI. - Documenter les limites et les usages autorisés. - Éviter toute
utilisation de surveillance ou de suivi de personnes sans base légale et
autorisation appropriée.

------------------------------------------------------------------------

# 20. Phase 17 --- Documentation, démonstration et livraison finale

## Objectif

Présenter un projet reproductible, compréhensible et professionnel.

## Tâches détaillées

### 17.1 README GitHub

Inclure : - Présentation. - Problématique. - Objectifs. -
Architecture. - Stack technique. - Installation. - Configuration. -
Préparation du dataset. - Entraînement YOLO. - Entraînement OCR. -
Évaluation. - Lancement de l'API. - Lancement Docker Compose. -
Monitoring. - Limites. - Références et licences des données.

### 17.2 Documentation technique

Créer : - `docs/architecture.md` - `docs/data_dictionary.md` -
`docs/dataset_card.md` - `docs/model_card_detection.md` -
`docs/model_card_ocr.md` - `docs/experiments.md` - `docs/api.md` -
`docs/deployment.md` - `docs/monitoring.md` -
`docs/known_limitations.md`

### 17.3 Préparer les démonstrations

Scénarios : 1. Une voiture avec une plaque nette. 2. Plusieurs voitures.
3. Une plaque inclinée. 4. Une image de nuit. 5. Une image sans plaque.
6. Une plaque difficile à lire.

Afficher le résultat, les erreurs éventuelles et les métriques réelles.

### 17.4 Préparer la soutenance

Présenter : - Problématique. - État de l'art. - Dataset et
gouvernance. - Architecture YOLO. - Architecture OCR. - Comparaison
expérimentale. - Pipeline complet. - MLOps. - Déploiement. -
Monitoring. - Limites et améliorations futures.

## Livrables finaux

-   Code source organisé.
-   Dataset versionné ou instructions d'accès.
-   Modèles et configurations versionnés.
-   Rapports d'expériences.
-   API et interface.
-   Conteneurs Docker.
-   Workflows CI/CD.
-   Monitoring.
-   Documentation.
-   Démonstration reproductible.

------------------------------------------------------------------------

# 21. Ordre d'exécution recommandé et critères de validation

    Ordre Phase         Résultat obligatoire avant de continuer
  ------- ------------- ----------------------------------------------
        0 Cadrage       Formats, périmètre et métriques définis
        1 Setup         Environnement fonctionnel
        2 Collecte      Données traçables et autorisées
        3 Annotation    Labels vérifiés et splits propres
        4 DVC           Dataset versionné et récupérable
        5 YOLO          Détecteur évalué
        6 OCR           OCR baseline évalué, CRNN si pertinent
        7 Pipeline      Détection et lecture intégrées
        8 Évaluation    Rapport end-to-end
        9 MLflow        Expériences et artefacts traçables
       10 FastAPI       API testée
       11 React         Interface utilisable
       12 Docker        Déploiement local reproductible
       13 CI/CD         Tests et build automatisés
       14 Monitoring    Métriques et dashboards actifs
       15 Retraining    Cycle de mise à jour contrôlé
       16 Gouvernance   Sécurité et politique de données documentées
       17 Livraison     Documentation et démo finale

------------------------------------------------------------------------

# 22. Planning indicatif sur 16 semaines

  Semaines   Travail
  ---------- -------------------------------------------
  1          Cadrage, architecture et setup
  2--3       Recherche, collecte et annotation
  4          Nettoyage, validation, splits et DVC
  5--6       Entraînement et comparaison YOLO
  7--8       OCR baseline et CRNN
  9          Pipeline complet
  10         Évaluation et robustesse
  11         MLflow et versioning des modèles
  12         FastAPI et PostgreSQL
  13         React
  14         Docker et CI/CD
  15         Prometheus, Grafana et retraining
  16         Tests finaux, documentation et soutenance

Ce planning est indicatif. Si la collecte des données tunisiennes ou
l'annotation prend plus de temps, il faut prolonger ces phases plutôt
que de réduire la qualité des données.

------------------------------------------------------------------------

# 23. Priorités : ce qu'il faut faire maintenant

Ne pas commencer par installer tout l'écosystème MLOps. Le projet doit
avancer par livrables validés.

## Première séquence de travail

-   [ ] Confirmer le système d'exploitation, le GPU et la VRAM.
-   [ ] Valider les formats de plaques inclus dans le projet.
-   [ ] Créer le dépôt GitHub.
-   [ ] Créer l'environnement Python.
-   [ ] Créer la structure du projet.
-   [ ] Rédiger le registre des sources de données.
-   [ ] Rechercher des datasets de plaques tunisiennes.
-   [ ] Vérifier les licences et la qualité.
-   [ ] Construire un petit dataset pilote.
-   [ ] Tester l'annotation sur un échantillon.
-   [ ] Valider la faisabilité avant de collecter à grande échelle.

**Premier jalon :** disposer d'un échantillon de données tunisiennes,
annotées et vérifiées, qui permet de lancer un premier entraînement
YOLO.

**Règle de travail :** chaque phase doit produire un livrable, des tests
et une validation avant de passer à la suivante. Les outils MLOps
servent à rendre le système traçable et reproductible ; ils ne
remplacent ni la qualité des données ni l'évaluation scientifique des
modèles.
