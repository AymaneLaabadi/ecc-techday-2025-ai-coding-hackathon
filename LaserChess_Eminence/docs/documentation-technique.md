# Documentation Technique - Laser Chess AI

## 1. Architecture du Système

Le projet Laser Chess AI est structuré autour de plusieurs composants clés qui travaillent ensemble pour créer et entraîner une intelligence artificielle capable de jouer au jeu Laser Chess.

### 1.1 Structure Globale

L'architecture du système est divisée en trois composants principaux:

1. **Moteur de jeu** (`game_state.py`)
   - Définit les règles et la logique du jeu Laser Chess
   - Gère l'état du jeu, les actions valides, et détermine quand une partie est terminée

2. **Système d'IA** (`improved_mcts.py`, `ai_player_learning.py`)
   - Implémente l'algorithme Monte Carlo Tree Search (MCTS) pour la prise de décision
   - Gère l'apprentissage et la persistance des connaissances de l'IA

3. **Système d'entraînement** (`self_play_training.py`)
   - Orchestre les sessions d'auto-apprentissage où l'IA joue contre elle-même
   - Supporte l'entraînement parallèle pour accélérer l'apprentissage

### 1.2 Flux de Données

Le flux de données à travers le système suit généralement ce schéma:

```
État du jeu → Adaptation MCTS → Recherche MCTS → Sélection d'action → Application de l'action → Nouvel état du jeu
```

Les connaissances acquises sont persistées sur disque pour permettre un apprentissage continu sur plusieurs sessions.

## 2. Choix des Algorithmes d'IA

### 2.1 Algorithme Principal: Monte Carlo Tree Search (MCTS)

Le MCTS a été choisi comme algorithme principal pour plusieurs raisons:

- **Absence de fonction d'évaluation prédéfinie**: Contrairement aux algorithmes minimax traditionnels, MCTS ne nécessite pas une fonction d'évaluation complexe prédéfinie pour le jeu.
- **Apprentissage automatique**: MCTS apprend par l'expérience, améliorant sa performance au fil du temps.
- **Exploration vs Exploitation**: L'algorithme équilibre naturellement l'exploration de nouvelles stratégies et l'exploitation des connaissances existantes.
- **Adaptabilité**: MCTS s'adapte bien aux jeux avec un grand espace d'états comme Laser Chess.

### 2.2 Améliorations de l'Algorithme MCTS

Plusieurs améliorations ont été apportées à l'implémentation standard de MCTS:

#### 2.2.1 MCTS Persistant

Une innovation clé est le développement d'un MCTS persistant, qui permet:
- La sauvegarde et le chargement de l'arbre de recherche entre les sessions
- L'accumulation de connaissances sur de longues périodes d'entraînement
- Le transfert de connaissances entre différentes parties

#### 2.2.2 Récompenses Intermédiaires

Le système utilise des récompenses intermédiaires sophistiquées basées sur:
- L'avantage matériel (nombre de pièces)
- La position des pièces clés comme le roi et le laser
- La proximité du laser par rapport au roi adverse
- L'alignement directionnel du laser

#### 2.2.3 Élargissement Progressif (Progressive Widening)

Pour gérer efficacement le grand facteur de branchement dans Laser Chess:
- Limite le nombre d'actions considérées en fonction du nombre de visites du nœud
- Améliore l'efficacité de l'exploration dans des espaces d'action larges

#### 2.2.4 Détection et Pénalisation des Coups Invalides

Le système dispose d'un mécanisme robuste pour:
- Détecter les coups qui ne font pas progresser le jeu
- Pénaliser ces coups dans l'arbre de recherche
- Éviter les boucles infinies et les impasses

## 3. Implémentation Détaillée

### 3.1 Classe LaserChessAdapter

Interface entre l'état du jeu et l'algorithme MCTS:
- Traduit l'état du jeu en format utilisable par MCTS
- Fournit des méthodes pour obtenir les actions légales, vérifier les états terminaux, et calculer les récompenses
- Calcule un hachage unique pour chaque état du jeu

### 3.2 Classe PersistentMCTS

Implémentation cœur de l'algorithme MCTS amélioré:

#### 3.2.1 Structure de l'Arbre

Chaque nœud dans l'arbre contient:
- `N`: Nombre de visites
- `Q`: Valeur totale accumulée
- `children`: Dictionnaire des nœuds enfants
- `actions`: Liste des actions légales
- `player`: Joueur actuel
- `terminal`: Indicateur d'état terminal
- `reward`: Récompense (pour états terminaux)
- `state`: État complet du jeu
- `parent`: Référence au nœud parent
- `depth`: Profondeur dans l'arbre
- `valid_visit_count`: Compteur de simulations valides

#### 3.2.2 Principales Méthodes

- `search()`: Lance la recherche MCTS pour trouver la meilleure action
- `_select()`: Sélectionne un chemin dans l'arbre jusqu'à un nœud feuille
- `_expand()`: Étend l'arbre en ajoutant un nouveau nœud
- `_rollout()`: Effectue une simulation à partir d'un nœud
- `_backpropagate()`: Propage les résultats de simulation vers le haut de l'arbre
- `update_after_move()`: Met à jour l'arbre après qu'un coup a été joué

### 3.3 Gestion de l'Apprentissage

Le module `ai_player_learning.py` gère:
- La persistance de l'arbre MCTS sur disque
- Les statistiques d'apprentissage (parties jouées, victoires, etc.)
- L'ajout de "livres d'ouverture" pour guider l'apprentissage initial
- L'adaptation des paramètres en fonction de la phase de jeu

### 3.4 Stratégies d'Entraînement

Le module `self_play_training.py` implémente:
- L'entraînement par auto-jeu (l'IA joue contre elle-même)
- L'apprentissage par curriculum (difficulté progressive)
- L'entraînement parallèle pour accélérer l'acquisition de connaissances
- La journalisation des résultats pour analyse ultérieure

## 4. Défis Techniques et Solutions

### 4.1 Gestion de la Mémoire

**Défi**: L'arbre MCTS peut croître excessivement avec le temps.

**Solution**:
- Élagage périodique des nœuds peu visités
- Conservation prioritaire des nœuds proches de la racine
- Filtrage des actions invalides pour réduire la taille de l'arbre

### 4.2 Entraînement Efficace

**Défi**: L'entraînement par auto-jeu peut être lent.

**Solution**:
- Parallélisation de l'entraînement sur plusieurs processus
- Curriculum d'apprentissage adaptatif
- Sauvegarde et chargement périodiques pour la résilience

### 4.3 Qualité des Décisions

**Défi**: Trouver un équilibre entre vitesse et qualité de décision.

**Solution**:
- Adaptation dynamique du temps de réflexion selon la phase de jeu
- Réglage des paramètres d'exploration en fonction des performances
- Utilisation de connaissances prédéfinies (livres d'ouverture) pour guider l'apprentissage initial

## 5. Perspectives d'Amélioration

### 5.1 Améliorations Algorithmiques

- Intégration de réseaux de neurones pour guider la recherche (AlphaZero-style)
- Perfectionnement des fonctions de récompense intermédiaire
- Optimisation de la détection des motifs de jeu récurrents

### 5.2 Optimisations Techniques

- Implémentation plus efficace en mémoire de l'arbre MCTS
- Utilisation de GPU pour accélérer les simulations
- Compression des données de l'arbre pour réduire l'empreinte de stockage

### 5.3 Améliorations de l'Entraînement

- Mise en place d'un système d'adversaires multiples avec différents styles de jeu
- Génération automatique de puzzles pour cibler des compétences spécifiques
- Apprentissage par imitation à partir de parties humaines de haut niveau
