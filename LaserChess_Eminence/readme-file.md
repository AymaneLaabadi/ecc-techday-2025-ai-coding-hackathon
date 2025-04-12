# Laser Chess AI

Un système d'intelligence artificielle pour jouer au jeu Laser Chess, basé sur l'algorithme Monte Carlo Tree Search (MCTS) avec apprentissage persistant.

## Structure du Projet

```
LaserChessAI/
├── src/
│   ├── game_state.py        # Moteur de jeu et règles
│   ├── improved_mcts.py     # Implémentation avancée de MCTS 
│   ├── ai_player_learning.py # Système d'IA avec apprentissage
│   ├── self_play_training.py # Système d'auto-entraînement
│   ├── main.py              # Point d'entrée principal
│   ├── ai_knowledge/        # Dossier des connaissances de l'IA
│   └── training_logs/       # Logs d'entraînement
├── docs/
│   └── documentation_technique.md # Documentation détaillée
└── assets/
    └── (ressources graphiques et autres)
```

## Fonctionnalités

- **IA Avancée**: Implémentation personnalisée de Monte Carlo Tree Search avec des améliorations pour le jeu Laser Chess
- **Apprentissage Continu**: L'IA accumule des connaissances au fil du temps grâce à un système de persistance
- **Auto-Entraînement**: Support pour l'entraînement par auto-jeu avec différentes stratégies
- **Entraînement Parallèle**: Accélération de l'apprentissage en utilisant plusieurs processeurs
- **Curriculum d'Apprentissage**: Progression adaptative de la difficulté pour un apprentissage optimal

## Configuration et Exécution

### Prérequis

- Python 3.7 ou supérieur
- NumPy
- Multiprocessing (inclus dans la bibliothèque standard Python)

### Installation

1. Clonez ce dépôt
   ```
   git clone https://github.com/votre-nom/laser-chess-ai.git
   cd laser-chess-ai
   ```

2. (Optionnel) Créez et activez un environnement virtuel
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows, utilisez: venv\Scripts\activate
   ```

3. Installez les dépendances
   ```
   pip install -r requirements.txt
   ```

### Exécution

Pour lancer une partie contre l'IA:
```
python src/main.py play
```

Pour lancer l'entraînement de l'IA:
```
python src/main.py train
```

Options d'entraînement avancées:

- Entraînement standard:
  ```
  python src/self_play_training.py
  ```

- Entraînement accéléré (parallèle):
  ```
  python src/self_play_training.py --accelerated
  ```

- Entraînement par curriculum:
  ```
  python src/self_play_training.py --curriculum
  ```

- Réinitialisation de l'IA:
  ```
  python src/ai_player_learning.py --reset
  ```

## Fonctionnement de l'IA

L'IA utilise Monte Carlo Tree Search (MCTS) pour déterminer les meilleurs coups. Les caractéristiques principales incluent:

1. **Exploration vs Exploitation**: Équilibre entre l'exploration de nouvelles stratégies et l'exploitation des connaissances existantes
2. **Apprentissage Persistant**: L'arbre de recherche est sauvegardé et rechargé entre les sessions
3. **Évaluation Sophistiquée**: Utilisation de critères multiples pour évaluer les positions (matériel, positionnement, proximité du laser au roi adverse)
4. **Détection de Coups Invalides**: Système robuste pour éviter les boucles infinies et les impasses

## Contribution

Les contributions sont les bienvenues! Veuillez consulter les directives de contribution dans le fichier CONTRIBUTING.md.

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
