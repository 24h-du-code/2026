# Hackathon 24h du Code 2026

## Présentation

Bienvenue au hackathon 24h du Code 2026, organisé par Le Mans School of AI ! Cette compétition de 24 heures défie les équipes de développeurs à créer des agents IA capables de jouer et gagner à divers jeux d'apprentissage par renforcement. Les équipes concourent en accumulant des points grâce à des parties réussies sur un serveur de jeu centralisé.

## Objectif

Votre objectif est de créer un ou plusieurs agents IA qui interagissent avec le serveur de jeu via son API REST. L'agent doit jouer aux jeux de manière autonome, prenant des décisions pour gagner autant de parties que possible dans le temps imparti. Les équipes sont classées par leur total de points accumulés.

## Règles

- **Authentification d'équipe** : Chaque équipe reçoit un token unique pour l'accès API. Incluez ce token dans l'en-tête `Authorization: Bearer <token>` pour toutes les requêtes API.
- **Contraintes de jeu** : Les équipes peuvent jouer à plusieurs jeux simultanément mais seulement une instance par type de jeu. Les jeux doivent être joués séquentiellement dans chaque type.
- **Limitation de taux** : Les endpoints de gameplay authentifiés `/api/newgame/`, `/api/get_state/` et `/api/act/` partagent une limite de **1 appel/seconde par équipe et par jeu**. Le polling de `/api/get_state/` compte dans cette limite.
- **Limite de temps** : La compétition dure 24 heures. Les points sont accumulés en temps réel.
- **Jeu équitable** : Les agents doivent respecter les règles du jeu. Toute tentative d'exploitation du serveur ou de contournement des limites entraînera une disqualification.
- **Score** : Les points sont attribués en fonction des résultats des parties (voir section Jeux). Les défaites déduisent des points.

## Jeux

Le serveur héberge 10 jeux différents, chacun avec une complexité et une valeur de points croissantes. Voici une description détaillée de chaque jeu, incluant les actions disponibles, l'état du jeu et le score.

### Jeu n°1 - Tic-Tac-Toe

Jeu classique du morpion sur une grille 3x3. Le joueur (X) doit aligner 3 symboles horizontalement, verticalement ou en diagonale pour gagner. Le serveur joue en tant qu'adversaire (O) avec une IA optimale mais effectuant quelques erreurs pour vous permettre de gagner la partie.

**Actions disponibles :**
- Positions : 00, 01, 02, 10, 11, 12, 20, 21, 22

**État du jeu :**
- Grille 3x3 avec 'X', 'O' ou '' (vide)
- Joueur actuel
- Statut de la partie

**Points :** Victoire = 1 point, Défaite = -1 point, Égalité = 1 point

### Jeu n°2 - Car Racing

Naviguez sur une route à 3 voies pendant 100 pas sans heurter les obstacles.

**Actions disponibles :**
- move_left, move_right, stay

**État du jeu :**
- Position actuelle (0-100)
- Voie actuelle (0, 1, 2)
- Liste des obstacles (position, voie)
- Obstacles visibles (10 prochains pas)

**Points :** Victoire = 2 points, Défaite = -1 point

### Jeu n°3 - Snake

Jeu classique du serpent sur une grille 20x20. Faites grandir le serpent jusqu'à 40 segments en mangeant de la nourriture.

**Actions disponibles :**
- up, down, left, right

**État du jeu :**
- Segments du serpent
- Position de la nourriture
- Score (segments mangés)
- Direction

**Points :** Victoire = 3 points, Défaite = -1 point

### Jeu n°4 - Rush Hour

Puzzle glissant sur une grille 6x6. Déplacez les véhicules pour libérer la voiture rouge (X) vers la sortie droite.
Attention : le générateur de grille peut produire des situations impossibles à résoudre, dans ce cas il faut stopper la partie pour ne pas perdre de points et en démarrer une nouvelle !

**Actions disponibles :**
- move_A_up, move_A_down, etc. (pour chaque véhicule A-Z)

**État du jeu :**
- Liste des véhicules (ID, position, orientation, longueur)
- Position de la sortie
- Pas effectués

**Points :** Victoire = 4 points, Défaite = -1 point

### Jeu n°5 - Adaptive Traffic Racing

Course sur 3 voies avec trafic non stationnaire. Les voitures adverses suivent un style de conduite caché, différent à chaque partie.

**Actions disponibles :**
- left, right, keep, accelerate, brake

**État du jeu :**
- Voie, vitesse, progression
- Véhicules proches
- `observation` + `observation_schema` pour alimenter un agent IA

**Points :** Victoire = 6 à 7 points, Défaite = -2 points, progression partielle en cas de timeout

### Jeu n°6 - Partial Visibility Maze

Labyrinthe à visibilité partielle avec un rayon de vision de 2 cases autour du joueur.

**État du jeu :**
- Grille partiellement visible dans une fenêtre jusqu'à 5x5 centrée sur le joueur (`?` pour inconnu)
- `exit_pos` n'est renvoyée que lorsque la sortie est visible

**Points :** Victoire = 7 points, Défaite = -1 point

### Jeu n°7 - Lava Maze

Labyrinthe avec des tuiles de lave (L) causant une défaite instantanée.

**État du jeu :**
- Grille avec # murs, . passages, L lave, S départ, E sortie

**Points :** Victoire = 8 points, Défaite = -1 point

### Jeu n°8 - Key & Door Maze

Labyrinthe avec une clé (K) à collecter et une porte verrouillée (D).

**État du jeu :**
- Grille avec # murs, . passages, K clé, D porte, S départ
- Statut de possession de la clé

**Points :** Victoire = 9 points, Défaite = -1 point

### Jeu n°9 - Lava + Key & Door Maze

Combine les dangers de lave et la collecte de clé.

**État du jeu :**
- Grille avec tous les éléments des jeux précédents

**Points :** Victoire = 10 points, Défaite = -1 point

### Jeu n°10 - Noisy Moon Lander Lite

Atterrissage lunaire avec bruit de capteurs, vent corrélé et paramètres physiques randomisés par épisode.

**Actions disponibles :**
- idle, main, left, right, main_left, main_right, stabilize

**État du jeu :**
- Position, altitude, vitesses, inclinaison, carburant, plateforme
- `observation` + `observation_schema`

**Points :** Victoire = 10 à 11 points, Défaite = -2 points

## Utilisation de l'API

Le serveur de jeu fournit une API REST pour les interactions de jeu. Toutes les requêtes nécessitent une authentification avec votre token d'équipe dans l'en-tête `Authorization: Bearer <token>`. Le token n'est pas accepté en query parameter.
Si les organisateurs suspendent temporairement les API equipes depuis l'administration, tous les endpoints authentifies repondent en `503 Service Unavailable` avec un message d'indisponibilite temporaire.

### Endpoints

#### GET /list_games
Liste les jeux disponibles.

**Réponse :**
```json
[
  {"id": 1, "name": "Tic-Tac-Toe"},
  ...
]
```

#### POST /newgame
Démarre une nouvelle session de jeu.

**Requête :**
```json
{"idgame": 1}
```

**Réponse :**
```json
{
  "gamesessionid": 123,
  "action_list": {
    "00": "Place piece at position (0, 0)",
    "01": "Place piece at position (0, 1)"
  }
}
```

#### GET /get_state
Récupère l'état actuel du jeu.

**Query :** ?gamesessionid=session_id

**Réponse :**
```json
{
  "state": { /* état spécifique au jeu */ }
}
```

Pour les jeux 5 et 10, `state` inclut en plus un vecteur `observation` et son `observation_schema`.

#### POST /act
Effectue une action dans le jeu.

**Requête :**
```json
{
  "gamesessionid": "session_id",
  "action": "00"
}
```

**Réponse :**
```json
{
  "state": { /* état mis à jour */ },
  "status": "continue", // ou "win", "lose", "tie", "max_steps"
  "remaining_steps": 95
}
```

#### POST /stop_game
Annule une session active.

**Requête :**
```json
{"gamesessionid": "session_id"}
```

**Réponse :**
```json
{"message": "Partie arrêtée"}
```

## Comment aborder les jeux IA (5 et 10)

Pour `Adaptive Traffic Racing` et `Noisy Moon Lander Lite`, la bonne approche est la suivante :

1. entraînez votre agent localement avec le simulateur fourni dans `hackathon_rl_envs`
2. utilisez l'API distante uniquement pour jouer et scorer
3. utilisez `state.observation` comme entrée standard de votre modèle
4. utilisez `state.observation_schema` comme source de vérité pour l'ordre des features

Les champs lisibles de `state` restent utiles pour le debug et la visualisation, mais l'input recommandé du modèle est `state.observation`.

### Boucle minimale côté client

```python
start = client.start_game(game_id)
session_id = start["gamesessionid"]

while True:
    payload = client.get_state(session_id)
    state = payload["state"]
    observation = state["observation"]
    action = policy.predict(observation)
    result = client.act(session_id, action)
    if result["status"] != "continue":
        break
```

### Jeu 5 : input exact

Jeu : `Adaptive Traffic Racing`

- entrée modèle : `state.observation`
- taille du vecteur : `110`
- ordre des features : `state.observation_schema`
- actions valides :
  - `left`
  - `right`
  - `keep`
  - `accelerate`
  - `brake`

Structure du vecteur :

- `lane_norm`
- `speed_norm`
- puis 4 blocs locaux :
  - `occ_t0_*` : occupation locale actuelle
  - `rel_t0_*` : vitesse relative locale actuelle
  - `occ_t1_*` : occupation au pas précédent
  - `occ_t2_*` : occupation deux pas plus tôt

En pratique, le schéma complet est fourni par l'API. Il est préférable de lire `observation_schema` plutôt que de hardcoder l'ordre des colonnes.

### Jeu 10 : input exact

Jeu : `Noisy Moon Lander Lite`

- entrée modèle : `state.observation`
- taille du vecteur : `10`
- ordre des features : `state.observation_schema`
- actions valides :
  - `idle`
  - `main`
  - `left`
  - `right`
  - `main_left`
  - `main_right`
  - `stabilize`

Features exposées :

- `altitude`
- `vx`
- `vy`
- `dx_pad`
- `fuel_fraction`
- `sin_theta`
- `cos_theta`
- `omega`
- `leg_contact_left`
- `leg_contact_right`

### Entraînement local recommandé

```python
from hackathon_rl_envs import make_env

env = make_env("NoisyMoonLanderLite-v0")
obs, info = env.reset(seed=0)
```

ou :

```python
from hackathon_rl_envs import make_env

env = make_env("AdaptiveTrafficRacing-v0")
obs, info = env.reset(seed=0)
```

Vous pouvez ensuite entraîner n'importe quelle politique discrète qui prend `obs` en entrée et renvoie une action valide.

### Ce qui est fourni, et ce qui ne l'est pas

Pour les jeux 5 et 10, les organisateurs fournissent les outils de travail, pas la solution finale.

**Fourni aux participants :**

- le contrat API
- le package local `hackathon_rl_envs`
- les scripts d'entraînement et de benchmark du dépôt
- des agents d'exemple simples pour montrer comment brancher un client

**Non fourni aux participants :**

- les seeds d'évaluation serveur
- les benchmarks organisateur internes
- les checkpoints entraînés par les organisateurs
- les baselines fortes utilisées pour valider l'équilibrage des jeux

En particulier, si vous utilisez le pipeline PPO ci-dessous, vous devez entraîner votre propre modèle. Aucun poids préentraîné n'est distribué aux équipes.

Pour `Noisy Moon Lander Lite`, un pipeline PPO optionnel avec `stable-baselines3` est aussi fourni :

1. installez les dépendances optionnelles :
   `pip install .[rl-sb3]`
2. entraînez localement :
   `PYTHONPATH=. python scripts/train_sb3_noisy_moonlander_lite.py --total-timesteps 250000`
3. récupérez votre meilleur checkpoint local :
   `artifacts/noisy_moonlander_ppo/best_model.zip`
4. utilisez ensuite ce checkpoint dans le client API dédié :
   `PYTHONPATH=. python players/noisy_moonlander_lite_sb3_agent.py --api-key <TOKEN> --model-path artifacts/noisy_moonlander_ppo/best_model.zip`

Le script PPO écrit aussi un fichier `training_summary.json` avec :

- les hyperparamètres PPO
- un warm start supervisé sur l'heuristique du lander
- les statistiques de normalisation des observations
- une évaluation offline sur des seeds fixes

Le client API SB3 recharge automatiquement ces statistiques si `training_summary.json` est présent à côté du checkpoint.

### Important

- Les checkpoints générés dans `artifacts/` sont vos sorties locales d'entraînement. Ils ne constituent pas un artefact officiel fourni par les organisateurs.
- N'entraînez pas massivement via l'API distante : le rate limiting en fera un mauvais pipeline de données.
- Les seeds d'évaluation serveur sont cachées.
- Les paramètres internes du simulateur ne font pas partie des inputs du modèle.
- Si vous faites du prétraitement, alignez-vous toujours sur `observation_schema`.

## Score et Classement

- Les points sont attribués immédiatement à la fin de la partie.
- Le classement est mis à jour en temps réel.
- Le score total détermine le classement.
- Interface web disponible pour surveiller les progrès et le classement.

## Pour Commencer

1. Obtenez votre token d'équipe auprès des organisateurs.
2. Familiarisez-vous avec l'API en utilisant le client de démonstration.
3. Pour les jeux 5 et 10, entraînez d'abord localement avec `hackathon_rl_envs`.
4. Branchez ensuite votre politique dans un client API.
5. Testez ce client sur le serveur.
6. Concourez pendant l'événement de 24 heures !

## Ressources Supplémentaires

- Client de démonstration : `streamlit run demo_client_streamlit.py`
- Environnements locaux IA : `hackathon_rl_envs`
- URL du serveur : Fournie par les organisateurs
- Surveillance web : Disponible à l'URL du serveur

Bonne chance, et que la meilleure IA gagne !
