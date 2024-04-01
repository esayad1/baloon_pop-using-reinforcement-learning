import os 
import sys
import time
import random
import pygame

from enum import Enum
from torch.utils.tensorboard import SummaryWriter

from ai.QLearning import QLearning # Q Learning
from ai.DQN import DQN # Deep Q Learning
from ai.DDQN import DDQN # Double Deep Q Learning
from ai.DDQNWER import DDQNWER # Double Deep Q Learning with Experience Replay
from ai.DDQNWPER import DDQNWPER # Double Deep Q Learning with Prioritized Experience Replay
from ai.Reinforce import Reinforce # Algorithme REINFORCE
from ai.ReinforceMeanBaseline import ReinforceMeanBaseline # Algorithme REINFORCE avec une ligne de base
from ai.ReinforceBaselineCritic import ReinforceBaselineCritic # Algorithme REINFORCE avec un critique

# Initialisation de pygame
pygame.init()
pygame.display.set_caption("Can't Stop")

########### CLASS ###########

class Colors(Enum):
    """
    Couleurs disponibles pour les joueurs.
    """
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)


class Dices:
    """
    Représente les dés du jeu.
    """
    
    class Dice:
        """
        Représente un dé.
        """
        
        def __init__(self) -> None:
            """
            Initialisation de la classe Dice.
            """
            self.value = 0
            
        def roll(self) -> None:
            """
            Fait rouler le dé.
            """
            self.value = random.randint(1, 6)
            
        def __str__(self) -> str:
            """
            Representation en chaîne de caractères du dé.
            """
            return str(self.value)
        
    def __init__(self) -> None:
        """
        Constructeur de la classe Dices.
        """
        self.dices = [self.Dice() for _ in range(4)]
        self.roll() # On lance les dés une première fois pour avoir des valeurs lors de l'initialisation.
        
    def roll(self) -> None:
        """
        Relance les dés.
        """
        for dice in self.dices:
            dice.roll()
            
    def get_combinations(self) -> list:
        """
        Retourne les combinaisons possibles pour les dés.
        """
        combinations = set()

        for i in range(len(self.dices)): 
            for j in range(i + 1, len(self.dices)):
                combinations.add(self.dices[i].value + self.dices[j].value)
        
        return list(combinations)
    
    def __str__(self) -> str:
        """
        Representation en chaîne de caractères des dés.
        """
        return " ".join([str(dice) for dice in self.dices])
    
class Player:
    """
    Représente un joueur.
    """
    
    def __init__(self, name, color) -> None:
        """
        Constructeur de la classe Player.
        Args:
            name (str): Nom du joueur.
            color (Colors): Couleur du joueur.
        """
        self.name = name       
        self.color = color
        
        self.checkpoints = {
            '2':  0,   # 2  (1, 2)
            '3':  0,   # 3  (1, 2, 3, 4)
            '4':  0,   # 4  (1, 2, 3, 4, 5, 6)
            '5':  0,   # 5  (1, 2, 3, 4, 5, 6, 7, 8)
            '6':  0,   # 6  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            '7':  0,   # 7  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            '8':  0,   # 8  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            '9':  0,   # 9  (1, 2, 3, 4, 5, 6, 7, 8)
            '10': 0,  # 10 (1, 2, 3, 4, 5, 6)
            '11': 0,  # 11 (1, 2, 3, 4)
            '12': 0   # 12 (1, 2)
        }   
        
        self.checkpoints_columns = []
        
        self.checkpoints_columns_finished = []
        
        self.checkpoints_tmp = self.checkpoints.copy()

        self.checkpoints_columns_tmp = self.checkpoints_columns.copy()
        
    def reset(self) -> None:
        """
        Réinitialise les checkpoints du joueur.
        """
        self.checkpoints = {
            '2':  0,   # 2  (1, 2)
            '3':  0,   # 3  (1, 2, 3, 4)
            '4':  0,   # 4  (1, 2, 3, 4, 5, 6)
            '5':  0,   # 5  (1, 2, 3, 4, 5, 6, 7, 8)
            '6':  0,   # 6  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            '7':  0,   # 7  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            '8':  0,   # 8  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            '9':  0,   # 9  (1, 2, 3, 4, 5, 6, 7, 8)
            '10': 0,  # 10  (1, 2, 3, 4, 5, 6)
            '11': 0,  # 11  (1, 2, 3, 4)
            '12': 0   # 12  (1, 2)
        }
        
        self.checkpoints_columns = []
        self.checkpoints_columns_finished = []
        
        self.checkpoints_tmp = self.checkpoints.copy()
        self.checkpoints_columns_tmp = self.checkpoints_columns.copy()
        
        
    def reset_checkpoints(self) -> None:
        """
        Réinitialise les checkpoints sauvegardés.
        """
        self.checkpoints = self.checkpoints_tmp.copy()
        self.checkpoints_columns = self.checkpoints_columns_tmp.copy()
        
    def save_checkpoints(self) -> None:
        """
        S
        """
        self.checkpoints_tmp = self.checkpoints.copy()
        self.checkpoints_columns_tmp = self.checkpoints_columns.copy()
        
    def __str__(self) -> str:
        return self.name

########### FONCTIONS POUR LES ALGORITHMES ###########

def get_state() -> list:
    """
    Retourne l'état actuel du jeu.
    """
    global current_step, current_player, opponent_player
    
    state = []
    # Les trois colonnes courantes du joueur actif (initialisation avec 3 colonnes vides, on ne peut pas avoir plus de 3 colonnes)
    state.extend([0, 0, 0])

    # 0 = Aucune colonne en cours, autre = Numéro de la colonne en cours    
    for col in current_player.checkpoints_columns:
        if col != 0:
            # On retire l'élement au début de la liste
            state.pop(0)
            
            # On ajoute à la fin de la liste
            state.append(col)
    
    # On fait la même chose pour les colonnes du joueur inactif
    state.extend([0, 0, 0])
    
    for col in opponent_player.checkpoints_columns:
        if col != 0:
            state.pop(3)
            state.append(col)
    
    # Ensuite on met la position de chaque pion sur les colonnes du joueur actif
    for col in state[:3]:
        if col == 0:
            state.append(0)
        else:
            state.append(current_player.checkpoints[str(col)])

    # Ensuite on met la position de chaque pion sur les colonnes du joueur inactif
    for col in state[3:6]:
        if col == 0:
            state.append(0)
        else:
            state.append(opponent_player.checkpoints[str(col)])
    
    # Pour finir on ajoute les valeurs des dés
    state.extend(dice.value for dice in dices.dices)
    
    # Puis on ajoute le numéro de tour actuel pour savoir si l'agent doit sauvegarder ses checkpoints
    state.append(int(current_step%3 == 0))
    
    return state

def get_reward(action):
    """
    Retourne la récompense pour une action donnée dans l'état actuel du jeu.
    """
    global current_step, dices_rolls, debug
    
    # On insite l'agent à sauvegarder ses checkpoints tous les 3 tours (pour éviter de les perdre, on joue sécurisé)
    if (current_step - 1)%3 == 0:
        if action == 1:
            if debug:
                print(f"Récompense: 1 | L'agent sauvegarde ses checkpoints")
            return 1
        else:
            if debug:
                print(f"Récompense: -10 | L'agent n'a pas sauvegardé ses checkpoints")
            # On met une grosse pénalité si l'agent ne sauvegarde pas ses checkpoints
            return -10

    # Si nous ne somme pas dans un tour multiple de 3, on met une grosse pénalité si l'agent sauvegarde ses checkpoints pour le forcer à avancer
    elif action == 1:
        if debug:
            print(f"Récompense: -10 | L'agent sauvegarde ses checkpoints")
        return -10

    # On met une grosse pénalité si l'agent avance sur une colonne qui n'est pas dans les combinaisons possibles (action non valide)
    elif len(dices_rolls) > 1 and action not in dices_rolls[-2].get_combinations():
        if debug:
            print(f"Récompense: -10 | L'agent avance sur une colonne non valide")
        return -10

    # Sinon on met une récompense si l'agent avance
    else:
        if debug:
            print(f"Récompense: 1 | L'agent avance sur une colonne")
        return 1

########### FONCTIONS DE DEBUG ###########
def display_debug():
    """
    Affiche les informations de debug.
    """
    global current_step, current_episode, current_player, opponent_player, dices, dices_rolls, board, player_one, player_two, nb_victories, agent

    print(f"_________________ {current_episode} - {current_step} __________________")
    print(f"Joueur actif: {current_player.name}")
    print(f"Joueur inactif: {opponent_player.name}")
    print(f"J1 Checkpoints: {player_one.checkpoints}")
    print(f"J2 Checkpoints: {player_two.checkpoints}")
    print(f"J1 Checkpoints Columns: {player_one.checkpoints_columns}")
    print(f"J2 Checkpoints Columns: {player_two.checkpoints_columns}")
    print(f"J1 Checkpoints Columns Finished: {player_one.checkpoints_columns_finished}")
    print(f"J2 Checkpoints Columns Finished: {player_two.checkpoints_columns_finished}")
    print(f"Nb Victories: {nb_victories}")
    print(f"Dices: {dices}")
    print(f"Dices Rolls Combinations: {dices.get_combinations()}")
    print(f"Board: {board}")
    if agent:
        print(f"___AI___")
        print(f"Loss: {agent.loss}")
        print(f"State: {get_state()}")
    print("_________________________________________________________")

########### FONCTIONS POUR LE JEU ###########

def reset():
    """
    Réinitialise le jeu.
    """
    global current_step, current_episode, current_player, opponent_player, dices, dices_rolls, board, player_one, player_two, nb_victories, loss
    
    # Réinitialisation des joueurs
    player_one.reset()
    player_two.reset()
    
    # Réinitialisation du joueur actif
    current_player = player_one
    
    # Réinitialisation du joueur inactif
    opponent_player = player_two
    
    # Réinitialisation des dés
    dices = Dices()
    
    # Réinitialisation des lancés de dés
    dices_rolls = []
    
    # Réinitialisation du plateau de jeu
    board = {
        '2': '',
        '3': '',
        '4': '',
        '5': '',
        '6': '',
        '7': '',
        '8': '',
        '9': '',
        '10': '',
        '11': '',
        '12': '',
    }
    
    # Réinitialisation du nombre de tours
    current_step = 1
    
    # Réinitialisation du nombre d'épisodes
    current_episode += 1
    
  
def is_over():
    """
    Retourne si la partie est terminée.
    """
    # Si aucune case n'est vide dans le plateau de jeu
    if '' not in board.values():        
        return True
    return False
    
def save_current_player_checkpoints():
    """
    Sauvegarde les checkpoints du joueur actif.
    """
    current_player.save_checkpoints()
    
def reset_current_player_checkpoints():
    """
    Réinitialise les checkpoints du joueur actif.
    """
    current_player.reset_checkpoints()

def switch_players():
    """
    Change le joueur actif et le joueur inactif.
    """
    global current_player, opponent_player, current_player_previous_actions
    
    # Switch des joueurs
    current_player, opponent_player = opponent_player, current_player
    
    # Réinitialisation des actions précédentes du joueur actif
    current_player_previous_actions = []
    
def get_winner():
    """
    Retourne le joueur gagnant.
    """
    # Il faut compter le nombre de colonnes terminées pour chaque joueur
    count_player_one = len([column for column in board.values() if column == 'J1'])
    count_player_two = len([column for column in board.values() if column == 'J2'])

    # Le joueur avec le plus de colonnes terminées gagne
    if count_player_one > count_player_two:
        return player_one
    
    elif count_player_one < count_player_two:
        return player_two
    
    else:
        return None
        
def get_current_player_possible_actions():
    """
    Retourne les actions possibles pour le joueur actif.
    """
    global dices
        
    # On récupère les combinaisons possibles pour le joueur actif
    possible_actions = dices.get_combinations()
        
    # On ajoute la sauvegarde des checkpoints
    possible_actions.append(1)
        
    return possible_actions

def get_current_player_random_action():
    """
    Retourne une action aléatoire parmis les actions possibles pour le joueur actif.
    """
    return random.choice(get_current_player_possible_actions())

def step(action):
    """
    Réalise une action pour le joueur actif.
    """
    global current_step, dices, dices_rolls, board, current_player, opponent_player, debug
    
    # Convrersion de l'action en string
    action = str(action)
    
    # On ajoute l'action à la liste des actions précédentes du joueur actif
    current_player_previous_actions.append(action)
    
    # On ajoute les dés lancés à la liste des lancés de dés
    dices_rolls.append(dices)
    
    # Si l'action est de sauvegarder les checkpoints
    if action == '1':
        # Si le debug est activé
        if debug:
            print(f"{current_player.name} sauvegarde ses checkpoints")
        
        # Sauvegarde des checkpoints du joueur actif
        save_current_player_checkpoints()
        
        # On change de joueur
        switch_players()
    
    # Si l'on souhaite joué une colonne    
    else:
       
       # Si le joueur à plus de 3 colonnes en cours
        if len(current_player.checkpoints_columns) == 3:
            print(f"{current_player.name} a déjà 3 colonnes en cours")
            # Si l'action qu'il souhaite joué n'est pas dans la liste de ses colonnes en cours
            if action not in current_player.checkpoints_columns:
                print(f"{current_player.name} ne peut pas jouer cette colonne")
                return
       
        if action not in current_player.checkpoints_columns and action not in current_player.checkpoints_columns_finished:
            current_player.checkpoints_columns.append(action)

        if action in current_player.checkpoints_columns and current_player.checkpoints[action] < board_limits[action]:
            current_player.checkpoints[action] += 1
            
        if current_player.checkpoints[action] == board_limits[action] and not action in current_player.checkpoints_columns_finished:
            # On retire la colonne terminée des colonnes du joueur actif
            current_player.checkpoints_columns.remove(action)
            current_player.checkpoints_columns_finished.append(action)
            
            # On retire la colonne terminée des colonnes du joueur inactif
            opponent_player.checkpoints_columns_finished.append(action)
            
            # Si il été en train de jouer cette colonne, on la retire
            if action in opponent_player.checkpoints_columns:
                opponent_player.checkpoints_columns.remove(action)
                
            # On déplace le pion du joueur actif sur la colonne terminée
            board[action] = current_player.name
            
            # On sauvagarde les checkpoints du joueur actif
            save_current_player_checkpoints()
            
    # On incrémente le nombre de tours
    current_step += 1
    
    # On relance les dés
    dices.roll()

########### FONCTIONS POUR DESSINER LE JEU ###########

def draw():
    """
    Dessin sur l'écran.
    """
    global current_player, dices, board, board_limits
    
    screen.fill((255, 255, 255))
    
    for i in range(2, 13):  
        for j in range(0, board_limits[str(i)]):
            pygame.draw.rect(screen, (0, 0, 0), (i * 50, 750 - j * 50, 50, 50), 2)
            
        color = Colors.RED if board[str(i)] == 'J1' else Colors.BLUE
        
        if board[str(i)] == '':
            text = font.render(str(i), True, (0, 0, 0))
            screen.blit(text, (i * 50 + 20, 760 - board_limits[str(i)] * 50))
        else:
            text = font.render(board[str(i)], True, color.value)
            screen.blit(text, (i * 50 + 20, 760 - board_limits[str(i)] * 50))
        
    pygame.draw.rect(screen, (0, 0, 0), (50, 150, 270, 50), 2)
        
    # On ajoute un texte pour savoir si le joueur peut sauvegarder ses checkpoints
    text = font.render("Sauvegarder les checkpoints", True, (0, 0, 0))
    screen.blit(text, (55, 155))
        
    text = font.render(f"Dés : {dices.get_combinations()}", True, (0, 0, 0))
    screen.blit(text, (50, 250))
        
    text = font.render(f"Joueur actuel : {current_player}", True, (0, 0, 0))
    screen.blit(text, (500, 100))
        
    text = font.render(f"Nombre d'épisodes : {current_episode}", True, (0, 0, 0))
    screen.blit(text, (500, 150))
            
    text = font.render(f"Nombre de pas : {current_step}", True, (0, 0, 0))
    screen.blit(text, (500, 200))
        
    text = font.render(f"Nombre de victoires J1 : {nb_victories['J1']}", True, (0, 0, 0))
    screen.blit(text, (500, 250))
        
    text = font.render(f"Nombre de victoires J2 : {nb_victories['J2']}", True, (0, 0, 0))
    screen.blit(text, (500, 300))
        
    for column in current_player.checkpoints_columns:
        pygame.draw.circle(screen, current_player.color.value, (int(column) * 50 + 25, 800 - (current_player.checkpoints[str(column)] - 1) * 50 - 25), 15)

    pygame.display.flip()

########### VARIABLES GLOBALES ###########

# Initialisation des joueurs
player_one = Player("J1", Colors.RED)
player_two = Player("J2", Colors.BLUE)

# Initialisation du joueur actif
current_player = player_one

# Initialisation du joueur inactif
opponent_player = player_two

# Initialisation du liste contenant les actions précédentes du joueur actif
current_player_previous_actions = []

# Initialisation des dés
dices = Dices()

# Initialisation d'une liste sauvegardant tous les lancés de dés
dices_rolls = []

# Initialisation des variables utiles
current_episode = 1
current_step = 1

# Initialisation du nombre de victoires
nb_victories= {
    player_one.name: 0,
    player_two.name: 0
}

# Initialisation du plateau de jeu ('' = case vide, 'J1' = colonne du joueur 1, 'J2' = colonne du joueur 2)
board = {
    '2': '',
    '3': '',
    '4': '',
    '5': '',
    '6': '',
    '7': '',
    '8': '',
    '9': '',
    '10': '',
    '11': '',
    '12': '',
}

# Limite du plateau (nombre de cases maximum pour chaque colonne)
board_limits = {
    '2' : 2,
    '3' : 4,
    '4' : 6,
    '5' : 8,
    '6' : 10,
    '7' : 12,
    '8' : 10,
    '9' : 8,
    '10': 6,
    '11': 4,
    '12': 2
}

# Actions possibles (1 = Sauvegarder les checkpoints, Reste = Avancer sur la colonne correspondante au nombre de l'action)
actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
actions_for_algorithm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Le mode débug est activé
debug = False

# Initialisation des hyperparamètres pour l'entraînement
lr = 0.1 # Taux d'apprentissage (Il permet de déterminer dans quelle mesure les nouveaux apprentissages doivent être pris en compte par rapport aux anciens)
gamma = 0.9 # Facteur de réduction (Il permet de déterminer dans quelle mesure les récompenses futures doivent être prises en compte par rapport aux récompenses immédiates)
epsilon = 0.2 # Taux d'exploration (Il permet de déterminer dans quelle mesure l'agent doit explorer de nouvelles actions plutôt que de suivre les actions déjà apprises)

input_dim = len(get_state()) # Dimension de l'entrée
output_dim = 12 # Dimension de la sortie

# Sauvegarde de la loss
loss = 0

# Initialisation du nom de l'algorithme utilisé
algorithm_name = ""

# Initialisation de l'agent
agent = None

# Initialisation de l'algorithme utilisé
if algorithm_name == "QLearning":
    agent = QLearning(lr, gamma, epsilon, actions_for_algorithm)
    
elif algorithm_name == "DQN":
    agent = DQN(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "DDQN":
    agent = DDQN(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "DDQNWER":
    agent = DDQNWER(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "DDQNWPER":
    agent = DDQNWPER(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "Reinforce":
    agent = Reinforce(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "ReinforceMeanBaseline":
    agent = ReinforceMeanBaseline(lr, gamma, epsilon, input_dim, actions_for_algorithm)
    
elif algorithm_name == "ReinforceBaselineCritic":
    agent = ReinforceBaselineCritic(lr, gamma, epsilon, input_dim, actions_for_algorithm)

writer = None

# On créer le writer pour TensorBoard
if algorithm_name != '':

    if not os.path.exists('runs/' + algorithm_name):
        os.makedirs('runs/' + algorithm_name)
    else:
        os.system('rm -rf runs/' + algorithm_name)

    # Création du writer
    writer = SummaryWriter('runs/' + algorithm_name)
    
else:
    # Création de la fenêtre pygame
    screen = pygame.display.set_mode((800, 800))
    
    # Initialisation de la police
    font = pygame.font.SysFont('Arial', 20)

########### FONCTIONS POUR LES ALGORITHMES ###########
def step_QLearning():
    """
    Réalise une étape pour l'algorithme QLearning.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Réupération du nouvel état du jeu
    new_state = get_state()
    
    # Apprentissage à partir de l'expérience
    agent.learn(state, action, reward, new_state)

def step_DQN():
    """
    Réalise une étape pour l'algorithme DQN.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Réupération du nouvel état du jeu
    new_state = get_state()
    
    # Apprentissage à partir de l'expérience
    agent.learn(state, action, reward, new_state)
    
def step_DDQN():
    """
    Réalise une étape pour l'algorithme DDQN.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Réupération du nouvel état du jeu
    new_state = get_state()
    
    # Apprentissage à partir de l'expérience
    agent.learn(state, action, reward, new_state)
    
def step_DDQNWER():
    """
    Réalise une étape pour l'algorithme DDQNWER.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Réupération du nouvel état du jeu
    new_state = get_state()
    
    # On ajoute à la mémoire de l'agent
    agent.add_in_memory(state, action, reward, new_state)
    
    # Apprentissage à partir de l'expérience
    agent.learn()
    
    # Tous les 20 épisodes, on met à jour les poids du réseau cible
    if current_episode%20 == 0:
        agent.update_target_network()
        
def step_DDQNWPER():
    """
    Réalise une étape pour l'algorithme DDQNWPER.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Réupération du nouvel état du jeu
    new_state = get_state()
    
    # On ajoute à la mémoire de l'agent
    agent.add_in_memory(state, action, reward, new_state)
    
    # Apprentissage à partir de l'expérience
    agent.learn()
    
    # Tous les 20 épisodes, on met à jour les poids du réseau cible
    if current_episode%20 == 0:
        agent.update_target_network()
    
def step_Reinforce():
    """
    Réalise une étape pour l'algorithme REINFORCE.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Ajout de la récompense à la liste des récompenses
    agent.rewards.append(reward)
    
    # Apprentissage à partir de l'expérience
    agent.learn()
    
def step_ReinforceMeanBaseline():
    """
    Réalise une étape pour l'algorithme REINFORCE avec une ligne de base.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Ajout de la récompense à la liste des récompenses
    agent.rewards.append(reward)
    
    # Apprentissage à partir de l'expérience
    agent.learn(state, action, reward)

def step_ReinforceBaselineCritic():
    """
    Réalise une étape pour l'algorithme REINFORCE avec un critique.
    """
    # Réupération de l'état actuel du jeu
    state = get_state()
    
    # Réupération de l'action à réaliser
    action = agent.choose_action(state)
    
    # Réalisation de l'action
    step(action + 1)
    
    # Réupération de la récompense pour l'action réalisée
    reward = get_reward(action)
    
    # Ajout de la récompense à la liste des récompenses
    agent.rewards.append(reward)
    
    # Apprentissage à partir de l'expérience
    agent.learn(state, action, reward)

if __name__ == '__main__':
   
   # On dessine le jeu une première fois
   if algorithm_name == '':
        draw()
   
   # Tant que l'on ne souhaite pas quitter
   while True:
    
        # Si il y'a un évenement dans la fenêtre pygame
        for event in pygame.event.get():
            # Si l'évenement est de quitter
            if event.type == pygame.QUIT:
                # On quitte le jeu
                pygame.quit()
                sys.exit()
                
            # Si l'évenement est un clic de souris
            if event.type == pygame.MOUSEBUTTONDOWN and algorithm_name == '':
                x, y = pygame.mouse.get_pos()
                        
                # Sauvegarder les checkpoints
                if 50 <= x <= 320 and 150 <= y <= 200:
                    step(str(1))
                    
                # Avancer
                for i in range(2, 13):
                    for j in range(0, board_limits[str(i)]):
                        if i * 50 <= event.pos[0] <= i * 50 + 50 and 750 - j * 50 <= event.pos[1] <= 750 - j * 50 + 50:
                            step(str(i))
            

        # On vérifie si le joueur peut jouer 
        if len(current_player.checkpoints_columns) == 3 and not any([column in get_current_player_possible_actions() for column in current_player.checkpoints_columns]):

            # On réinitialise les checkpoints du joueur actif
            reset_current_player_checkpoints()
            
            # On change de joueur
            switch_players()
        
        # Si le joueur actif est un agent    
        if current_player.name == "J1":
            if algorithm_name == "QLearning":
                step_QLearning()
            
            elif algorithm_name == "DQN":
                step_DQN()
            
            elif algorithm_name == "DDQN":
                step_DDQN()
                
            elif algorithm_name == "DDQNWER":
                step_DDQNWER()
                
            elif algorithm_name == "DDQNWPER":
                step_DDQNWPER()
            
            elif algorithm_name == "Reinforce":
                step_Reinforce()
                
            elif algorithm_name == "ReinforceMeanBaseline":
                step_ReinforceMeanBaseline()
                
            elif algorithm_name == "ReinforceBaselineCritic":
                step_ReinforceBaselineCritic()
            
        else:
            # On récupère l'action à réaliser
            action = get_current_player_random_action()
            
            # On réalise l'action
            step(action)
            
        # Si la partie est terminée
        if is_over():
            
            # Si on utilise un algorithme
            if algorithm_name != '':
                # on sauvegarde dans TensorBoard les informations utiles
                writer.add_scalar('Loss', agent.loss, current_episode) # Moyenne de la loss
                writer.add_scalar('Nb Victories J1', nb_victories[player_one.name], current_episode)
                writer.add_scalar('Nb Victories J2', nb_victories[player_two.name], current_episode)
          
            # On récupère le joueur gagnant
            winner = get_winner()
            
            # Si il y'a un gagnant
            if winner:
                # On incrémente le nombre de victoires du joueur gagnant
                nb_victories[winner.name] += 1
            
            # On réinitialise le jeu
            reset()  

        if algorithm_name == '':
            draw()
            time.sleep(1)
            

  