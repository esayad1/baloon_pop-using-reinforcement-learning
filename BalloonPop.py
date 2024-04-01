import os
import pygame
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from ai.QLearning import QLearning # Q Learning
from ai.DQN import DQN # Deep Q Learning
from ai.DDQN import DDQN # Double Deep Q Learning
from ai.DDQNWER import DDQNWER # Double Deep Q Learning with Experience Replay
from ai.DDQNWPER import DDQNWPER # Double Deep Q Learning with Prioritized Experience Replay
from ai.Reinforce import Reinforce # Algorithme REINFORCE
from ai.ReinforceMeanBaseline import ReinforceMeanBaseline # Algorithme REINFORCE avec une ligne de base
from ai.ReinforceBaselineCritic import ReinforceBaselineCritic # Algorithme REINFORCE avec un critique
# Initialisation de pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
FONT_COLOR = (0, 0, 0)
GRID_COLOR = (0, 0, 0)
MARGIN_LEFT = 50  # Marge à gauche pour positionner la grille
MARGIN_BOTTOM = 50  # Marge en bas pour positionner la grille
CELL_WIDTH = 40
CELL_HEIGHT = 40
COLUMN_SPACING = 20  # Espacement entre les colonnes

# Données des colonnes
columns_data = [
    [0, 3, 7, 11, 15, 3],
    [1, 3, 5, 7, 9, 12, 8],
    [0, 0, 0, 2, 4, 6, 8, 10, 14, 6],
    [1, 2, 3, 5, 7, 10, 13, 16, 4],
    [2, 3, 4, 5, 7, 9, 12, 5],
    [1, 3, 6, 10, 13, 7]
]

AUTOMATIC = False

# ALGORITHM = 'random'
ALORITHM = 'qlearning'

actions = [True, False]
model = None
    
# Nombre de parties jouées
nb_games = 0
loss = []


# Fonction pour dessiner la grille
def draw_grid(screen):
    for col, column in enumerate(columns_data):
        for row, number in enumerate(column):
            pygame.draw.rect(screen, GRID_COLOR, (
                MARGIN_LEFT + col * (CELL_WIDTH + COLUMN_SPACING), HEIGHT - MARGIN_BOTTOM - (row + 1) * CELL_HEIGHT,
                CELL_WIDTH, CELL_HEIGHT), 1)


# Fonction pour remplir la grille avec des chiffres
def fill_grid_with_numbers(screen):
    font = pygame.font.SysFont(None, 40)
    for col, column in enumerate(columns_data):
        for row, number in enumerate(column):
            text_surf = font.render(str(number), True, FONT_COLOR)
            text_rect = text_surf.get_rect(center=(MARGIN_LEFT + col * (CELL_WIDTH + COLUMN_SPACING) + CELL_WIDTH // 2,
                                                   HEIGHT - MARGIN_BOTTOM - (row + 1) * CELL_HEIGHT + CELL_HEIGHT // 2))
            screen.blit(text_surf, text_rect)

def draw_score_boxes(screen):
    score_labels = ["Break 1", "Break 2", "Break 3", "Score Final"]
    scores = ["40", "40", "40", "40"]
    box_width, box_height = 100, 40
    spacing = 10

    for i, label in enumerate(score_labels):
        rect = (10, 10 + i * (box_height + spacing), box_width, box_height)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        font = pygame.font.SysFont(None, 25)
        label_surf = font.render(label, True, (0, 0, 0))
        label_rect = label_surf.get_rect(
            center=(10 + box_width // 2, 10 + i * (box_height + spacing) + box_height // 2 - 10))
        screen.blit(label_surf, label_rect)
        score_surf = font.render(scores[i], True, (0, 0, 0))
        score_rect = score_surf.get_rect(
            center=(10 + box_width // 2, 10 + i * (box_height + spacing) + box_height // 2 + 10))
        screen.blit(score_surf, score_rect)

# Classes et fonctions du fichier dice.py
class Die:
    def __init__(self, faces, image_paths):
        self.faces = faces
        self.current_face = faces[0]
        self.images = [pygame.image.load(path) for path in image_paths]

    def roll(self):
        self.current_face = random.choice(self.faces)
        return self.current_face

    def get_image(self, width=50, height=50):
        index = self.faces.index(self.current_face)
        resized_image = pygame.transform.scale(self.images[index], (width, height))
        return resized_image


class SelectableDie(Die):
    def __init__(self, faces, image_paths):
        super().__init__(faces, image_paths)
        self.selected = False

    def toggle_selected(self):
        self.selected = not self.selected

    def reset_selection(self):
        self.selected = False

    def get_image(self, width=50, height=50):
        index = self.faces.index(self.current_face)
        resized_image = pygame.transform.scale(self.images[index], (width, height))
        if self.selected:
            pygame.draw.circle(resized_image, (0, 0, 0), (width // 2, height // 2), width // 2, 2)
        return resized_image


class Button:
    def __init__(self, x, y, width, height, text, color=(0, 128, 0), hover_color=(50, 200, 50)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hover_color = hover_color

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        if self.x < mouse_pos[0] < self.x + self.width and self.y < mouse_pos[1] < self.y + self.height:
            pygame.draw.rect(screen, self.hover_color, (self.x, self.y, self.width, self.height))
        else:
            pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

        font = pygame.font.SysFont(None, 25)
        text_surf = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surf, text_rect)

    def is_clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        return self.x < mouse_pos[0] < self.x + self.width and self.y < mouse_pos[1] < self.y + self.height


# Chemins des images
IMAGES = ["./assets/images/red_star.png", "./assets/images/red_moon.png", "./assets/images/red_kite.png",
          "./assets/images/blue_star.png", "./assets/images/blue_moon.png", "./assets/images/yellow_star.png"]

# Faces des dés
FACES = ["Rouge étoile", "Rouge lune", "Rouge cerf-volant", "Bleu étoile", "Bleu lune", "Jaune étoile"]

# Initialisation des dés avec les chemins d'images
die = SelectableDie(FACES, IMAGES)
die_2 = SelectableDie(FACES, IMAGES)
die_3 = SelectableDie(FACES, IMAGES)
die_4 = SelectableDie(FACES, IMAGES)
die_5 = SelectableDie(FACES, IMAGES)

dice_list = [die, die_2, die_3]

def roll_dice(dice):
    for die in dice:
        die.roll()


def update_grid(screen):
    for col, column in enumerate(columns_data):
        for row, number in enumerate(column):
            pygame.draw.rect(screen, GRID_COLOR, (
                MARGIN_LEFT + col * (CELL_WIDTH + COLUMN_SPACING), HEIGHT - MARGIN_BOTTOM - (row + 1) * CELL_HEIGHT,
                CELL_WIDTH, CELL_HEIGHT), 1)
            if (col, row) in highlighted_cells:
                pygame.draw.rect(screen, (255, 0, 0), (
                    MARGIN_LEFT + col * (CELL_WIDTH + COLUMN_SPACING), HEIGHT - MARGIN_BOTTOM - (row + 1) * CELL_HEIGHT,
                    CELL_WIDTH, CELL_HEIGHT), 3)

highlighted_cells = set()  # Ensemble pour stocker les cases à mettre en évidence

last_highlighted_position = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1}


def highlight_value(screen, column, count):
    global last_highlighted_position
    if count <= len(columns_data[column]):
        row = last_highlighted_position[column] + count
        if row < len(columns_data[column]):
            pygame.draw.rect(screen, (255, 0, 0), (
                MARGIN_LEFT + column * (CELL_WIDTH + COLUMN_SPACING), HEIGHT - MARGIN_BOTTOM - (row + 1) * CELL_HEIGHT,
                CELL_WIDTH, CELL_HEIGHT), 3)
            last_highlighted_position[column] = row


def highlight_current_values(screen):
    for column, position in last_highlighted_position.items():
        if 0 <= position < len(columns_data[column]):
            pygame.draw.rect(screen, (255, 0, 0), (
                MARGIN_LEFT + column * (CELL_WIDTH + COLUMN_SPACING),
                HEIGHT - MARGIN_BOTTOM - (position + 1) * CELL_HEIGHT,
                CELL_WIDTH, CELL_HEIGHT), 3)


def apply_scoring(dice_faces, screen):
    global highlighted_cells, break_scores
    colors = {
        "Rouge": 2,
        "Jaune": 0,
        "Bleu": 1
    }
    motifs = {
        "étoile": 3,
        "lune": 4,
        "cerf-volant": 5
    }

    for color, column in colors.items():
        count = dice_faces.count(color)
        if count:
            highlight_value(screen, column, count)

    for motif, column in motifs.items():
        count = dice_faces.count(motif)
        if count:
            highlight_value(screen, column, count)

    for col, column in enumerate(columns_data):
        if (col, len(column) - 1) in highlighted_cells:
            # Trouve le premier score de break qui n'a pas encore été défini
            for i in range(3):
                if break_scores[i] is None:
                    break_scores[i] = sum(columns_data[col][cell[1]] for cell in highlighted_cells if cell[0] == col)
                    break

break_scores = [0, 0, 0]

def last_element_selected(column):
    """Vérifie si le dernier élément d'une colonne est sélectionné."""
    return last_highlighted_position[column] == len(columns_data[column]) - 1


columns_used_for_breaks = []

last_scored_column = None

completed_columns = []


def sum_highlighted_scores():
    """Calcule la somme des scores des cases entourées en rouge."""
    total_score = 0
    for col, position in last_highlighted_position.items():
        if position != -1:  # Si une valeur est entourée en rouge dans cette colonne
            total_score += columns_data[col][position]
    return total_score

game_over = False


def update_break_scores():
    """Mettre à jour les scores des breaks."""
    global break_scores, game_over, columns_used_for_breaks

    completed_columns_this_turn = [col for col, position in last_highlighted_position.items() if
                                   position == len(columns_data[col]) - 1]
    for col in completed_columns_this_turn:
        if col not in columns_used_for_breaks:
            # Trouver le premier score de break qui n'a pas encore été défini
            for i in range(3):
                if break_scores[i] == 0:
                    break_scores[i] = sum_highlighted_scores()
                    columns_used_for_breaks.append(col)
                    break
            break

    if all(score != 0 for score in break_scores):
        game_over = True

def draw_score_boxes(screen):
    """Dessine les boîtes de score sur l'écran."""
    score_labels = ["Break 1", "Break 2", "Break 3", "Score Final"]
    scores = [str(break_scores[i]) if break_scores[i] is not None else "" for i in range(3)]
    if all(val is not None for val in break_scores):
        scores.append(str(sum(filter(None, break_scores))))
    else:
        scores.append("")
    box_width, box_height = 100, 40
    spacing = 10

    for i, label in enumerate(score_labels):
        rect = (10, 10 + i * (box_height + spacing), box_width, box_height)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        font = pygame.font.SysFont(None, 25)
        label_surf = font.render(label, True, (0, 0, 0))
        label_rect = label_surf.get_rect(
            center=(10 + box_width // 2, 10 + i * (box_height + spacing) + box_height // 2 - 10))
        screen.blit(label_surf, label_rect)
        score_surf = font.render(scores[i], True, (0, 0, 0))
        score_rect = score_surf.get_rect(
            center=(10 + box_width // 2, 10 + i * (box_height + spacing) + box_height // 2 + 10))
        screen.blit(score_surf, score_rect)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

def get_state():
    """Retourne l'état courant du jeu."""
    global last_highlighted_position
    state = []
    
    # On recupère les valeurs des cases entourées en rouge
    for col, position in last_highlighted_position.items():
        if position != -1:
            state.append(columns_data[col][position])
        else:
            state.append(0)
            
    return state

def get_reward(action):
    """Retourne la récompense."""
    # Si on est tous en haut de chaque colonne
    if all(position == len(columns_data[col]) - 1 for col, position in last_highlighted_position.items()) and action == 1:
        return -10
    else:
        bonus = sum(filter(None, break_scores))
        return 1 + bonus / 100

def reset_game():
    global highlighted_cells, last_highlighted_position, break_scores, columns_used_for_breaks, last_scored_column, completed_columns, game_over, loss
    highlighted_cells = set()  # Ensemble pour stocker les cases à mettre en évidence

    last_highlighted_position = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1}

    break_scores = [0, 0, 0]

    columns_used_for_breaks = []

    last_scored_column = None

    completed_columns = []

    game_over = False

    loss = []

def display_dice(screen, automatic=False, algorithm='random', sleep_time=1, episode=0):
    global dice_list, die, die_2, die_3, die_4, die_5, game_over, nb_games, model, loss, reward

    roll_dice(dice_list)  # Lancer automatiquement les dés au démarrage

    running = True
    roll_button = Button(WIDTH - 200, 50, 150, 40, "Conserver tous !")
    reroll_button = Button(WIDTH - 400, 50, 150, 40, "Relancer", color=(128, 0, 0), hover_color=(200, 50, 50))
    reroll_button_visible = True
    quit_button = Button(WIDTH - 600, 50, 150, 40, "Quitter", color=(128, 0, 0), hover_color=(200, 50, 50))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

        if not automatic:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if roll_button.is_clicked():
                    dice_faces = [die.current_face.split(" ")[0] for die in dice_list]  # Extract colors
                    dice_faces += [die.current_face.split(" ")[1] for die in dice_list]  # Extract motifs
                    apply_scoring(dice_faces, screen)
                    dice_to_roll = [die for die in dice_list if not die.selected]
                    roll_dice(dice_to_roll)
                    update_break_scores()

                    for die in dice_list:
                        die.reset_selection()

                    dice_list = [SelectableDie(FACES, IMAGES) for _ in range(3)]
                    roll_dice(dice_list)

                elif reroll_button.is_clicked():
                    dice_to_roll = [die for die in dice_list if not die.selected]
                    roll_dice(dice_to_roll)
                    if len(dice_list) < 5:
                        dice_list.append(SelectableDie(FACES, IMAGES))
                        dice_list[-1].roll()
                else:
                    mouse_pos = pygame.mouse.get_pos()
                    x_coord = WIDTH - 120
                    for i, die in enumerate(dice_list):
                        y_coord = HEIGHT - 120 - (90 * i)
                        if x_coord < mouse_pos[0] < x_coord + 50 and y_coord < mouse_pos[1] < y_coord + 50:
                            die.toggle_selected()
                            break

    if automatic:
        # Action par défaut
        action = True

        if algorithm == 'random':
            action = random.choice(actions)

        # Q Learning
        elif algorithm == 'qlearning':
            state = get_state()
            action = model.choose_action(state, actions)
            reward = get_reward(action)
            next_state = get_state()
            model.learn(state, action, reward, next_state)
            loss.append(model.loss)
        
        # Deep Q Learning
        elif algorithm == 'dqn':
            state = get_state()
            action = model.choose_action(state)
            reward = get_reward(action)
            next_state = get_state()
            
            print(f"State:      {state},\nAction:     {int(action)},\nReward:     {reward},\nNext State: {next_state} | loss: {model.loss}")
            print("___"*20)
            
            model.learn(state, action, reward, next_state)
            loss.append(model.loss)
        
        # Double Deep Q Learning
        elif algorithm == 'ddqn':
            state = get_state()
            action = model.choose_action(state)
            reward = get_reward(action)
            next_state = get_state()
            
            print(f"State:      {state},\nAction:     {int(action)},\nReward:     {reward},\nNext State: {next_state} | loss: {model.loss}")
            print("___"*20)
            
            model.learn(state, action, reward, next_state)
            loss.append(model.loss)
            
            # Toutes les 20 parties, on met à jour le réseau cible
            if episode % 20 == 0:
                model.update_target_network()
        
        # Double Deep Q Learning with Experience Replay
        elif algorithm == 'ddqnwer':
            state = get_state()
            action = model.choose_action(state)
            reward = get_reward(action)
            next_state = get_state()
            
            print(f"State:      {state},\nAction:     {int(action)},\nReward:     {reward},\nNext State: {next_state} | loss: {model.loss}")
            print("___"*20)
            
            model.add_in_memory(state, action, reward, next_state)
            model.learn()
            loss.append(model.loss)
            
            # Toutes les 20 parties, on met à jour le réseau cible
            if episode % 20 == 0:
                model.update_target_network()
        
        # Double Deep Q Learning with Prioritized Experience Replay  
        elif algorithm == 'ddqnwper':
            state = get_state()
            action = model.choose_action(state)
            reward = get_reward(action)
            next_state = get_state()
            
            print(f"State:      {state},\nAction:     {int(action)},\nReward:     {reward},\nNext State: {next_state} | loss: {model.loss}")
            print("___"*20)
                 
            model.add_in_memory(state, action, reward, next_state)
            model.learn()
            loss.append(model.loss)
            
            # Toutes les 20 parties, on met à jour le réseau cible
            if episode % 20 == 0:
                model.update_target_network()
                
        elif algorithm == 'reinforce' or algorithm == 'reinforcemeanbaseline' or algorithm == 'reinforcebaselinecritic':
            state = get_state()
            action = model.choose_action(state)
            reward = get_reward(action)
            
            print(f"State:      {state},\nAction:     {int(action)},\nReward:     {reward},\nNext State: {next_state} | loss: {model.loss}")
            print("___"*20)
        
            model.rewards.append(reward)
            model.learn()
            loss.append(model.loss)
        
        if action:
            dice_faces = [die.current_face.split(" ")[0] for die in dice_list]  # Extract colors
            dice_faces += [die.current_face.split(" ")[1] for die in dice_list]  # Extract motifs
            apply_scoring(dice_faces, screen)
            dice_to_roll = [die for die in dice_list if not die.selected]
            roll_dice(dice_to_roll)
            update_break_scores()
            for die in dice_list:
                die.reset_selection()
            dice_list = [SelectableDie(FACES, IMAGES) for _ in range(3)]
            roll_dice(dice_list)
        
        else:
            dice_to_roll = [die for die in dice_list if not die.selected]
            roll_dice(dice_to_roll)
            if len(dice_list) < 5:
                dice_list.append(SelectableDie(FACES, IMAGES))
                dice_list[-1].roll()
    
    screen.fill(BACKGROUND_COLOR)
    update_grid(screen)
    highlight_current_values(screen)
    fill_grid_with_numbers(screen)
    draw_score_boxes(screen)

    x_coord = WIDTH - 120
    y_coord_start = HEIGHT - 120

    for i, die in enumerate(dice_list):
        y_coord = y_coord_start - (90 * i)
        screen.blit(die.get_image(), (x_coord, y_coord))
        if die.selected:
            pygame.draw.circle(screen, (0, 0, 0), (x_coord + 25, y_coord + 25), 30, 2)

    roll_button.draw(screen)
    reroll_button.draw(screen)

    pygame.display.flip()

    if len(dice_list) == 5 and not game_over:
        dice_faces = [die.current_face.split(" ")[0] for die in dice_list]  # Extract colors
        dice_faces += [die.current_face.split(" ")[1] for die in dice_list]  # Extract motifs
        apply_scoring(dice_faces, screen)
        dice_list = [SelectableDie(FACES, IMAGES) for _ in range(3)]
        roll_dice(dice_list)

def main(automatic=False, algorithm='random', sleep_time=1, nb_episodes=1000):
    global screen, model, actions, game_over, nb_games, loss, reward

    if algorithm == 'qlearning':
        model = QLearning(0.1, 0.9, 0.2, actions)    
    
    elif algorithm == 'dqn':
        model = DQN(0.01, 0.9, 0.2, len(get_state()), actions)
    
    elif algorithm == 'ddqn':
        model = DDQN(0.01, 0.9, 0.2, len(get_state()), actions)
        
    elif algorithm == 'ddqnwer':
        model = DDQNWER(0.01, 0.9, 0.2, len(get_state()), actions)
    
    elif algorithm == 'ddqnwper':
        model = DDQNWPER(0.01, 0.9, 0.2, len(get_state()), actions)
    
    elif algorithm == 'reinforce':
        model = Reinforce(0.01, 0.9, 0.2, len(get_state()), actions)
    
    elif algorithm == 'reinforcemeanbaseline':
        model = ReinforceMeanBaseline(0.01, 0.9, 0.2, len(get_state()), actions)
        
    elif algorithm == 'reinforcebaselinecritic':
        model = ReinforceBaselineCritic(0.01, 0.9, 0.2, len(get_state()), actions)


    if not os.path.exists('runs/' + algorithm):
        os.makedirs('runs/' + algorithm)

    # On créer un summary writer pour tensorboard
    writer = SummaryWriter('runs/' + algorithm)

    pygame.display.set_caption("Dice Game Grid")

    for episode in range(nb_episodes):
        print(f"Episode en cours: {episode}")
        while not game_over:
            time.sleep(sleep_time)
            screen.fill(BACKGROUND_COLOR)
            draw_grid(screen)
            fill_grid_with_numbers(screen)
            draw_score_boxes(screen)
            display_dice(screen, automatic, algorithm, sleep_time, episode)
            pygame.display.flip()

            if game_over:
                nb_games += 1
                
                total_score = sum(filter(None, break_scores))
                writer.add_scalar('Score', total_score, nb_games)
                writer.add_scalar('Loss', np.mean(loss), nb_games)
                reset_game()
                
                break
            
            time.sleep(sleep_time)

    pygame.quit()


if __name__ == "__main__":
    #main(automatic=True, algorithm='random', sleep_time=0, nb_episodes=1000)
    # main(automatic=False, algorithm='', sleep_time=0, nb_episodes=1000)
    #main(automatic=True, algorithm='qlearning', sleep_time=0, nb_episodes=1000)
    # main(automatic=True, algorithm='dqn', sleep_time=0, nb_episodes=1000)
    #main(automatic=True, algorithm='ddqn', sleep_time=0, nb_episodes=1000)
     #main(automatic=True, algorithm='ddqnwer', sleep_time=0, nb_episodes=1000)
    # main(automatic=True, algorithm='ddqnwper', sleep_time=0, nb_episodes=1000)
    # main(automatic=True, algorithm='reinforce', sleep_time=0, nb_episodes=1000)
    # main(automatic=True, algorithm='reinforcemeanbaseline', sleep_time=0, nb_episodes=1000)
    # main(automatic=True, algorithm='reinforcebaselinecritic', sleep_time=0, nb_episodes=1000)
