import arcade
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Constantes
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
SCREEN_TITLE = "Snake AI - Arcade & PyTorch"
BLOCK_SIZE = 20
SPEED = 20 # Vitesse pour la visualisation (plus c'est bas, plus c'est rapide en mode "no render")

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.BLACK)
        self.reset()

    def reset(self):
        # État initial
        self.direction = Direction.RIGHT
        self.head = Point(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (SCREEN_WIDTH-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (SCREEN_HEIGHT-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Gérer les événements (pour pouvoir fermer la fenêtre)
        self.dispatch_events()
        self.on_draw() # Dessiner l'état actuel
        self.update() # Mettre à jour l'écran

        # 2. Déplacer le serpent selon l'action de l'IA
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. Vérifier Game Over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Manger ou avancer
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hit boundary
        if pt.x > SCREEN_WIDTH - BLOCK_SIZE or pt.x < 0 or pt.y > SCREEN_HEIGHT - BLOCK_SIZE or pt.y < 0:
            return True
        # Hit self
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # Action [Straight, Right, Left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y -= BLOCK_SIZE # ATTENTION: Arcade Y vers le bas = soustraction
        elif self.direction == Direction.UP:
            y += BLOCK_SIZE # ATTENTION: Arcade Y vers le haut = addition

        self.head = Point(x, y)

    def on_draw(self):
        self.clear()
        # Dessiner Snake
        for pt in self.snake:
            arcade.draw_rectangle_filled(pt.x + BLOCK_SIZE/2, pt.y + BLOCK_SIZE/2, 
                                         BLOCK_SIZE, BLOCK_SIZE, arcade.color.BLUE)
            arcade.draw_rectangle_outline(pt.x + BLOCK_SIZE/2, pt.y + BLOCK_SIZE/2, 
                                          BLOCK_SIZE, BLOCK_SIZE, arcade.color.WHITE)
        
        # Dessiner Tête (différente couleur)
        arcade.draw_rectangle_filled(self.head.x + BLOCK_SIZE/2, self.head.y + BLOCK_SIZE/2, 
                                     BLOCK_SIZE, BLOCK_SIZE, arcade.color.CYAN)

        # Dessiner Nourriture
        arcade.draw_rectangle_filled(self.food.x + BLOCK_SIZE/2, self.food.y + BLOCK_SIZE/2, 
                                     BLOCK_SIZE, BLOCK_SIZE, arcade.color.RED)
        
        # Score
        arcade.draw_text(f"Score: {self.score}", 
                         10, SCREEN_HEIGHT - 20, arcade.color.WHITE, 14)

    def update(self):
        # Arcade a besoin de flip manuellement ici car on n'utilise pas arcade.run()
        try:
            super().flip()
        except:
            pass # Parfois nécessaire selon les versions d'Arcade lors du training rapide
