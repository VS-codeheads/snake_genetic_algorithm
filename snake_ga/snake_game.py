import pygame
import sys
import random
import numpy as np
import time
import copy
from numpy.core.multiarray import ndarray
from pygame.math import Vector2
from typing import List

# ---------------------------------------------
# SnakeGame, Vector, Snake, Food (game engine)
# ---------------------------------------------
def relu(a):
    return np.maximum(0, a)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)     # numerisk stabilisering
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ------------------------------------
# Vector class representing 2D vectors
# ------------------------------------
class Vector:
    # Constructor
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __str__(self):
        # Text representation
        return f'Vector({self.x}, {self.y})'

    def __add__(self, other: 'Vector') -> 'Vector':
        # Addition af to vektorer - essensen i Snake - ny position = gammel position + bevægelse
        return Vector(self.x + other.x, self.y + other.y)

    def within(self, scope: 'Vector') -> bool:
        # Within(scope) tjekker om vektoren er inde i banen
        return self.x <= scope.x and self.x >= 0 and self.y <= scope.y and self.y >= 0

    def __eq__(self, other: 'Vector') -> bool:
        # Sammenligning af to vektorer - er to positioner ens?
        return self.x == other.x and self.y == other.y

    @classmethod
    def random_within(cls, scope: 'Vector') -> 'Vector':
        # lav en tilfældig posistion indenfor scope (banen)
        return Vector(random.randint(0, scope.x - 1), random.randint(0, scope.y - 1))

# ---------------------------------------------------
# Food class representing the food object in the game
# ---------------------------------------------------
class Food:
    # Constructor
    def __init__(self, game: SnakeGame): # Den modtager spillobjektet fra SnakeGame
        self.game = game # fortæller Food hvor stor banen er
        self.p = Vector.random_within(self.game.grid) # finder en tilfældig position indenfor banen

# ----------------------------------------------
# Snake class representing the snake in the game
# ----------------------------------------------
class Snake:
    def __init__(self, *, game: SnakeGame):
        self.game = game # Slangen får adgang til hele spillet, fx grid-størrelse og settings
        self.score = 0 # Starter på 0, stiger når slangen spiser mad
        self.v = Vector(0, 0) # slangen starter uden bevægelse, når brugeren trykker - ændres v til henholdsvis (-1,0),(1,0),(0,-1),(0,1)
        self.body = deque() # deque er en dobbelt-ended kø — perfekt fordi slangen: får nyt hoved forrest (appendleft) smider halen bagerst (pop)
        self.body.append(Vector.random_within(self.game.grid)) # start position

    def move(self):
        # slage bevægelse
        # self.p henter slangens hoved
        # self.p + self.v lægger retningen til ny position
        self.p = self.p + self.v # Slangen bevæger sig ét grid-felt i retningen v

    # Kollision med egen hale
    @property
    def cross_own_tail(self):
        try:
            self.body.index(self.p, 1)
            return True
        except ValueError:
            return False

    # Accessors for p (hovedets position)
    # Getter
    @property
    def p(self): # Slangens hoved er første element i body-listen
        return self.body[0]

    # Setter
    @p.setter
    def p(self, value): # Slangen bevæger sig frem uden at ændre længde
        self.body.appendleft(value) # Ny hovedposition tilføjes forrest i body
        self.body.pop() # Sidste led fjernes (halen)

    # Slangen vokser når den spiser
    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

    def debug(self):
        # Debugging - viser slangens segmenter
        print('===')
        for i in self.body:
            print(str(i))