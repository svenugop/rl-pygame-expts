import pygame
from QAgent import *
import numpy as np

pygame.init()
screen = pygame.display.set_mode((1000,1000))
done = False
clock = pygame.time.Clock()

x=200
y=200

step = 2
stalker = QLearningAgent()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        pressed = False
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]: y -= step
        if pressed[pygame.K_DOWN]: y += step
        if pressed[pygame.K_LEFT]: x -= step
        if pressed[pygame.K_RIGHT]: x += step

        screen.fill((0, 0, 0))
        color = (0,128,255)
        stalkerColor = (255,128,0)

        pygame.draw.rect(screen, color, pygame.Rect(x, y, 20, 20))

        if (pressed):
            myPosition = np.array([x,y])
            # The stalker's turn
            stalker.followMe(myPosition)
            (sx, sy) = stalker.getPosition()
            pygame.draw.rect(screen, stalkerColor, pygame.Rect(sx, sy, 20, 20))

        pygame.display.flip()

        clock.tick(60)