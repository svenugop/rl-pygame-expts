import cv2
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

        # Move the leader box either manually or along an automated track
        pressed = False
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]: y -= step
        if pressed[pygame.K_DOWN]: y += step
        if pressed[pygame.K_LEFT]: x -= step
        if pressed[pygame.K_RIGHT]: x += step

        screen.fill((0, 0, 0))
        color = (0,128,255)
        colorBGR = (255,128,0)
        stalkerColor = (255,128,0)
        stalkerColorBGR = (0,128,255)

        pygame.draw.rect(screen, color, pygame.Rect(x, y, 20, 20))


        # Capture the game state (the rectangle along with both players);
        # OR you can pass in to the training code -- the dimensions of the play area, the position of both players
        currScreen = np.zeros((1000,1000,3),np.uint8)
            
        ## Draw the leader rect
        cv2.rectangle(currScreen,(x,y),(x+20,y+20),colorBGR,-1)
        ## Draw the stalker rect
        (sx, sy) = stalker.getPosition()
        cv2.rectangle(currScreen,(sx,sy),(sx+20,sy+20),stalkerColorBGR,-1)

        # Pass in the current state to the QAgent object (stalker) to use for training
        
        # # Call stalker.trainModel

        # stalker.train(currentState)
        # (sx, sy) = stalker.getPosition()
        pygame.draw.rect(screen, stalkerColor, pygame.Rect(sx, sy, 20, 20))

        pygame.display.flip()

        clock.tick(60)
