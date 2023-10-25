##S
import numpy as np
import gymnasium as gym
from enum import Enum
import pygame

class occupancy(Enum):
    EMPTY = 0
    WHITE = 1
    BLACK = 2     

class state():
    def __init__(self, boardX = 8, boardY = 8):
        self.board = {}
        self.width = boardX
        self.height = boardY
        #initialize board to empty
        for y in range(self.height):
            for x in range(self.width):
                  self.board[(x,y)] = occupancy.EMPTY
        #set the starting places
        self.set(3,3,occupancy.BLACK)
        self.set(4,3,occupancy.WHITE)
        self.set(3,4,occupancy.WHITE)
        self.set(4,4,occupancy.BLACK)

    def __str__(self):
        string = ""
        for y in range(self.height):
            for x in range(self.width):
                string += self.board[(x,y)].name
                string += "\t"
            string += "\n\n"
        return string

    def set(self, x, y, occupancy):
        self.board[(x,y)] = occupancy
                 
class gameBoard(gym.Env):
    metadata = {"render_fps":20}
    def __init__(self, render_mode=None):

            # From any state the agent can perform one of four actions, up, down, left or right
            #self._n_actions = 4
            #self._n_states = int(np.sum(self._occupancy == 0))
            self._limits = [8,8] #default board size
            #initial state:
            self.init_state = state(boardX = self._limits[0], boardY = self._limits[1])


            # Standard Gym interface
            #idk how we are goint to do this: self.observation_space = gym.spaces.Box(low=0, high=self.n_states-1, dtype=int)  # cell index
            #idk how we are goint to do this: self.action_space = gym.spaces.Discrete(self.n_actions)
            assert render_mode is None
            self.render_mode = render_mode

            self.window_cell_size = 50  # The length of a cell in the PyGame window
            self.window_size = np.array(self._limits) * self.window_cell_size
            self.window = None  # window we draw to
            self.clock = None  # control framerate
            self.state = self.init_state
            #remove render once we have code to drive the enviroment
            self.render()

    def reset(self, init_state=None):
        if init_state is not None:
            #assuming this is a valid state
            state = init_state
        #render
        self.render()
        return state

    def render(self):
        #mostly copied from A2helpers.py
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        #Draw background
        for y in range(self._limits[1]):
            for x in range(self._limits[0]):
                #draw background
                pygame.draw.rect(
                    canvas,
                    (20, 148, 3),
                    pygame.Rect(
                        (y * self.window_cell_size, x * self.window_cell_size),
                        (self.window_cell_size, self.window_cell_size),
                    ),
                )
        #Draw peices
        for y in range(self._limits[1]):
            for x in range(self._limits[0]):
                if self.state.board[x,y] == occupancy.WHITE:
                    #draw peice
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        ((x + 0.5) * self.window_cell_size , (y + 0.5) * self.window_cell_size),
                        self.window_cell_size / 3,
                    )
                if self.state.board[x,y] == occupancy.BLACK:
                    #draw peice
                    pygame.draw.circle(
                        canvas,
                        0,
                        ((x + 0.5) * self.window_cell_size , (y + 0.5) * self.window_cell_size),
                        self.window_cell_size / 3,
                    )

        # Finally, add some gridlines
        for x in range(self._limits[0]):
            pygame.draw.line(
                canvas,
                0,
                (self.window_cell_size * x, 0),
                (self.window_cell_size * x, self.window_size[1]),
                width=3,
            )
        for y in range(self._limits[1]):
            pygame.draw.line(
                canvas,
                0,
                (0, self.window_cell_size * y),
                (self.window_size[0], self.window_cell_size * y),
                width=3,
            )
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
