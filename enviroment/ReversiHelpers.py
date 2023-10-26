import numpy as np
import gymnasium as gym
import pygame

EMPTY_SPACE = 0
DISK_WHITE = 1
DISK_BLACK = -1     

ADJACENCY_GRID = [
    (-1, -1), (-1,  0), (-1,  1), 
    ( 0, -1),           ( 0,  1), 
    ( 1, -1), ( 1,  0), ( 1,  1), 
]

class OthelloEnvironment(gym.Env):
    metadata = {"render_fps":20}

    def __init__(self, render_mode=None, init_state : np.ndarray = None, initial_player : int = DISK_WHITE):

            # Initial info :
            self.current_player = initial_player
            self.length = 8
            self.shape = (8,8)


            # observation & action space
            self.action_space = gym.spaces.Discrete(n=self.length**2) # 8*8=64 actions
            self.observation_space = gym.spaces.Box(
                low=DISK_BLACK, 
                high=DISK_WHITE, 
                shape=self.shape,
                dtype=np.int_
            )

            self.quit_render = False
            self.render_mode = render_mode
            self.window_cell_size = 50  # The length of a cell in the PyGame window
            self.window_size = np.array(self.shape) * self.window_cell_size
            self.window = None  # window we draw to
            self.clock = None  # control framerate

            if(init_state is not None):
                self.board = init_state.astype(np.int_)
            else:
                self.board = self.reset()

    def reset(self):
        self.quit_render = False
        board = np.zeros((8,8),dtype=np.int_)
        board[3,3] = DISK_WHITE
        board[4,4] = DISK_WHITE
        board[3,4] = DISK_BLACK
        board[4,3] = DISK_BLACK
        return board

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_render = True
                    pygame.quit()
                    return

        # Fill background
        self.window.fill((20,148,3))
        
        # Add some gridlines
        for x in range(self.length):
            pygame.draw.line(
                self.window,
                0,
                (self.window_cell_size * x, 0),
                (self.window_cell_size * x, self.window_size[1]),
                width=3,
            )
            pygame.draw.line(
                self.window,
                0,
                (0, self.window_cell_size * x),
                (self.window_size[0], self.window_cell_size * x),
                width=3,
            )

        # Draw Disks
        for r, c in np.ndindex(self.shape):
            if(self.board[r,c] in [DISK_WHITE, DISK_BLACK]):
                color = (255,255,255) if self.board[r,c] == DISK_WHITE else 0
                pygame.draw.circle(
                    self.window,
                    color,
                    ((c + 0.5) * self.window_cell_size , (r + 0.5) * self.window_cell_size),
                    self.window_cell_size / 3,
                )

       
        # The following line copies our drawings from `canvas` to the visible window
        pygame.event.pump()
        pygame.display.update()
        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def end_render(self):
        return self.quit_render
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
