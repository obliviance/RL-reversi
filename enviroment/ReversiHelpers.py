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
    def _player_to_action_space(self,player):
        if(player==DISK_BLACK): return 0
        if(player==DISK_WHITE): return 1
        else: raise ValueError("Invalid input for player: " + str(player))

    def __init__(self, render_mode=None, init_state : np.ndarray = None, starting_player:int = DISK_BLACK, my_player : int = DISK_BLACK):

            # Initial info :
            assert my_player in [DISK_WHITE, DISK_BLACK]
            self.player = my_player
            self.current_player = starting_player
            self.length = 8
            self.shape = (8,8)


            # observation & action space
            self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(self.length),gym.spaces.Discrete(self.length))) # 8*8=64 actions
            self.observation_space = gym.spaces.Tuple((gym.spaces.Box(
                low=DISK_BLACK, 
                high=DISK_WHITE, 
                shape=self.shape,
                dtype=np.int_
            ), gym.spaces.Discrete(2)))

            self.quit_render = False
            self.render_mode = render_mode
            self.window_cell_size = 50  # The length of a cell in the PyGame window
            self.window_size = np.array(self.shape) * self.window_cell_size
            self.window = None  # window we draw to
            self.clock = None  # control framerate

            if(init_state is not None):
                self.board = init_state.astype(np.int_)
            else:
                self.board = self.reset()[0]

    def set_state(self, state : np.ndarray, current_player: int, my_player: int = DISK_BLACK):
        self.quit_render = False
        self.current_player = current_player
        self.player = my_player
        self.board = state
        return (self.board, self._player_to_action_space(self.current_player))

    def reset(self):
        self.quit_render = False
        self.current_player = DISK_BLACK
        self.board = np.zeros((8,8),dtype=np.int_)
        self.board[3,3] = DISK_WHITE
        self.board[4,4] = DISK_WHITE
        self.board[3,4] = DISK_BLACK
        self.board[4,3] = DISK_BLACK
        return (self.board, self._player_to_action_space(self.current_player))

    def render(self, draw_legal_moves = False):
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

        legal_moves = None
        if draw_legal_moves:
            legal_moves = self.get_legal_moves()

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
            if legal_moves is not None and legal_moves[r,c]:
                color = (255,25,55)
                pygame.draw.circle(
                    self.window,
                    color,
                    ((c + 0.5) * self.window_cell_size , (r + 0.5) * self.window_cell_size),
                    self.window_cell_size / 6,
                )
       
        # The following line copies our drawings from `canvas` to the visible window
        pygame.event.pump()
        if self.is_game_over():
            pygame.display.set_caption("White is winner!" if self.get_winner() else "Black is Winner!")
        else:
            pygame.display.set_caption("White to move." if self.current_player == DISK_WHITE else "Black to move.")
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

    def check_if_legal_move(self, action: tuple, board = None):
        if board is None:
            board = self.board
        if board[action[0], action[1]] != EMPTY_SPACE:
            return (False, [])
        
        my_color = self.current_player
        opponent_color = -my_color

        action = np.array(action)
        is_legal = False
        legal_directions = []
        for direction in ADJACENCY_GRID:
            legal = True
            direction = np.array(direction)
            length = -1
            for i in range(1,8):
                current_space = action + direction * i

                # Bounds Checking
                if(not ((current_space>=0) & (current_space < 8)).all()):
                    legal = False
                    break

                # Get actual value of current space
                current_board_value = board[current_space[0],current_space[1]].astype(np.int_) 
                
                if(i > 1 and current_board_value == my_color):
                    legal = True
                    length = i
                    break
                if(current_board_value != opponent_color):
                    legal = False
                    break

            if legal:
                legal_directions.append((direction, length))
                is_legal = True
        return is_legal, legal_directions
    
    def get_legal_moves(self, board=None, return_as="board"):
        legal_moves = np.zeros((8,8))
        for r, c in np.ndindex(self.shape):
            legal_moves[r,c] = self.check_if_legal_move((r,c), board=board)[0]

        if(return_as.lower() == "board"):
            return legal_moves
        elif(return_as.lower() == "list"):
            return np.argwhere(legal_moves)
        else:
            raise ValueError("return_as has invalid value: " + return_as)
        
    def is_game_over(self):
        return (self.board != EMPTY_SPACE).all()
    
    def get_winner(self):
        return DISK_WHITE if self.board.sum() > 0 else DISK_BLACK
    
    def _take_action(self, action : tuple):
        is_legal, legal_moves = self.check_if_legal_move(action)
        if not is_legal: return
        
        action = np.array(action)
        for direction, length in legal_moves:
            for i in range(length):
                current_space = action + direction * i
                self.board[current_space[0],current_space[1]] = self.current_player
        
    def step(self, action : tuple):
        legal_moves = self.get_legal_moves(return_as="list")

        if(action not in legal_moves):
            return (self.board, self._player_to_action_space(self.current_player)), 0, False, False, {"error":"invalid action"}
       
        self._take_action(action)
        self.current_player = -self.current_player
        if len(self.get_legal_moves(return_as="list")) == 0:
            self.current_player = -self.current_player
        

        legal_moves = self.get_legal_moves(return_as="list")
        reward = 0
        terminated = False
        info = {"legal_moves":legal_moves}
        if self.is_game_over() or len(legal_moves) == 0:
            gameSum = self.board.sum()
            reward = -1 if gameSum == 0 else np.sign(gameSum * self.player)
            terminated = True
            info = {}
        
        return (self.board, self._player_to_action_space(self.current_player)), reward, terminated, False, info

            
