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

    metadata = {"render_fps":2}

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
        """
        Sets the board to be equal to the `state`, 
        with the next player to go being current_player and the player we currently are being `my_player`
        """
        self.quit_render = False
        self.current_player = current_player
        self.player = my_player
        self.board = state
        self.is_game_over = False
        self.winner = None
        return (self.board, self._player_to_action_space(self.current_player))

    def reset(self):
        """
        Resets the environment to basic grid with 4 tiles in the middle, 
        as well as current player being the black player, and the next player to move is the black player
        """
        self.quit_render = False
        self.current_player = DISK_BLACK
        self.board = np.zeros((8,8),dtype=np.int_)
        self.board[3,3] = DISK_WHITE
        self.board[4,4] = DISK_WHITE
        self.board[3,4] = DISK_BLACK
        self.board[4,3] = DISK_BLACK
        self.is_game_over = False
        self.winner = None
        return (self.board, self._player_to_action_space(self.current_player))

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

        legal_moves = None
        if self.render_mode == "human":
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
        font = pygame.font.SysFont(None, 24)
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

                # Display Text on legal move
                text = str(r) + "," + str(c)
                img = font.render(text, True, (255,25,55))
                color = (255,25,55)
                
                self.window.blit(img, (c* self.window_cell_size, r* self.window_cell_size))
                pygame.draw.circle(
                    self.window,
                    color,
                    ((c + 0.5) * self.window_cell_size , (r + 0.5) * self.window_cell_size),
                    self.window_cell_size / 6,
                )
       
        pygame.event.pump()
        if self.is_game_over:
            pygame.display.set_caption("White is winner!" if self.winner == DISK_WHITE else "Black is Winner!" if self.winner == DISK_BLACK else "Draw!")
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
        """
        Checks if the given action is legal for the given board (use the environment board if none is given) given the current environment player.
        - Returns a tuple of (`is_legal`, `[list of legal directions and the lengths of those directions]`)
        """
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
            legal = False
            direction = np.array(direction)
            length = -1
            for i in range(1,8):
                current_space = action + direction * i
                length = i
                # Bounds Checking
                if(not ((current_space>=0) & (current_space < 8)).all()):
                    legal = False
                    break

                # Get actual value of current space
                current_board_value = board[current_space[0],current_space[1]].astype(np.int_) 
                
                if(i > 1 and current_board_value == my_color):
                    legal = True
                    break

                if(current_board_value != opponent_color):
                    legal = False
                    break

            if legal:
                legal_directions.append((direction, length))
                is_legal = True
        return is_legal, legal_directions
    
    def get_legal_moves(self, board=None, return_as="board"):
        """
        Gets all the legal moves for the current player on the specified `board`
        - If no `board` is specified, use the environment board
        - `return_as`: specifies what the return type is
            - `return_as="board"` specifies a 8*8 grid of zeros with ones as the legal spaces
            - `return_as="list"` specifies a list of legal moves, each as a 2-tuple of int
        """
        
        legal_moves = np.zeros((8,8))
        for r, c in np.ndindex(self.shape):
            legal_moves[r,c] = self.check_if_legal_move((r,c), board=board)[0]

        if(return_as.lower() == "board"):
            return legal_moves
        elif(return_as.lower() == "list"):
            return np.argwhere(legal_moves)
        else:
            raise ValueError("return_as has invalid value: " + return_as)
        
    
    def get_winner(self):
        return self.winner
    
    def _take_action(self, action : tuple):
        is_legal, legal_moves = self.check_if_legal_move(action)
        if not is_legal: return
        
        action = np.array(action)
        for direction, length in legal_moves:
            for i in range(length):
                current_space = action + direction * i
                self.board[current_space[0],current_space[1]] = self.current_player
        
    def step(self, action : tuple):
        """
        given an action of form `(row, column)`:
        - grabs all legal moves
        - checks if current action is a legal move
        - takes action, flipping disks
        - switches player if and only if the other player has legal moves
        - check if game is over, (no more free spaces, or no legal moves for either player), giving a reward of -1 on loss or draw for the player you started as.
        - return state `(board, current_player)`, reward, terminated, truncated, and info which has either error messages, or legal moves for next player.
        """
        legal_moves = self.get_legal_moves(return_as="list")

        # Check if action in legal moves
        if(not any((legal_moves[:] == action).all(1))):
            return (self.board, self._player_to_action_space(self.current_player)), 0, False, False, {"error":"invalid action"}
       
        self._take_action(action)
        self.current_player = -self.current_player
        if len(self.get_legal_moves(return_as="list")) == 0:
            self.current_player = -self.current_player
        

        legal_moves = self.get_legal_moves(return_as="list")
        reward = 0
        terminated = False
        info = {"legal_moves":legal_moves}
        if len(legal_moves) == 0:
            gameSum = self.board.sum()
            if gameSum != 0:
                self.winner = np.sign(gameSum)
            else:
                self.winner = EMPTY_SPACE
            reward = -1 if gameSum == 0 else np.sign(gameSum * self.player)
            terminated = True
            info = {}
            self.is_game_over = True
        
        return (self.board, self._player_to_action_space(self.current_player)), reward, terminated, False, info

            
