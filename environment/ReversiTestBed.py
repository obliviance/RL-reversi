from ReversiHelpers import *
import sys 

def human_game():
    render_mode = "human" # or "human" to engage human play
    env = OthelloEnvironment(render_mode = render_mode)
    if (render_mode == "human"):
        terminated = False
        while(not terminated):

            #render the enviroment to show the player the state and the actions they can take
            env.render()

            # Ask for Input
            input_string = input("Please choose a position to place your piece, in the form 'row col': ")
            input_values = input_string.split(' ')
            
            # Verify if all 2 inputs are digits
            if(not(len(input_values) == 2 and all([element.isdigit() for element in input_values]))):
                print("Invalid Input")
                continue

            # Verify if action is legal
            legal_moves = env.get_legal_moves(return_as="list")
            action = tuple((int(x) for x in input_values))
            if(action not in legal_moves):
                print("Action not a legal move")
                continue
            
            # Do Move
            (_, _), _, terminated, _, _= env.step(action)

        # Game over
        if (env.get_winner() == DISK_WHITE):
            print("----CONGRATULATIONS TO WHITE FOR WINNING THE GAME----")
        if (env.get_winner() == DISK_BLACK):
            print("----CONGRATULATIONS TO BLACK FOR WINNING THE GAME----")

    env.close()

def auto_game():
    # Make Environment
    env = OthelloEnvironment(render_mode = "human")
    env.reset()
    legal_moves = env.get_legal_moves(return_as="list")

    for i in range(64):

        if(len(legal_moves) == 0):
            break

        (_, _), _, terminated, _, info = env.step( legal_moves[np.random.choice(len(legal_moves))])

        if(terminated):
            break

        legal_moves = info['legal_moves']  
        env.render()

    env.render()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'human':
            human_game()
            quit(0)
    
    auto_game()