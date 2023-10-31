#testbed for enviroment demo, mirrors the a2 enviroment testbed
from ReversiHelpers import *


def _plotGameBoard():
    render_mode = "human" # or "human" to engage human play
    env = OthelloEnvironment(render_mode = render_mode)
    #do something to the enviroment
    if (render_mode == "human"):
        terminated = False
        while(not terminated):
            #render the enviroment to show the player the state and the actions they can take
            env.render()
            if(len(env.get_legal_moves()) == 0):
                break
            input_string = input("Please choose a position to place your piece, in the form 'X Y': ")
            input_string = ''.join(c for c in input_string if c.isdigit())
            action = tuple(map(int, input_string))
            inputFlag = False
            if (len(action) != 2 or (action[0]>=env.length or action [1] >= env.length) or (action[0] < 0 or action [1] < 0)):
                inputFlag = True
            else:
                (_, _), _, terminated, _, info= env.step(action)
            while(inputFlag or 'error' in info.keys()):
                #user typed an unallowed action, try again#this wont work with just strings...
                print("That was not a legal move, try again", inputFlag, "\n")
                input_string = input("Please choose a valid position to place your piece, in the form 'X Y': ")
                input_string = ''.join(c for c in input_string if c.isdigit())
                action = tuple(map(int, input_string))
                if (len(action) != 2 or (action[0]>=env.length or action [1] >= env.length) or (action[0] < 0 or action [1] < 0)):
                    inputFlag = True
                else: 
                    inputFlag = False 
                    (_, _), _, terminated, _, info= env.step(action)
        #game over
        if (env.get_winner() == DISK_WHITE):
            print("----CONGRATULATIONS TO WHITE FOR WINNING THE GAME----")
        if (env.get_winner() == DISK_BLACK):
            print("----CONGRATULATIONS TO BLACK FOR WINNING THE GAME----")
    env.close()

if __name__ == "__main__":
    _plotGameBoard()
