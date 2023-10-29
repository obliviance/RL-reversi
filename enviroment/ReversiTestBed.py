#testbed for enviroment demo, mirrors the a2 enviroment testbed
from ReversiHelpers import OthelloEnvironment

def _plotGameBoard():
    env = OthelloEnvironment()
    #do something to the enviroment

    #loop to render the board
    while not env.end_render():
        env.render()

    env.close()
    
if __name__ == "__main__":
    _plotGameBoard()