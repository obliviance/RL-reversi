#testbed for enviroment demo, mirrors the a2 enviroment testbed
from ReversiHelpers import gameBoard

def _plotGameBoard():
    env = gameBoard()
    #do something to the enviroment

    #loop to render the board
    while True:
        env.render()

    env.close()
    
if __name__ == "__main__":
    _plotGameBoard()