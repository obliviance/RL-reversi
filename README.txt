Cameron Humphreys 101162528 
Lauris Petlah 101156789
Sukhrobjon Eshmirzaev 101169793
Awwab Mahdi 101225637

Installation.
    run `pip install -r requirements`

Compare Players.
    run `python -m comparison.ComparePlayers`
    usage: ComparePlayers [-h] [--games GAMES] [--verbosity {0,1,2}] [--human-render] p1 p2

    Compare players against one another

    positional arguments:
    p1
    p2

    options:
    -h, --help           show this help message and exit
    --games GAMES
    --verbosity {0,1,2}
    --human-render

    Available players are:
        dqn1
        dqn1-rs
        dqn1-selfplay
        dqn1-rs-selfplay
        dqn2
        dqn2-rs
        dqn2-selfplay
        dqn2-rs-selfplay
        dsqn
        dsqn-selfplay
        mcts
        random
        human


Train DQN
    run `python -m dqn.train <model_name>`
    usage: train [-h] [-a ALPHA] [-g GAMMA] [-e EPSILON] [--episodes EPISODES] [--replay REPLAY] [--selfplay] [--reward-shaping] [--dqn2]
             [--evaluate EVALUATE]
             model_name
    Trains a DQN

    positional arguments:
    model_name            Name of Model. Used as filename and outputted chart title

    options:
    -h, --help            show this help message and exit
    -a ALPHA, --alpha ALPHA
                            Learning Rate. Default=0.001
    -g GAMMA, --gamma GAMMA
                            Discount Factor. Default=0.1
    -e EPSILON, --epsilon EPSILON
                            Randomness in Policy. (Decreases to zero as you approach final episode). Default=0.1
    --episodes EPISODES   Number of training episodes. Default=25
    --replay REPLAY       Size of episode replay buffer. Default=4096
    --selfplay            Train with previous iteration of agent
    --reward-shaping      Add additional per step reward based on current board
    --dqn2                Train with Neural Network with IN:(State) -> OUT:(Actions x Q(State, Actions)) [ OMIT to have IN:(State + Action) ->
                            OUT:(Q(State,Action))]
    --evaluate EVALUATE   Number of evaluation episodes. Default=50

Train MCTS
    run `python -m mcts.train <model_name>`
    usage: train [-h] [--evaluate EVALUATE] model_name

    Trains a MonteCarlo

    positional arguments:
    model_name

    options:
    --evaluate EVALUATE