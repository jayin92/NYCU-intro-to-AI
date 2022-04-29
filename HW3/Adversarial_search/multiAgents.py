# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent, Actions

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        actions = gameState.getLegalActions(0) # Get Legal Action of pacman (pacman's index is 0)
        candidates = [] # Initialize a list to track legal action and its score for the first max layer.
        for action in actions: # Iterate all possible action
            candidates.append((action, self.minimax(gameState.getNextState(0, action), self.depth-1, 1, False))) # Call recursive function self.minimax
        action, _ = max(candidates, key=lambda item: item[1][1]) # Get the action with the highest score
        return action # return that action
        # End your code
    def minimax(self, gameState, depth, agentIdx, maximize):
        if gameState.isWin() or gameState.isLose() or (depth == 0 and agentIdx == 0): # If current game state is terminal state
            return (gameState, self.evaluationFunction(gameState)) # return (state, score) pair
        actions = gameState.getLegalActions(agentIdx) # Get legal action of a character (pacman or ghosts)
        candidates = [] # Initialize a list to track legal action and its corresponding score
        if maximize: # If current layer is a max layer
            # Becuase current layer is a max layer, the next layer will be a min layer with the first ghost whose index equals to 1.
            for action in actions:
                candidates.append(self.minimax(gameState.getNextState(agentIdx, action), depth-1, 1, False))
            stateScore = max(candidates, key=lambda item: item[1]) # Max layer, take max over the candidates' score
            
        elif agentIdx < gameState.getNumAgents()-1: # If current layer is a min layer, and current ghost is not the last ghosts
            # Because current layer is a min layer and current ghost is not the last ghost, the next layer will still a min layer with a ghost whose index is the current index + 1
            for action in actions:
                candidates.append(self.minimax(gameState.getNextState(agentIdx, action), depth, agentIdx+1, False))
            stateScore = min(candidates, key=lambda item: item[1]) # Min layer, take min over the candidates' score
        else: # If current layer is a min layer, and current ghost is the last ghost
            # Current ghost is the last ghost, the next layer will a max layer with the pacman, whose index eqauls to 1
            for action in actions:
                candidates.append(self.minimax(gameState.getNextState(agentIdx, action), depth, 0, True))
            stateScore = min(candidates, key=lambda item: item[1]) # Min layer, take min over the candidates' score
        
        return stateScore # Return (state, score) pair
            
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """
    def expectimax(self, gameState, depth, agentIdx, maximize):
        if gameState.isWin() or gameState.isLose() or (depth == 0 and agentIdx == 0):
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(agentIdx)
        candidates = []
        if maximize:
            for action in actions:
                candidates.append(self.expectimax(gameState.getNextState(agentIdx, action), depth-1, 1, False))
            score = max(candidates)
        elif agentIdx < gameState.getNumAgents()-1:
            tmp = 0
            for action in actions:
                tmp += (self.expectimax(gameState.getNextState(agentIdx, action), depth, agentIdx+1, False))
            score = tmp / len(actions)
        else:
            tmp = 0
            for action in actions:
                tmp += (self.expectimax(gameState.getNextState(agentIdx, action), depth, 0, True))
            score = tmp / len(actions)
           
        return score 
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing :uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        actions = gameState.getLegalActions(0)
        candidates = []
        for action in actions:
            candidates.append((action, self.expectimax(gameState.getNextState(0, action), self.depth-1, 1, False)))
        action,  _= max(candidates, key=lambda item: item[1])
        return action
        # End your code

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Begin your code
    if currentGameState.isLose():
        return -1e20
    elif currentGameState.isWin():
        return 1e20
    score = currentGameState.getScore()
    cnt_food = currentGameState.getNumFood()
    cnt_cap  = len(currentGameState.getCapsules())
    dis = closestFood(currentGameState.getPacmanPosition(), currentGameState.getFood(), currentGameState.getWalls())
    val = 1 * score
    if dis is not None:
        val -= 10 * dis
    val -= cnt_food * 100
    val -= 30 * cnt_cap
    return val
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = betterEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
