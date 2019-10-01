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

# for debugging
import subprocess

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """

      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.

      It is opposite concept against planning agent.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        action \in legalMoves
        score After action
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        Walls = successorGameState.getWalls()

 
        "*** YOUR CODE HERE ***"
        # manhatten distance
        Ghost_distances = []
        Sign_Ghosts = []
        Foods_distance = []
        Foods = 0
        newScore = 0

        for i in range(newFood.width):
          for j in range(newFood.height):
            if newFood[i][j]:
              Foods_distance.append(abs(newPos[0] - i) + abs(newPos[1] - j))
              Foods += 1

        Foods_distance.sort()
        # float division

        # minus weight if ghost can attack, plus weight if pacman can eat scared ghost
        for i , ghost_state in enumerate(newGhostStates):
            newGhostPos = ghost_state.getPosition()
            Ghost_distances.append(abs(newPos[0] - newGhostPos[0]) + abs(newPos[1] - newGhostPos[1]))
            if Ghost_distances[i] <= newScaredTimes[i]:
              Sign_Ghosts.append(1)
            else :
              Sign_Ghosts.append(-1) 

        for i in range(len(newGhostStates)):
          if Ghost_distances[i] > 2 or Sign_Ghosts[i] == 1:
            pass
          else:
            return -10000

        newScore += (newFood.width*newFood.height-Foods) *(newFood.width+newFood.height)

        # find Nearset foods
        visited = {}
        dx = [-1,0,1,0]
        dy = [0,1,0,-1]
        Queue = [(newPos,0)]
        visited[newPos] = True
        head = 0
        
        tail = 1
        while(head < tail):
          cur,dist = Queue[head]
          head += 1

          if newFood[cur[0]][cur[1]]:
            break
          for d in range(4):
            next_cur = (cur[0]+dx[d],cur[1]+dy[d])
            if (Walls[next_cur[0]][next_cur[1]] == False and (next_cur not in visited)):
              visited[next_cur] = True
              Queue.append((next_cur,dist + 1))
              tail += 1
            
        if Foods > 0:
          newScore += -dist + (newFood.width + newFood.height)
        else :
          newScore += newFood.width + newFood.height

        return newScore

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
      Your minimax agent (question 2)
    """
    def maxValue(self,gameState,depth,agentNumber):
      v = -float('inf')

      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        next_value,_ = self.Value(succesorState,depth,1,action)
        if v < next_value:
          v = next_value
          max_action = action
    
      return v,max_action

    def minValue(self,gameState,depth,agentNumber):
      v = float('inf')

      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        if agentNumber < gameState.getNumAgents()-1:
          next_value,_ = self.Value(succesorState,depth,agentNumber+1,action)
        else:
          next_value,_ = self.Value(succesorState,depth-1,0,action)

        if v > next_value:
          v = next_value
          min_action = action

      return v,min_action
    
    def Value(self,gameState,depth,agentNumber,action = None):
      if gameState.isWin() or gameState.isLose():
        return gameState.getScore(),action
      
      if agentNumber == 0 and depth ==0:
        return gameState.getScore(),action
      elif agentNumber == 0 : # nextAgent is MAX
        return self.maxValue(gameState,depth,agentNumber)
      else: #nextAgent is MIN
        return self.minValue(gameState,depth,agentNumber)


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """   
        _,action = self.Value(gameState,self.depth,0)

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """


    def maxValue(self,gameState,depth,agentNumber,alpha,beta):
      v = -float('inf')
     
      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        next_value,_ = self.Value(succesorState,depth,1,alpha,beta,action)
        if v < next_value:
          v = next_value
          max_action = action

        if v > beta:
          return v,action

        alpha = max(alpha,v)

      return v,max_action

    def minValue(self,gameState,depth,agentNumber,alpha,beta):
      v = float('inf')

      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        if agentNumber < gameState.getNumAgents()-1:
          next_value,_ = self.Value(succesorState,depth,agentNumber+1,alpha,beta,action)
        else:
          next_value,_ = self.Value(succesorState,depth-1,0,alpha,beta,action)

        if v > next_value:
          v = next_value
          min_action = action

        if v < alpha:
          return v,action

        beta = min(beta,v)

      return v,min_action
    
    def Value(self,gameState,depth,agentNumber,alpha,beta,action = None,):

      if (agentNumber == 0 and depth ==0) or gameState.isWin() or gameState.isLose():
        return gameState.getScore(),action
      elif agentNumber == 0 : # nextAgent is MAX
        return self.maxValue(gameState,depth,agentNumber,alpha,beta)
      else: #nextAgent is MIN
        return self.minValue(gameState,depth,agentNumber,alpha,beta)


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = -alpha
        _,action = self.Value(gameState,self.depth,0,alpha,beta)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self,gameState,depth,agentNumber):
      v = -float('inf')

      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        next_value,_ = self.Value(succesorState,depth,1,action)
        if v < next_value:
          v = next_value
          max_action = action
    
      return v,max_action

    def expValue(self,gameState,depth,agentNumber):
      v = 0
      p = 1.0/len(gameState.getLegalActions(agentNumber))

      for action in gameState.getLegalActions(agentNumber):
        succesorState = gameState.generateSuccessor(agentNumber,action)
        if agentNumber < gameState.getNumAgents()-1:
          next_value,_ = self.Value(succesorState,depth,agentNumber+1,action)
        else:
          next_value,_ = self.Value(succesorState,depth-1,0,action)

        v += p*next_value

      return v,None
    
    def Value(self,gameState,depth,agentNumber,action = None):      
      if (agentNumber == 0 and depth ==0) or (gameState.isWin() or gameState.isLose()):
        return self.evaluationFunction(gameState),action
        #return gameState.getScore(),action
      elif agentNumber == 0 : # nextAgent is MAX
        return self.maxValue(gameState,depth,agentNumber)
      else: #nextAgent is MIN
        return self.expValue(gameState,depth,agentNumber)
    
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        _,action = self.Value(gameState,self.depth,0)
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Return Score
    """

    # Useful information you can extract from a GameState (pacman.py)
    #return currentGameState.getScore()

    '''
'generateSuccessor', 'getAndResetExplored', 'getCapsules', 'getFood',
 'getGhostPosition', 'getGhostPositions', 'getGhostState', 'getGhostStates', 
 'getLegalActions', 'getLegalPacmanActions', 'getNumAgents', 'getNumFood',
  'getPacmanPosition', 'getPacmanState', 'getScore',
 'getWalls', 'hasFood', 'hasWall', 'initialize', 'isLose', 'isWin
    '''

    #print (dir(currentGameState))
    Pos = currentGameState.getPacmanPosition()
    numFoods = currentGameState.getNumFood()
    Walls = currentGameState.getWalls()
    Foodgrid = currentGameState.getFood()
    FoodMinDistance = float('inf')
    # pass : .7, .2, .1
    w1,w2,w3= (.7,.2,.1)
    ghostDistances = 0
    ghostMinDistance = float('inf')

    for ghostAgentNumber in range(1,currentGameState.getNumAgents()):
        ghostPos = currentGameState.getGhostPosition(ghostAgentNumber)
        #if ghostDistances > manhattanDistance(Pos,ghostPos):
        ghostDistances += manhattanDistance(Pos,ghostPos)
        if ghostMinDistance > manhattanDistance(Pos,ghostPos):
          ghostMinDistance = manhattanDistance(Pos,ghostPos)


    visited = {}
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    Queue = [(Pos,0)]
    visited[Pos] = True
    head = 0
    
    tail = 1
    while(head < tail):
      cur,dist = Queue[head]
      head += 1

      if Foodgrid[cur[0]][cur[1]]:
        break
      for d in range(4):
        next_cur = (cur[0]+dx[d],cur[1]+dy[d])
        if (Walls[next_cur[0]][next_cur[1]] == False and (next_cur not in visited)):
          visited[next_cur] = True
          Queue.append((next_cur,dist + 1))
          tail += 1

    if ghostMinDistance == 0:
      return -1000

    score = 0
    score += (Foodgrid.width*Foodgrid.height-numFoods) *(Foodgrid.width+Foodgrid.height)
    if numFoods > 0:
      score += -dist + (Foodgrid.width + Foodgrid.height)
    else :
        score += Foodgrid.width + Foodgrid.height
    
    return w1*currentGameState.getScore() +w2*score + w3*(ghostDistances)

    
# Abbreviation
better = betterEvaluationFunction

