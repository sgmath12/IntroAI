# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a `graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    2019/09/12
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH


    start_state = problem.getStartState()
    myStack = util.Stack()
    myStack.push(start_state)
    parents = {}
    path = []
    Visited = {}

    while(myStack.isEmpty() is False):
        
        cur_state = myStack.pop()
        Visited[cur_state] = True
        if problem.isGoalState(cur_state):
            break
        #Successors = list of (state,action,cost)
        Successors = problem.getSuccessors(cur_state)
        for succesor in Successors:
            if succesor[0] not in Visited:
                myStack.push(succesor[0])
                parents[succesor[0]] = (cur_state,succesor[1])
    
    while (cur_state != start_state):
        path.append(parents[cur_state][1])
        cur_state = parents[cur_state][0]

    return path[::-1]
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH


    start_state = problem.getStartState()
    myQueue = util.Queue()
    myQueue.push(start_state)
    parents = {}
    path = []
    Visited = {}
    
    
    while(myQueue.isEmpty() is False):
        
        cur_state = myQueue.pop()
        Visited[cur_state] = True
        if problem.isGoalState(cur_state):
            break
        #Successors = list of (state,action,cost)
        Successors = problem.getSuccessors(cur_state)
        for succesor in Successors:
            if succesor[0] not in Visited:
                myQueue.push(succesor[0])
                parents[succesor[0]] = (cur_state,succesor[1])
    
    while (cur_state != start_state):
        path.append(parents[cur_state][1])
        cur_state = parents[cur_state][0]

    return path[::-1]
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH


    start_state = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    # 1 : cost.
    # item : state, cost
    myPriorityQueue.push(start_state,0)
    parents = {}
    cumulative_cost = {}
    path = []
    Visited = {}
    cumulative_cost[start_state] = 0
    
    while(myPriorityQueue.isEmpty() is False):
        cur_state = myPriorityQueue.pop()
        cur_cost = cumulative_cost[cur_state]
        Visited[cur_state] = True
        if problem.isGoalState(cur_state):
            break
        #Successors = list of (state,action,cost)
        Successors = problem.getSuccessors(cur_state)
        for succesor in Successors:
            if succesor[0] not in Visited:
                myPriorityQueue.update(succesor[0],cur_cost + succesor[2])
                parents[succesor[0]] = (cur_state,succesor[1])
                cumulative_cost[succesor[0]] = cur_cost + succesor[2]
    
    while (cur_state != start_state):
        path.append(parents[cur_state][1])
        cur_state = parents[cur_state][0]

    return path[::-1]
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    #from searchAgents import manhattanHeuristic
       
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH


    start_state = problem.getStartState()
    myPriorityQueue = util.PriorityQueue()
    # 1 : cost.
    myPriorityQueue.push(start_state,1)
    parents = {}
    path = []
    Visited = {}
    
    print heuristic
    while(myPriorityQueue.isEmpty() is False):
        cur_state = myPriorityQueue.pop()
        Visited[cur_state] = True
        if problem.isGoalState(cur_state):
            break
        #Successors = list of (state,action,cost)
        Successors = problem.getSuccessors(cur_state)
        for succesor in Successors:
            if succesor[0] not in Visited:
                myPriorityQueue.update(succesor[0],succesor[2] + heuristic(cur_state,problem))
                parents[succesor[0]] = (cur_state,succesor[1])
    
    while (cur_state != start_state):
        path.append(parents[cur_state][1])
        cur_state = parents[cur_state][0]

    return path[::-1]

    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch