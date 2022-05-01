# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


"""
part 2-1
"""

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Begin your code
        for _ in range(1, self.iterations+1): #Run self.iteration times of value iteration algorithm
            previous_value = self.values.copy() # Copy old self.values to avoid overwriting problem when updating the current self.values
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): # If current state is terminal state, then set its value to 0
                    self.values[state] = 0
                    continue
                maxi_value = -1e9 # Initalize maxi_value to a small value, this variable will track the value of all possible actions
                for action in self.mdp.getPossibleActions(state): # Iterate all possible action
                    sumOfAllState = 0
                    for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action): # Get all the possible state and their correspoding probability
                        # Use the main formula of value iteration method to update self.values
                        # Noticed that I use previous_value to compute the correct update value
                        sumOfAllState += prob * (self.mdp.getReward(state, action, nextState) + self.discount*previous_value[nextState]) 
                    maxi_value = max(maxi_value, sumOfAllState) # Taking max over the corresponding value of all possible state
                self.values[state] = maxi_value # set self.values to the maximum possible value
                
        # End your code


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        res = 0 # Initialize q value
        for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action): # Get all teh possible state and their correspoding probability
            res += prob * (self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState]) # Compute q-value using formula
        return res # return q value
        # End your code

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        #check for terminal
        if self.mdp.isTerminal(state): # If this state is a terminal state, then agents can't move. Therefore, return None
            return None
        actions = self.mdp.getPossibleActions(state) # Otherwise, get all possible actions
        qValues = util.Counter() # Initailze a Counter to track every q value after taking action
        for action in actions:
            qValues[action] = self.getQValue(state, action) # Using getQValue to get q-value after taking this action

        return qValues.argMax() # argMax will return the key (which is action in this function) that has the highest q-value
        
        # End your code

    def getPolicy(self, state):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        return self.computeQValueFromValues(state, action)
