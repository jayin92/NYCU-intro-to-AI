# qlearningAgents.py
# ------------------
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


from collections import defaultdict

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

"""
part 2-2 & part 2-3
"""

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Begin your code
        self.value = defaultdict(lambda: defaultdict(float)) # Use nested defaultdict to store q-value of given state and action as self.value[state][action]

        # End your code


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        return self.value[state][action] # Just return the corresponding q value
        # End your code


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        actions = self.getLegalActions(state) # Get all legal actions
        if len(actions) == 0: # If no legal actions
            return 0.0 # then return 0
        else:
            q_value = -1e9 # Initalize a variable to track the maximum q value of current game state
            for a in actions:
                q_value = max(q_value, self.getQValue(state, a)) # Get q-value of given state and action, and update the maximum q-value

        return q_value # return maximum q state
        # End your code

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        legalActions = self.getLegalActions(state) # Get all legal actions
        action = None # Initalize the variable to track optimal action
        "*** YOUR CODE HERE ***"
        # Begin your code
        if len(legalActions) != 0:
            q_value = -1e9 # Initalize a variable to track maximum q value
            for a in legalActions:
                if self.getQValue(state, a) > q_value: # Update maximum q value and the corresponding action
                    q_value = self.getQValue(state, a)
                    action = a
                 
        return action # return optimal action
        # End your code

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Begin your code
        if util.flipCoin(self.epsilon): # Implementation of epsilon greedy
            # Random sample
            if len(legalActions) != 0: # If have legal actions
                action = random.choice(legalActions) # then randomly select an action
        else:
            action = self.computeActionFromQValues(state) # Otherwise, use q value to get the optimal actions of current game state
            
        return action # return that action
        # End your code
        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # Use q-learning update formula to update Q(s, a)
        self.value[state][action] = (1 - self.alpha) * self.value[state][action] + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        # End your code

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


"""
part 2-4
"""

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        # get weights and feature
        featureVectors = self.featExtractor.getFeatures(state, action) # Get feature vectors (type = util.Counter()) using getFeatures(state, action)
        res = 0 # Initalize return value
        # Dot product of w * featureVector
        for feature in featureVectors:
            res += featureVectors[feature] * self.weights[feature]
        
        return res
        # End your code

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        featureVectors = self.featExtractor.getFeatures(state, action) # Get feature vectors (type = util.Counter()) using getFeatures(state, action)
        # Using ApproximateQLearningAgent's formula to update every weights that corrsponds to a specific feature
        for feature in featureVectors:
            correction = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
            self.weights[feature] = self.weights[feature] + self.alpha * correction * featureVectors[feature]
        # End your code


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
