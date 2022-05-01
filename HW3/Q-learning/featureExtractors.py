# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    This function is actually from the q-learning part.
    It uses BFS to find the closest food.
    """
    fringe = [(pos[0], pos[1], 0)] # Initalize a list. This list will be later used as a queue.
    expanded = set() # Initialize a set to track the visited positions
    while fringe: # While queue is not empty
        pos_x, pos_y, dist = fringe.pop(0) # Pop the first element in queue
        if (pos_x, pos_y) in expanded: # If this position has already visited
            continue
        expanded.add((pos_x, pos_y)) # Add current postion to visited set
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        capsules = state.getCapsules()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        features = util.Counter()
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away which is not in scared status
        # We can use state.data.agentStates[i+1].scaredTimer to determine a ghost is scared now. If scaredTimer == 0, then this ghost is not scared. Otherwise, it's scared now.
        # Pacman can eat these scared ghosts to get a higher score
        features["#-of-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g, walls) and state.data.agentStates[i+1].scaredTimer == 0) for i, g in enumerate(ghosts))
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        features["cnt-food"] = state.getNumFood() / 200 # Get total number of remaining food

        dist = closestFood((next_x, next_y), food, walls) # Get closest food using BFS
        dist_cap = None # Distance of closest capsule
        dist_scared = None # Distance of closest scared ghost

        if len(capsules) != 0: # If has remaining capsules
            # Using Manhattan distance to evaluate closest capsules
            # Using BFS here will make training process really slow
            dist_cap = abs(next_x-capsules[0][0]) + abs(next_y-capsules[0][1])
            for cap in capsules[1:]:
                dist_cap = min(dist_cap, abs(next_x-cap[0]) + abs(next_y-cap[1]))

        for i in range(1, len(ghosts)):
            if state.data.agentStates[i].scaredTimer != 0:
                # If this ghost is scared now
                # Using Manhattan distance to evaluate closest ghosts
                if dist_scared == None:
                    dist_scared = abs(next_x-ghosts[i-1][0]) + abs(next_y-ghosts[i-1][1])
                else:
                    dist_scared = min(dist_scared, abs(next_x-ghosts[i-1][0]) + abs(next_y-ghosts[i-1][1])) 

        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            # Using different weight 2.5
            features["closest-food"] = float(dist) / (walls.width * walls.height) * 2.5

        if dist_cap is not None and dist_scared is None:
            # Using different weight 10
            features["closet-capsule"] = float(dist_cap) / (walls.width * walls.height) * 10

        if dist_scared is not None:
            # Using different weight 1
            features["closet-scared"] = float(dist_scared) / (walls.width * walls.height)

        features.divideAll(10.0) # Divide all features value with 10.0
        return features
