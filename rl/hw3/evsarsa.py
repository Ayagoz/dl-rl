# qlearningAgents.py
# ------------------
## based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random,math

import numpy as np
from collections import defaultdict

class EvSarsaAgent():
  """
    Q-Learning Agent

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate aka gamma)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
      - self.getQValue(state,action)
        which returns Q(state,action)
      - self.setQValue(state,action,value)
        which sets Q(state,action) := value
    
    !!!Important!!!
    NOTE: please avoid using self._qValues directly to make code cleaner
  """
  def __init__(self,alpha,epsilon,discount,getLegalActions):
    "We initialize agent and Q-values here."
    self.getLegalActions= getLegalActions
    self._qValues = defaultdict(lambda:defaultdict(lambda:0))
    self.alpha = alpha
    self.epsilon = epsilon
    self.discount = discount
    
  def getQValue(self, state, action):
    """
      Returns Q(state,action)
    """
    return self._qValues[state][action]

  def setQValue(self,state,action,value):
    """
      Sets the Qvalue for [state,action] to the given value
    """
    self._qValues[state][action] = value

#---------------------#start of your code#---------------------#

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.
    """
    
    possibleActions = self.getLegalActions(state)
    #If there are no legal actions, return 0.0
    if len(possibleActions) == 0:
    	return 0.0

    "*** YOUR CODE HERE ***"
    
    v = max([self.getQValue(state,action) for action in possibleActions])
    
    
    
    return v

  def getExpectedValue(self, state):
    possibleActions = self.getLegalActions(state)
    #If there are no legal actions, return 0.0
    if len(possibleActions) == 0:
    	return 0.0
    sum_a = np.sum([self.getQValue(state, a) for a in possibleActions])
    
    v = np.sum(self.epsilon*sum_a/len(possibleActions) + (1-self.epsilon)*self.getValue(state))
    
    return v
  
  def getPolicy(self, state):
    """
      Compute the best action to take in a state. 
      
    """
    possibleActions = self.getLegalActions(state)

    #If there are no legal actions, return None
    if len(possibleActions) == 0:
    	return None
    
    best_action = None

    "*** YOUR CODE HERE ***"
    
    
    
    best_action = max(possibleActions, key = lambda action: self.getQValue(state, action))
    
    
    return best_action

  def getAction(self, state):
    """
      Compute the action to take in the current state, including exploration.  
      
      With probability self.epsilon, we should take a random action.
      otherwise - the best policy action (self.getPolicy).

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)

    """
    
    # Pick Action
    possibleActions = self.getLegalActions(state)
    action = None
    
    #If there are no legal actions, return None
    if len(possibleActions) == 0:
    	return None

    #agent parameters:
    epsilon = self.epsilon

    "*** YOUR CODE HERE ***"
    
    if np.random.random() < epsilon:
        return np.random.choice(possibleActions)
    
    else:
        return self.getPolicy(state)
    
    return action

  def update(self, state, action, nextState, reward):
    """
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf


    """
    #agent parameters
    gamma = self.discount
    learning_rate = self.alpha
    
    "*** YOUR CODE HERE ***"    
    a2 = self.getPolicy(nextState)
    Qt1 = self.getExpectedValue(nextState)
    #<the "correct state value", uses reward and the value of next state>
    Qt = self.getQValue(state,action)
    updated_qvalue = Qt + learning_rate *(reward + gamma*Qt1 - Qt)
    self.setQValue(state,action,updated_qvalue)


#---------------------#end of your code#---------------------#


