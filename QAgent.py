import util
import random
import numpy as np

class QLearningAgent():

    def __init__(self):
        self.x = 60
        self.y = 60

        self.qVals = util.Counter()
        self.state = [-20, -20]

        self.discount = 0.05
        self.alpha = 0.7
        self.epsilon = 0.1

    def getPosition(self):
        return np.array([self.x,self.y])

    def followMe(self, leaderPos):

        # Get all possible actions to take from this state
        actions = self.getPossibleActions()

        self.state = self.getPosition() - leaderPos
        # With some randomness either choose a random action or the action that results in the maximum Q value (exploration vs. exploitation)
        if (util.flipCoin(self.epsilon)):
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(self.state)


        if (action == 'move-left'):
            self.x -= 1
        elif (action == 'move-right'):
            self.x += 1
        elif (action == 'move-up'):
            self.y -= 1
        elif (action == 'move-down'):
            self.y += 1


        nextState = self.getPosition() - leaderPos

        # Compute reward for taking that action
        reward = self.getReward(self.state, nextState)
        print "Received reward {}".format(reward)
        # Update Q-value estimate of the state you were in, based on received reward


        self.updateQValue(self.state, action, nextState, reward)
        self.state = nextState

    def getPossibleActions(self):
        """
          Returns possible actions
          for the states in the
          current state
        """

        actions = list()

        # move left
        actions.append('move-left')
        # move right
        actions.append('move-right')
        # move up
        actions.append('move-up')
        # move down
        actions.append('move-down')

        return actions

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        legalActions = self.getPossibleActions()
        q_action_pair = []

        action = None

        if len(legalActions) > 0:
            for lAction in legalActions:
                qVal = self.getQValue(state, lAction)
                q_action_pair.append((qVal, lAction))

            print q_action_pair
            bestActions = [pair for pair in q_action_pair if pair == max(q_action_pair)]
            # In case of tie, break it randomly
            bestActionPair = random.choice(bestActions)

            action = bestActionPair[1]

        return action


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        legalActions = self.getPossibleActions()

        qValList = []

        if len(legalActions) == 0:
            return 0.0

        for lAction in legalActions:
            qValList.append(self.getQValue(state, lAction))

        return max(qValList)


    def getQValue(self, state, action):
        return self.qVals[state[0], state[1], action]

    def updateQValue(self, state, action, newState, reward):

        currQVal = self.getQValue(state, action)
        sample = (reward + self.discount * self.computeValueFromQValues(newState))
        newQVal = currQVal + self.alpha * (sample - currQVal)

        self.qVals[state[0], state[1], action] = newQVal

    def getReward(self, state, nextState):
        winZone = abs(nextState[0]) < 11 and abs(nextState[1]) < 11
        aliveZone = (abs(nextState[0]) < abs(self.state[0])) or (abs(nextState[1]) < abs(self.state[1]))

        if (winZone):
            reward = 10
        elif (aliveZone):
            reward = 5
        else:
            reward = -10

        return reward