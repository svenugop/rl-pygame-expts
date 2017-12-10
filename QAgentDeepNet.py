import util
import random
import cv2
import numpy as np
import math

class QLearningAgent():

    def __init__(self):
        self.x = 60
        self.y = 60

        self.qVals = util.Counter()

        self.state = 'origin'

        self.memory = []

        self.discount = 0.05
        self.alpha = 0.7
        self.epsilon = 0.5

    def getPosition(self):
        return np.array([self.x,self.y])

    def followMe(self, leaderPos):

        # Get all possible actions to take from this state
        actions = self.getPossibleActions()

        currTargetDistance = self.getPosition() - leaderPos
        currTargetDistance = math.sqrt(currTargetDistance[0]*currTargetDistance[0] + currTargetDistance[1]*currTargetDistance[1])
        print("currTargetDistance = ", currTargetDistance)

        # With some randomness either choose a random action or the action that results in the maximum Q value (exploration vs. exploitation)
        if (util.flipCoin(self.epsilon)):
            action = random.choice(actions)
        else:
            # @todo -- write another version of computeActionFromQValues that uses an FC layer in tensorflow to compute Q(s,a)
            # @todo -- feed the frame(?) as an input to a CNN layer to decide Q(s,a)
            action = self.computeActionFromQValues(self.state)


        if (action == 'move-left'):
            self.x -= 5
        elif (action == 'move-right'):
            self.x += 5
        elif (action == 'move-up'):
            self.y -= 5
        elif (action == 'move-down'):
            self.y += 5

        nextTargetDistance = self.getPosition() - leaderPos
        nextTargetDistance = math.sqrt(nextTargetDistance[0]*nextTargetDistance[0] + nextTargetDistance[1]*nextTargetDistance[1])
        print("nextTargetDistance = ", nextTargetDistance)

        if (nextTargetDistance < currTargetDistance):
            nextState = 'closer'
        else:
            nextState = 'farther'


        # Compute reward for taking that action
        reward = self.getReward(self.state, nextState)
        print("Received reward ", reward)
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

    def computeActionFromQValues(self, state, mode):
        """
          Compute the best action to take in a state.
          This function uses a neural network to approximate the Q(s,a) function 
        """
        action = 'move-right'

        # The network will have an input layer --> a single convolutional layer --> followed by a FC layer (for classification)

        # The input is an image of size 1000 x 1000
        # @todo: consider resizing or cropping to reduce input dimension
        # - the second argument is [batch_size (-1 if batch size to be dynamically computed, input_width, input_height, input_channels)]
        input_layer = tf.reshape(state, [-1, 1000, 1000, 1])        

        # Convolutional layer which applies 5 filters of kernel size 5x5
        # -- padding = "same" so that the input is 0 padded on all sides to get an output of same size as input i.e. 1000x1000
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=5,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # A max pooling layer to reduce the dimensions of the input to FC layer
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[10, 10], strides=2)

        
        # An FC layer for classification
        # -- the input will be a flattened version of the 2d output of the previous version
        pool1_flat = tf.reshape(pool2, [-1, 100 * 100 * 5])

        dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(inputs=dense, 
                                    rate=0.4, 
                                    training=mode == tf.estimator.ModeKeys.TRAIN)
        # Logits Layer
        # -- 4 outputs corresponding to each action
        logits = tf.layers.dense(inputs=dropout, units=4)

        # @todo: Return the logits or the softmax outputs from this function; handle argmax, loss and accuracy prediction outside this function

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # if predict mode, return prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


        # if TRAIN or EVAL mode do the following
        # Calculate Loss (for both TRAIN and EVAL modes)
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          eval_metric_ops=eval_metric_ops)




        # legalActions = self.getPossibleActions()
        # q_action_pair = []

        # action = None

        # if len(legalActions) > 0:
        #     for lAction in legalActions:
        #         qVal = self.getQValue(state, lAction)
        #         q_action_pair.append((qVal, lAction))

        #     print(q_action_pair)
        #     bestActions = [pair for pair in q_action_pair if pair == max(q_action_pair)]
        #     # In case of tie, break it randomly
        #     bestActionPair = random.choice(bestActions)

        #     action = bestActionPair[1]

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

        if nextState == 'closer':
            reward = 10
        else:
            reward= -5

        return reward

    def updatePosition(self, leaderPos, gameScreenDims):

        colorBGR = (255,128,0)
        selfColorBGR = (0,128,255)

        currState = np.zeros(gameScreenDims ,np.uint8)
        x = leaderPos[0]
        y = leaderPos[1]

        ## Draw the leader rect
        cv2.rectangle(currState,(x,y),(x+20,y+20), (255,128,0),-1)
        ## Draw the self rect
        (sx, sy) = self.getPosition()
        cv2.rectangle(currState,(sx,sy),(sx+20,sy+20),selfColorBGR,-1)

        self.state = currState

        # Get all possible actions to take from this state
        actions = self.getPossibleActions()

        # With some randomness either choose a random action or the action that results in the maximum Q value (exploration vs. exploitation)
        if (util.flipCoin(self.epsilon)):
            action = random.choice(actions)
        else:
            # @todo -- write another version of computeActionFromQValues that uses an FC layer in tensorflow to compute Q(s,a)
            # @todo -- feed the frame(?) as an input to a CNN layer to decide Q(s,a)
            action = self.computeActionFromQValues(self.state)

        if (action == 'move-left'):
            self.x -= 5
        elif (action == 'move-right'):
            self.x += 5
        elif (action == 'move-up'):
            self.y -= 5
        elif (action == 'move-down'):
            self.y += 5


        nextState = np.zeros(gameScreenDims ,np.uint8)
        x = leaderPos[0]
        y = leaderPos[1]

        ## Draw the leader rect
        cv2.rectangle(nextState,(x,y),(x+20,y+20), (255,128,0),-1)
        ## Draw the self rect
        (sx, sy) = self.getPosition()
        cv2.rectangle(nextState,(sx,sy),(sx+20,sy+20),selfColorBGR,-1)
        

        # Compute reward for taking that action
        reward = self.getReward('farther', 'closer')
        # reward = self.getReward(self.state, nextState)
        print "Received reward {}".format(reward)

        newMemoryElement = (currState, nextState, action, reward)
        self.memory.append(newMemoryElement)

        print self.memory[-1][2]

        cv2.imshow("Test", self.memory[-1][0])
        cv2.waitKey()


    def runTrainingStep(self):
        pass
        # # Add the current game state to the replay memory buffer
        # replayBuffer.addToBuffer(gameState)

        # replayBuffer.runTraining()
