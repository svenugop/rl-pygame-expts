import util
import random
import cv2
import numpy as np
import math
import tensorflow as tf

NUM_TRAINING_ELEMENTS = 10

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

    def QNetwork(self, state):
        # The network will have an input layer --> a single convolutional layer --> followed by a FC layer (for classification)

        # The input is an image of size 1000 x 1000
        # @todo: consider resizing or cropping to reduce input dimension
        # - the second argument is [batch_size (-1 if batch size to be dynamically computed, input_width, input_height, input_channels)]
        #   this allows us the option to pass in a 'batch' input during training and a single input during normal operation
        input_layer = tf.reshape(state, [-1, 50, 50, 1])        

        # Convolutional layer which applies 5 filters of kernel size 5x5
        # -- padding = "same" so that the input is 0 padded on all sides to get an output of same size as input i.e. 1000x1000
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=5,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu)

        # A max pooling layer to reduce the dimensions of the input to FC layer
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[5, 5], strides=2)

        
        # An FC layer for classification
        # -- the input will be a flattened version of the 2d output of the previous version
        pool1_flat = tf.reshape(pool1, [-1, 23 * 23 * 5])

        dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(inputs=dense, 
                                    rate=0.4, 
                                    training=True)

        # dropout = tf.layers.dropout(inputs=dense, 
        #                             rate=0.4, 
        #                             training= (mode == tf.estimator.ModeKeys.TRAIN))

        # Logits Layer
        # -- 4 outputs corresponding to each action
        logits = tf.layers.dense(inputs=dropout, units=4)

        return logits

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
          This function uses a neural network to approximate the Q(s,a) function 
        """
        legalActions = self.getPossibleActions()

        # @todo: handle argmax, loss and accuracy prediction somewhere else?
        logits = self.QNetwork(state)
        max_index = tf.argmax(input=logits, axis=1)
        action_probability = tf.nn.softmax(logits, name="softmax_tensor")

        init = tf.global_variables_initializer()

        action = []

        print ""
        print "** In computeActionFromQValues()** "
        print ""

        with tf.Session() as sess:
            sess.run(init)
            ind = max_index.eval()[0]
            action_probability.eval()

        print "max_index value = {}".format(ind)
        action = legalActions[ind]

        print "Chose action {}".format(action)
        # print action_probability

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

    def getCurrentStateAsImage(self,x,y,sx,sy):

        colorBGR = (255,255,255)
        selfColorBGR = (100,100,100)
        
        lx = 25
        ly = 25
        w = 5
        d = 15

        currState = np.zeros((50,50,1) ,np.float32)
        cv2.rectangle(currState,(lx,ly),(lx+w,ly+w), colorBGR, -1)

        x1 = lx
        y1 = ly

        if (sx < x):
            x1 = lx - d
        elif (sx > x):
            x1 = lx + d

            
        if (sy < y):
            y1 = ly - d
        elif(sy > y):
            y1 = ly + d

        cv2.rectangle(currState,(x1,y1),(x1+w,y1+w), selfColorBGR, -1)

        return currState

        
    def updatePosition(self, leaderPos, gameScreenDims):

        colorBGR = (255,255,255)
        selfColorBGR = (100,100,100)

        x = leaderPos[0]
        y = leaderPos[1]

        (sx, sy) = self.getPosition()

        # ## Draw the leader rect
        # cv2.rectangle(currState,(x,y),(x+20,y+20), colorBGR, -1)
        # ## Draw the self rect
        # cv2.rectangle(currState,(sx,sy),(sx+20,sy+20),selfColorBGR, -1)

        # Calling getCurrentStateAsImage() as a stop-gap instead of passing in the original image
        # -- passing in the original image causes GPU to run out of memory
        currState = self.getCurrentStateAsImage(x,y,sx,sy)
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


        # Compute reward for taking that action
        reward = self.getReward('farther', 'closer')
        # reward = self.getReward(self.state, nextState)
        print "Received reward {}".format(reward)

        #newMemoryElement = (currState, nextState, action, reward)
        newMemoryElement = (currState, action)
        self.memory.append(newMemoryElement)

        # cv2.imshow("Test", self.memory[-1][0])
        # cv2.waitKey()

        
    def getRandomReplayBuffer(self):
        memSubset = self.memory[:]

        random.shuffle(memSubset)
        replayBufferLength = np.min([NUM_TRAINING_ELEMENTS, len(memSubset)])

        # return the first N elements
        memSubset = memSubset[0:replayBufferLength]

        return memSubset

    def runTrainingStep(self):
        
        # Obtain a subset of samples from the memory buffer
        samples =  self.getRandomReplayBuffer()

        # Obtain the batch of states from samples
        batch_of_states = [sample[0] for sample in samples]
        batch_of_states = np.asarray(batch_of_states)

        # Use integer labels so that we can do one hot encoding
        labels = [];
        for sample in samples:
            if (sample[1] == 'move-left'):
                labels.append(0)

            if (sample[1] == 'move-right'):
                labels.append(1)

            if (sample[1] == 'move-up'):
                labels.append(2)

            if (sample[1] == 'move-down'):
                labels.append(3)


        print ""
        print "** In runTrainingStep() ** "
        print ""
        print "batch_of_states dims = {}".format(batch_of_states.shape[0])
        # perform the training step by passing through the QNetwork and computing loss
        logits = self.QNetwork(batch_of_states)

        # @todo: tensorflow needs the labels NOT in string format
        onehot_labels = tf.one_hot(indices=labels, depth=4)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            sess.run(train_op)

        # # @todo: run the training step within a session
        # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # # Add evaluation metrics (for EVAL mode)
        # eval_metric_ops = {
        #     "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

        # return tf.estimator.EstimatorSpec(mode=mode, 
        #                                   loss=loss, 
        #                                   eval_metric_ops=eval_metric_ops)

        
