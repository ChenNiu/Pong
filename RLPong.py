import tensorflow as tf
import cv2  
import pygame
import numpy as np  
import random  
from collections import deque  


# the game PONG

# size of everything
WinW = 100
WinH = 100

PadW = 1
PadH = 4
edge = 0

BallW = 2
BallH = 2

PadSpeed = 1
BallSpeedX = 1
BallSpeedY = 1


# init screen
screen = pygame.display.set_mode((WinW, WinH))


def drawBall(x, y):
    ball = pygame.Rect(x, y, BallW, BallH)
    pygame.draw.rect(screen, (0, 0, 0), ball)


def drawPadLeft(y):
    paddle1 = pygame.Rect(edge, y, PadW, PadH)
    pygame.draw.rect(screen, (0, 0, 0), paddle1)


def drawPadRight(y):
    paddle2 = pygame.Rect(WinW - edge - PadW, y, PadW, PadH)
    pygame.draw.rect(screen, (0, 0, 0), paddle2)


def updateBall(p1y, p2y, bx, by, bxDir, byDir):
    # update the x and y position
    bx = bx + bxDir * BallSpeedX
    by = by + byDir * BallSpeedY
    score = 0

    # if hit the left side 
    if (bx <= edge + PadW and by + BallH >= p1y and by - BallH <= p1y + PadH):
        bxDir = 1
    elif (bx <= 0):
        bxDir = 1
        score = -1
        return [score, p1y, p2y, bx, by, bxDir, byDir]

    # if hit right side
    if (bx >= WinW - PadW - edge and by + BallH >= p2y and by - BallH <= p2y + PadH):
        bxDir = -1
    elif (bx >= WinW - BallW):
        bxDir = -1
        score = 1
        return [score, p1y, p2y, bx, by, bxDir, byDir]

    # hit top and bottom
    if (by <= 0):
        by = 0;
        byDir = 1;
    elif (by >= WinH - BallH):
        by = WinH - BallH
        byDir = -1
    return [score, p1y, p2y, bx, by, bxDir, byDir]


def updatePadLeft(action, p1y):
    # up
    if (action[1] == 1):
        p1y = p1y - PadSpeed
    # down
    if (action[2] == 1):
        p1y = p1y + PadSpeed

    # keep in screen
    if (p1y < 0):
        p1y = 0
    if (p1y > WinH - PadH):
        p1y = WinH - PadH
    return p1y


def updatePadRight(p2y, by):
    if (p2y + PadH / 2 < by + BallH / 2):
        p2y = p2y + PadSpeed
    if (p2y + PadH / 2 > by + BallH / 2):
        p2y = p2y - PadSpeed
    if (p2y < 0):
        p2y = 0
    if (p2y > WinH - PadH):
        p2y = WinH - PadH
    return p2y


class Pong:
    def __init__(self):
        self.scoreBoard = 0
        self.p1y = WinH / 2 - PadH / 2
        self.p2y = WinH / 2 - PadH / 2
        self.bxDir = 1
        self.byDir = 1
        # starting point
        self.bxPos = WinW / 2 - BallW / 2

        self.byPos = random.randint(0, 4) * (WinH - BallH) / 4


    def getFrame(self):
        pygame.event.pump()
        screen.fill((255, 255, 255))
        drawPadLeft(self.p1y)
        drawPadRight(self.p2y)
        drawBall(self.bxPos, self.byPos)
        mat = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return mat

    # update screen
    def updateFrame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill((255, 255, 255))
        self.p1y = updatePadLeft(action, self.p1y)
        drawPadLeft(self.p1y)
        self.p2y = updatePadRight(self.p2y, self.byPos)
        drawPadRight(self.p2y)
        [score, self.p1y, self.p2y, self.bxPos, self.byPos, self.bxDir, self.bxDir] = updateBall(self.p1y, self.p2y, self.bxPos, self.byPos,self.bxDir, self.bxDir)
        drawBall(self.bxPos, self.byPos)
        mat = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        self.scoreBoard = self.scoreBoard + score
        return [score, mat]
		
		
		
		


numActions = 3  # up,down, stay
learningRate = 0.99
Ini_Eps = 1.0
End_Eps = 0.05
EXPLORE = 1000
OBSERVE = 100
REPLAY_MEMORY = 100000
BatchSize = 100


def convo_net():
    weights = {'W_conv1':tf.Variable(tf.zeros([5, 5, 4, 32])),
			   'W_conv2':tf.Variable(tf.zeros([5, 5, 32, 64])),
			   'W_fc3':tf.Variable(tf.zeros([3136, 1024])),
			   'out':tf.Variable(tf.zeros([1024, numActions]))}
    biases = {'b_conv1':tf.Variable(tf.zeros([32])),
			  'b_conv2':tf.Variable(tf.zeros([64])),
			  'b_fc3':tf.Variable(tf.zeros([1024])),
			  'out':tf.Variable(tf.zeros([numActions]))}

    x = tf.placeholder("float", [None, 50, 50, 4])

    conv1 = tf.nn.relu(tf.nn.conv2d(x, weights['W_conv1'], strides=[1, 1, 1, 1], padding="VALID") + biases['b_conv1'])

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['W_conv2'], strides=[1, 1, 1, 1], padding="VALID") + biases['b_conv2'])

    conv2_flat = tf.reshape(conv2, [-1, 3136])

    fc3 = tf.nn.relu(tf.matmul(conv2_flat, weights['W_fc3']) + biases['b_fc3'])

    output = tf.matmul(fc3, weights['out']) + biases['out']

    return x, output


def train_the_net(x, y, sess):
    argmax = tf.placeholder("float", [None, numActions])
    groundTruth = tf.placeholder("float", [None])  

    action = tf.reduce_sum(tf.mul(y, argmax), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(action - groundTruth))
    trainStep = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # init game
    game = Pong()
	
	# experiences
    Q = deque()

    frame = game.getFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (50, 50)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    inputTensor = np.stack((frame, frame, frame, frame), axis=2)

    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    iter = 0
    eps = Ini_Eps

    while (1):
        outputTensor = y.eval(feed_dict={x: [inputTensor]})[0]
        argmax_t = np.zeros([numActions])

        if (random.random() <= eps):
            maxIndex = random.randrange(numActions)
        else:
            maxIndex = np.argmax(outputTensor)
        argmax_t[maxIndex] = 1

        if eps > End_Eps:
            eps -= (Ini_Eps - End_Eps) / EXPLORE

        # reward 
        reward_t, frame = game.updateFrame(argmax_t)
		
        frame = cv2.cvtColor(cv2.resize(frame, (50, 50)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (50, 50, 1))
		
		
        inputTensor1 = np.append(frame, inputTensor[:, :, 0:3], axis=2)

        Q.append((inputTensor, argmax_t, reward_t, inputTensor1))

        if len(Q) > REPLAY_MEMORY:
            Q.popleft()

        if iter > OBSERVE:
            miniBatchSize = random.sample(Q, BatchSize)

            inputTensor_BatchSize = [q[0] for q in miniBatchSize]
            argmax_BatchSize = [q[1] for q in miniBatchSize]
            reward_BatchSize = [q[2] for q in miniBatchSize]
            inputTensor1_BatchSize = [q[3] for q in miniBatchSize]

            gt_BatchSize = []
            out_BatchSize = y.eval(feed_dict={x: inputTensor1_BatchSize})

            for i in range(0, len(miniBatchSize)):
                gt_BatchSize.append(reward_BatchSize[i] + learningRate * np.max(out_BatchSize[i]))

            # train
            trainStep.run(feed_dict={argmax: argmax_BatchSize,x: inputTensor_BatchSize})

        # update
        inputTensor = inputTensor1
        iter = iter + 1

        if iter % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=iter)

        print(
        "Iteration", iter, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(outputTensor), "/ Action", maxIndex)


def main():
    sess = tf.InteractiveSession()
    x, y = convo_net()
    train_the_net(x, y, sess)


if __name__ == "__main__":
    main()