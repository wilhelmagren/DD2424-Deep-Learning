import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.num_iter = 1000
        self.num_actions = 4
        self.gamma = 0.97
        self.initial_epsilon = 0.8
        self.final_epsilon = 0.001
        self.epsilon = 0.1
        self.batchsize = 1000
        self.buff_size = 100
        self.memory = []
        self.fc1 = nn.Linear(in_features=12, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Snake(object):
    def __init__(self, width, height):
        self.body = [[4, 4], [3, 4], [2, 4], [1, 4]]
        self.board = np.zeros((height, width), dtype=np.int32)
        self.food = [random.randint(0, height - 1), random.randint(0, width - 1)]
        self.__updatebitmap__()
        self.score = 0
        self.length = len(self.body)
        self.dir = (1, 0)
        self.momentum = 0
        self.movesincefood = 0

    def __len__(self):
        return self.length

    def __repr__(self):
        s = ''
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if self.board[x, y] == 2:
                    s += ' O'
                elif self.board[x, y] == 3:
                    s += ' +'
                elif self.board[x, y] == 1:
                    s += ' o'
                else:
                    s += ' .'
            s += '\n'
        return s

    def reset_game(self, bool):
        self.body = [[4, 4], [3, 4], [2, 4], [1, 4]]
        self.board = np.zeros((10, 10), dtype=np.int32)
        self.food = [random.randint(0, 10 - 1), random.randint(0, 10 - 1)]
        self.__updatebitmap__()
        self.score = 0
        self.length = len(self.body)
        self.dir = (1, 0)
        self.momentum = 0
        self.movesincefood = 0
        if bool:
            return self.serialize()

    def __updatebitmap__(self):
        self.board = np.zeros(self.board.shape)
        bitmaplist = self.body.copy()
        bitmaplist.append(self.food)
        headx, heady = 0, 0
        foodx, foody = 0, 0
        for idx, bodypiece in enumerate(bitmaplist):
            if self.__inborder__(bodypiece):
                if idx == 0:
                    # Head
                    headx, heady = bodypiece[1], bodypiece[0]
                elif idx == len(bitmaplist) - 1:
                    # Food
                    foodx, foody = bodypiece[1], bodypiece[0]
                else:
                    # Body
                    self.board[bodypiece[1], bodypiece[0]] = 1
        self.board[foodx, foody] = 3
        self.board[headx, heady] = 2

    def serialize(self):
        bitmap = torch.zeros((1, 12))
        foodx, foody = self.food
        xdx, ydx = self.body[0][0], self.body[0][1]
        bitmap[0, 0] = 1 if 0 > xdx - 1 or self.__check_col__(xdx, ydx, xdx - 1) else 0
        bitmap[0, 1] = 1 if xdx + 1 >= 10 or self.__check_col__(xdx, ydx, xdx + 1) else 0
        bitmap[0, 2] = 1 if 0 > ydx - 1 or self.__check_col__(xdx, ydx, ydx - 1) else 0
        bitmap[0, 3] = 1 if ydx + 1 >= 10 or self.__check_col__(xdx, ydx, ydx + 1) else 0
        bitmap[0, 4] = 1 if self.dir == (1, 0) else 0
        bitmap[0, 5] = 1 if self.dir == (-1, 0) else 0
        bitmap[0, 6] = 1 if self.dir == (0, 1) else 0
        bitmap[0, 7] = 1 if self.dir == (0, -1) else 0
        bitmap[0, 8] = 1 if foodx < xdx else 0
        bitmap[0, 9] = 1 if foodx > xdx else 0
        bitmap[0, 10] = 1 if foody < ydx else 0
        bitmap[0, 11] = 1 if foody > ydx else 0
        return bitmap

    def alive(self):
        head_x = self.body[0][0]
        head_y = self.body[0][1]
        for idx in range(1, len(self.body)):
            if head_x == self.body[idx][0] and head_y == self.body[idx][1]:
                return False
        return self.__inborder__(self.body[0])

    def __check_col__(self, posx, posy, check):
        for idx in range(1, len(self.body)):
            if posx == self.body[idx][0] and posy == self.body[idx][1]:
                return True
        return False

    def move_left(self):
        if self.momentum != 0:
            self.momentum = 1
            self.dir = (-1, 0)

    def move_right(self):
        if self.momentum != 1:
            self.momentum = 0
            self.dir = (1, 0)

    def move_up(self):
        if self.momentum != 3:
            self.momentum = 2
            self.dir = (0, -1)

    def move_down(self):
        if self.momentum != 2:
            self.momentum = 3
            self.dir = (0, 1)

    def update_pos(self):
        self.body.insert(0, self.body.pop())
        self.body[0][0] = self.body[1][0] + self.dir[0]
        self.body[0][1] = self.body[1][1] + self.dir[1]
        self.movesincefood += 1
        self.__updatebitmap__()

    def random_move(self):
        l = random.randint(0, 4)
        if l == 0:
            self.move_left()
        if l == 1:
            self.move_down()
        if l == 2:
            self.move_right()
        if l == 3:
            self.move_up()

    def collision_food(self):
        return self.body[0][0] == self.food[0] and self.body[0][1] == self.food[1]

    def got_food(self):
        self.score += 1
        self.movesincefood = 0
        self.food = [random.randint(0, 10 - 1), random.randint(0, 10 - 1)]
        if self.dir[1] == 0 and self.dir[0] == -1:
            self.body.append([self.body[-1][0], self.body[-1][1] + 1])
        if self.dir[1] == 0 and self.dir[0] == 1:
            self.body.append([self.body[-1][0], self.body[-1][1] - 1])
        if self.dir[1] == 1 and self.dir[0] == 0:
            self.body.append([self.body[-1][0] - 1, self.body[-1][1]])
        if self.dir[1] == -1 and self.dir[0] == 0:
            self.body.append([self.body[-1][0] + 1, self.body[-1][1]])
        self.length = len(self.body)

    def next_state(self, action):
        """
        Returns serialized state, reward, terminal
        """
        ac = torch.argmax(action)
        if ac == 0:
            self.move_left()
        elif ac == 1:
            self.move_up()
        elif ac == 2:
            self.move_right()
        elif ac == 3:
            self.move_down()
        else:
            print('wtf got more than 4 actions!!!')
        self.update_pos()
        new_to_food = math.sqrt((self.body[0][0] - self.food[0]) ** 2 + (self.body[0][1] - self.food[1]) ** 2)
        reward = 0.1
        # reward += 1 - new_to_food/5
        if self.collision_food():
            self.got_food()
            reward = 10
        terminal = not self.alive()
        if terminal:
            reward = -5
        return self.serialize(), torch.tensor(reward)[None], torch.tensor(terminal)[None]

    @staticmethod
    def __inborder__(coordinate):
        return 0 <= coordinate[0] <= 10 - 1 and 0 <= coordinate[1] <= 10 - 1


def RLtrain(starttime, episodes, max_steps=100):
    """
    RL,
        for every move not die => +0.1
        for every move_count >= 20 and not get food => -1
        for food get => +1
        for every die => -1

    Three possible actions,
        [1, 0, 0]: tilt the snake movement counter-clockwise
        [0, 1, 0]: don't change the movement
        [0, 0, 1]: tilt the snake movement clockwise
    """
    # weights = torch.load('./snake_126_50.pth', map_location=lambda storage, loc: storage)
    agent = SnakeNet()
    # agent.load_state_dict(weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.cuda()
    environment = Snake(10, 10)
    epsilon_decrements = np.linspace(agent.initial_epsilon, agent.final_epsilon, agent.num_iter)
    optimizer = optim.Adam(agent.parameters(), lr=0.003)
    scores = []
    softmax = nn.Softmax(dim=1)
    # initialize mean squared error loss
    criterion = nn.MSELoss()
    summary(agent, (1, 12))
    totstep = 0
    for ep in range(episodes):
        state = environment.reset_game(True)
        score = 0
        totloss = 0
        if ep % 50 == 0 and ep != 0:
            torch.save(agent.state_dict(), "./snake_{}.pth".format(ep))
        for step in range(agent.num_iter):
            initserialized_state = state.to(device)
            output = softmax(agent(initserialized_state.float()))
            action = torch.zeros(agent.num_actions, dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.to(device)

            # Epsilon exploration
            rand_action = random.random() <= agent.epsilon
            idx_action = torch.randint(agent.num_actions, torch.Size([]),
                                       dtype=torch.int16) if rand_action else torch.argmax(output)
            if torch.cuda.is_available():
                idx_action = idx_action.to(device)
            action[idx_action] = 1
            next_state, reward, terminal = environment.next_state(action)
            score += reward.item()
            agent.memory.append((state, action, reward, next_state, terminal))
            state = next_state
            if len(agent.memory) > agent.batchsize:
                agent.memory.pop(0)
            agent.epsilon = epsilon_decrements[step]

            # sample random batch
            batch = random.sample(agent.memory, min(len(agent.memory), agent.batchsize))

            statebatch = torch.cat(tuple(d[0] for d in batch), dim=0)
            actionbatch = torch.cat(tuple(d[1][None] for d in batch), dim=0)
            rewardbatch = torch.cat(tuple(d[2][None] for d in batch), dim=0)
            newstatebatch = torch.cat(tuple(d[3] for d in batch), dim=0)

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                statebatch = statebatch.to(device).float()
                actionbatch = actionbatch.to(device).float()
                rewardbatch = rewardbatch.to(device).float()
                newstatebatch = newstatebatch.to(device).float()
            newoutputbatch = softmax(agent(newstatebatch)).to(device)
            ybatch = torch.cat(tuple(rewardbatch[i] + agent.gamma*torch.max(newoutputbatch[i]) for i in range(len(agent.memory)))).to(device)
            # Get q-values for each action
            qvals = torch.sum(agent(statebatch)*actionbatch, dim=1).to(device)
            optimizer.zero_grad()
            loss = criterion(qvals, ybatch)
            loss.backward()
            totloss += loss.item()
            optimizer.step()
            totstep += 1
            if step % 50 == 0 and step != 0:
                print(' |+++! episode: {}, step: {},\ttime: {:.2f}s, score: {:.1f}, loss: {:.2f}'.format(ep, totstep, time.time() - starttime, score, totloss))
                torch.save(agent.state_dict(), "./snake_{}_{}.pth".format(ep, totstep))
            if terminal.item():
                print(' |---! episode: {}, step: {},\ttime: {:.2f}s, score: {:.1f}, loss: {:.2f}'.format(ep, totstep, time.time() - starttime, score, totloss))
                scores.append(score)
                break
    return scores


def makemove(state, agent):
    soft = torch.nn.Softmax(dim=1)
    output = soft(agent(state.float()))
    idx_action = torch.argmax(output)
    tt = torch.zeros(4, dtype=torch.int16)
    tt[idx_action] = 1
    return tt


if __name__ == '__main__':
    # """
    scoree = RLtrain(time.time(), 1000)
    print(scoree)
    """
    state = Snake(10, 10)
    agent = SnakeNet()
    weights = torch.load('./prettygood_semiold_model.pth', map_location=lambda storage, loc: storage)
    agent.load_state_dict(weights)
    while state.alive():
        print(state)
        serializedstate = state.serialize()
        action = makemove(serializedstate, agent)
        _, _, _ = state.next_state(action)
        time.sleep(0.1)
    print("LOOOOSER! score: {}".format(state.score))
    """
