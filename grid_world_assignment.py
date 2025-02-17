import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BOARD_ROWS = 5
BOARD_COLS = 5
START = (1, 0)
WIN_STATE = (4, 4)
JUMP_STATE = (3, 3)
OBSTACLE = [(3,2),(2,2),(2,3),(2,4)]
DETERMINISTIC = False

class Environment:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def rewardEvaluation(self, flag):
        if self.state == WIN_STATE and flag == "win":
            return 10
        elif self.state == WIN_STATE and flag == "jump":
            return 15
        else:
            return -1

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True

    def actionProb(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def nxtPos(self, action):
        if self.determine:
            if action == "north":
                nxtPos = (self.state[0] - 1, self.state[1])
            elif action == "south":
                nxtPos = (self.state[0] + 1, self.state[1])
            elif action == "west":
                nxtPos = (self.state[0], self.state[1] - 1)
            else:
                nxtPos = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self.actionProb(action)
            self.determine = True
            nxtPos = self.nxtPos(action)

        if 0 <= nxtPos[0] < BOARD_ROWS and 0 <= nxtPos[1] < BOARD_COLS:
            if nxtPos not in OBSTACLE:
                return nxtPos
            if nxtPos == (2,3) and action == "south":
                self.state = (3,3)
        return self.state

class Agent:
    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["north", "south", "west", "east"]
        self.Environment = Environment()
        self.isEnd = self.Environment.isEnd
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9
        self.Q_values = {}  # Q values for state-action pairs

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {a: 0 for a in self.actions}

        self.df = pd.DataFrame.from_dict(self.Q_values, orient='index')

    def chooseAction(self):
        mx_nxt_reward = -1
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # Greedy action selection
            for a in self.actions:
                current_position = self.Environment.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.Environment.nxtPos(action)
        return Environment(state=position)

    def reset(self):
        self.states = []
        self.Environment = Environment()
        self.isEnd = self.Environment.isEnd

    def trainAgent(self, rounds=100):
        i = 0
        CummulativeList = []
        cummulativeToken = False
        avg_list = []

        while (i < rounds and cummulativeToken == False):
            print("-----------------------------------------------> episode ", i)
            jumped = False

            if self.Environment.isEnd:
                if [(1, 3), "south"] in self.states:
                    reward = self.Environment.rewardEvaluation("jump")
                    jumped = True
                else:
                    reward = self.Environment.rewardEvaluation("win")
                rewardSum = reward
                CummulativeList.append(reward)

                for a in self.actions:
                    self.Q_values[self.Environment.state][a] = reward

                for s in reversed(self.states):
                    try:
                        current_q_value = self.Q_values[s[0]][s[1]]
                    except:
                        current_q_value = -1

                    if s[0] == (1, 3) and s[1] == "south":
                        reward = 5
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    else:
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                    rewardSum += reward
                average = rewardSum / len(self.states)
                avg_list.append(average)

                if CummulativeList[-30:].count(15) == 30:
                    self.df = pd.DataFrame.from_dict(self.Q_values, orient='index')
                    print(self.df)
                    print(CummulativeList[-30:])
                    cummulativeToken = True
                    break

                self.reset()
                i += 1
                if jumped:
                    print("Agent jumped during this episode.")
                else:
                    print("Agent did not jump during this episode.")
            else:
                action = self.chooseAction()
                self.states.append([(self.Environment.state), action])
                print("current position {} action {}".format(self.Environment.state, action))
                self.Environment = self.takeAction(action)
                self.Environment.isEndFunc()
                print("nxt state", self.Environment.state)
                print("---------------------")
                self.isEnd = self.Environment.isEnd

        # Plot average rewards over episodes
        plt.plot(avg_list)
        plt.show()

    # Function to display Q values on the grid
    def displayGrid(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(max(self.Q_values[(i, j)].values())).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

if __name__ == "__main__":
    agent = Agent()
    print("initial Q-values ... \n")
    print(agent.Q_values)
    print(agent.df.reset_index())
    agent.trainAgent(100)
    print("latest Q-values ... \n")
    print(agent.Q_values)
    agent.displayGrid()