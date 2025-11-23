import copy
import random
from src.environment.blackjack import BlackjackEnv

STAND = 0
HIT = 1
DISCOUNT = 0.95  # This is the gamma value for all value calculations
WIN_STATE = (1,0,0)
DRAW_STATE = (0,0,0)
LOSE_STATE = (-1,0,0)

states = []
states.append(WIN_STATE)
states.append(DRAW_STATE)
states.append(LOSE_STATE)
for player_sum in range(2,22):
    for dealer_card in range(1,11):
        for usable_ace in [False, True]:
            s = (player_sum, dealer_card, usable_ace)
            states.append(s)

class QAgent:
    def __init__(self):

        # For Q-learning values
        self.Q_values = {}   # Dictionary: Store the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state for each action

        # Initialization of the values
        for s in states:
            self.Q_values[s] = [0,0]  # First element is the Q value of "Stand", second element is the Q value of "Hit"
            self.N_Q[s] = [0,0]  # First element is the number of visits of "Stand" at state s, second element is the Q value of "Hit" at s

        # Game environment
        self.env = BlackjackEnv()

    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)

    def Q_run(self, num_simulation, epsilon=0.4):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            state = self.env.reset()
            done = False
            reward = 0

            while not done:
                action = self.pick_action(state, epsilon)
                next_state, next_reward, done, info = self.env.step(action)

                if done:
                    if next_reward > 0:
                        next_state = WIN_STATE
                    elif next_reward < 0:
                        next_state = LOSE_STATE
                    else:
                        next_state = DRAW_STATE

                # Update the Q value for current state and action
                self.N_Q[state][action] += 1
                self.Q_values[state][action] += self.alpha(self.N_Q[state][action]) * (reward + DISCOUNT * max(self.Q_values[next_state]) - self.Q_values[state][action])

                state = next_state
                reward = next_reward

                # Update the Q values for terminal state
                if done:
                    self.N_Q[state][HIT] += 1
                    self.N_Q[state][STAND] += 1
                    self.Q_values[state][HIT] += self.alpha(self.N_Q[state][HIT]) * (reward - self.Q_values[state][HIT])
                    self.Q_values[state][STAND] += self.alpha(self.N_Q[state][STAND]) * (reward - self.Q_values[state][STAND])

    def pick_action(self, s, epsilon):
        if random.random() < epsilon:
            return random.choice([STAND, HIT])
        else:
            return self.autoplay_decision(s)

    def autoplay_decision(self, state):
        standQ, hitQ = self.Q_values[state][STAND], self.Q_values[state][HIT]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT  # Before Q-learning takes effect, just always HIT