import copy
import random
from blackjack_lib.environment.blackjack import BlackjackEnv, MAX_HAND_VALUE

STAND = 0
HIT = 1
DISCOUNT = 0.95  # This is the gamma value for all value calculations
WIN_STATE = (1,0,0)
DRAW_STATE = (0,0,0)
LOSE_STATE = (-1,0,0)

def generate_states(max_hand_value):
    states = []
    states.append(WIN_STATE)
    states.append(DRAW_STATE)
    states.append(LOSE_STATE)
    for player_sum in range(2, max_hand_value + 1):
        for dealer_card in range(1,11):
            for usable_ace in [False, True]:
                s = (player_sum, dealer_card, usable_ace)
                states.append(s)
    return states

class QAgent:
    def __init__(self, discount=0.95, lr_base=10.0, max_hand_value=MAX_HAND_VALUE):

        self.discount = discount
        self.lr_base = lr_base
        self.max_hand_value = max_hand_value

        states = generate_states(max_hand_value)

        # For Q-learning values
        self.Q_values = {}   # Dictionary: Store the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state for each action

        # Initialization of the values
        for s in states:
            self.Q_values[s] = [0,0]  # First element is the Q value of "Stand", second element is the Q value of "Hit"
            self.N_Q[s] = [0,0]  # First element is the number of visits of "Stand" at state s, second element is the Q value of "Hit" at s

        # Game environment
        self.env = BlackjackEnv(max_hand_value=max_hand_value)
        
        self.training_history = {
            'game_numbers': [],
            'win_rates': [],
            'rewards': []
        }

    def alpha(self, n):
        return self.lr_base/(9 + n)

    def Q_run(self, num_simulation, epsilon=0.4, track_performance=False, eval_interval=1000):

        if track_performance:
            self.training_history = {
                'game_numbers': [],
                'win_rates': [],
                'rewards': []
            }
        
        eval_window_wins = 0
        eval_window_rewards = 0
        eval_window_games = 0

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):
            state = self.env.reset()
            done = False
            reward = 0
            episode_reward = 0

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
                self.Q_values[state][action] += self.alpha(self.N_Q[state][action]) * (reward + self.discount * max(self.Q_values[next_state]) - self.Q_values[state][action])

                state = next_state
                reward = next_reward
                episode_reward += reward

                # Update the Q values for terminal state
                if done:
                    self.N_Q[state][HIT] += 1
                    self.N_Q[state][STAND] += 1
                    self.Q_values[state][HIT] += self.alpha(self.N_Q[state][HIT]) * (reward - self.Q_values[state][HIT])
                    self.Q_values[state][STAND] += self.alpha(self.N_Q[state][STAND]) * (reward - self.Q_values[state][STAND])
                    
                    if track_performance:
                        eval_window_games += 1
                        eval_window_rewards += episode_reward
                        if episode_reward > 0:
                            eval_window_wins += 1
                        
                        if eval_window_games >= eval_interval:
                            win_rate = eval_window_wins / eval_window_games
                            avg_reward = eval_window_rewards / eval_window_games
                            
                            self.training_history['game_numbers'].append(simulation + 1)
                            self.training_history['win_rates'].append(win_rate)
                            self.training_history['rewards'].append(avg_reward)
                            
                            eval_window_wins = 0
                            eval_window_rewards = 0
                            eval_window_games = 0

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