import copy
import random
from blackjack_lib.environment.blackjack import BlackjackEnv

STAND = 0
HIT = 1
WIN_STATE = (1, 0, 0)
DRAW_STATE = (0, 0, 0)
LOSE_STATE = (-1, 0, 0)


class QAgent:
    def __init__(self, discount=0.95, lr_base=10.0,
                 max_hand_value=21,
                 force_ace_value=None,
                 dealer_stick_threshold=17):  # <--- NEW PARAMETER

        self.discount = discount
        self.lr_base = lr_base
        self.max_hand_value = max_hand_value

        # Pass the threshold to the environment
        self.env = BlackjackEnv(max_hand_value=max_hand_value,
                                force_ace_value=force_ace_value,
                                dealer_stick_threshold=dealer_stick_threshold)  # <--- PASS IT

        states = self._generate_states(max_hand_value)

        self.Q_values = {}
        self.N_Q = {}

        for s in states:
            self.Q_values[s] = [0, 0]
            self.N_Q[s] = [0, 0]

        self.training_history = {
            'game_numbers': [],
            'win_rates': [],
            'rewards': []
        }

    # ... (Keep _generate_states, alpha, Q_run, pick_action, autoplay_decision exactly as they were) ...
    def _generate_states(self, max_val):
        states = [WIN_STATE, DRAW_STATE, LOSE_STATE]
        for player_sum in range(2, max_val + 1):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    states.append((player_sum, dealer_card, usable_ace))
        return states

    def alpha(self, n):
        return self.lr_base / (9 + n)

    def Q_run(self, num_simulation, epsilon=0.4, track_performance=False, eval_interval=1000):
        # ... (Same logic as provided in previous steps) ...
        # (Paste the full Q_run function from the previous response here)
        if track_performance:
            self.training_history = {'game_numbers': [], 'win_rates': [], 'rewards': []}

        eval_window_wins = 0
        eval_window_rewards = 0
        eval_window_games = 0

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

                self.N_Q[state][action] += 1
                best_next = max(self.Q_values[next_state]) if next_state in self.Q_values else 0
                self.Q_values[state][action] += self.alpha(self.N_Q[state][action]) * (
                            reward + self.discount * best_next - self.Q_values[state][action])

                state = next_state
                reward = next_reward
                episode_reward += reward

                if done:
                    if state in self.N_Q:
                        self.N_Q[state][HIT] += 1
                        self.N_Q[state][STAND] += 1
                        self.Q_values[state][HIT] += self.alpha(self.N_Q[state][HIT]) * (
                                    reward - self.Q_values[state][HIT])
                        self.Q_values[state][STAND] += self.alpha(self.N_Q[state][STAND]) * (
                                    reward - self.Q_values[state][STAND])

                    if track_performance:
                        eval_window_games += 1
                        eval_window_rewards += episode_reward
                        if episode_reward > 0: eval_window_wins += 1

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
        if state not in self.Q_values: return HIT
        standQ, hitQ = self.Q_values[state][STAND], self.Q_values[state][HIT]
        if hitQ > standQ: return HIT
        if standQ > hitQ: return STAND
        return HIT