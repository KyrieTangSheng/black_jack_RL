from typing import Literal
from seqlearn.hmm import MultinomialHMM
from blackjack_lib.environment.blackjack import BlackjackEnv
import json
import numpy as np
from tqdm import tqdm
import copy

RAW_DATA_PATH = 'blackjack_data.json'

raw_data = json.load(open(RAW_DATA_PATH))

def card_to_index(card: str):
    # converts card to 0-51 index
    special_char_index = {
        'A': 1,
        'J': 10,
        'Q': 10,
        'K': 10,
        '♠': 1,
        '♣': 2,
        '♥': 3,
        '♦': 4
    }
    rank = card[0] if len(card) == 2 else card[:2]
    rank = special_char_index[rank] if rank in special_char_index else int(rank)
    suit = card[1] if len(card) == 2 else card[2]
    return rank

def special_to_index(special: Literal['cont', 'win', 'lose', 'draw']):
    # maps special emissions to 52-54
    descs = ['cont', 'win', 'lose', 'draw']
    return descs.index(special)+2

def description_to_state(desc: Literal['hit', 'stand']):
    descs = ['hit', 'stand']
    return descs.index(desc)

def process_data(data):
    emissions = []
    states = []
    lengths = []
    for sample in tqdm(data):
        cur_emissions = []
        cur_states = []

        cur_player_sum = card_to_index(sample['player_hand'][0])+card_to_index(sample['player_hand'][1])
        dealer_up_card = card_to_index(sample['dealer_hand'][0])
        
        # get turn states and emissions
        for turn in sample['turns']:
            cur_states.append(description_to_state(turn['prev_action']))
            if turn['new_card']:
                cur_player_sum += card_to_index(turn['new_card'])
            cur_emissions.append([cur_player_sum, dealer_up_card, special_to_index('cont')])

        # get last emission
        special = 'win'
        if sample['outcome'] == 'dealer_win' or sample['outcome'] == 'player_bust':
            special = 'lose'
        elif sample['outcome'] == 'draw':
            special = 'draw'
        cur_emissions[-1][2] = special_to_index(special)

        states.extend(cur_states)
        emissions.extend(cur_emissions)
        lengths.append(len(cur_states))

    states = np.array(states)
    emissions = np.array(emissions)
    return emissions, states, lengths


emissions, states, lengths = process_data(raw_data)

wins = 0
draws = 0
for sample in tqdm(raw_data):
    wins += 1 if (sample['outcome'] == 'player_win' or sample['outcome'] == 'dealer_bust') else 0
    draws += 1 if sample['outcome'] == 'draw' else 0
print('win rate (random):', wins/len(raw_data))
print('draw rate (random):', draws/len(raw_data))

def test_hmm(N_rounds):
    mhmm = MultinomialHMM()
    mhmm.fit(emissions, states, lengths=lengths)
    env = BlackjackEnv()
    wins = 0
    draws = 0
    for _ in tqdm(range(N_rounds)):
        env.reset()
        cur_emissions = []
        dealer_up_card = card_to_index(env.dealer_hand[0])
        player_sum = card_to_index(env.player_hand[0])+card_to_index(env.player_hand[1])
        while not env.game_over:
            actions = 0
            total = 0
            # get every possible card in deck (card counting):
            for card in env.deck.cards + [env.dealer_hand[1]]:
                prospective_emissions = cur_emissions + [[player_sum+card_to_index(card), dealer_up_card, special_to_index('win')]]
                actions += mhmm.predict(np.array(prospective_emissions))[0]
                total += 1
            action = 1 if actions / total >= 0.5 else 0
            # action = 0 if action else 1 # flip action since we predicted lose
            _, _, _, info = env.step(action)
            player_sum += card_to_index(env.player_hand[-1])
            cur_emissions.append([player_sum, dealer_up_card, special_to_index('cont')])

        wins += 1 if (info['result'] == 'player_win' or info['result'] == 'dealer_bust') else 0
        draws += 1 if info['result'] == 'draw' else 0
    return wins/N_rounds, draws/N_rounds

winrate, drawrate = test_hmm(10000)
print('win rate (HMM):', winrate) # 0.384 optimize for win 0.383 optimize for lose
print('draw rate (HMM):', drawrate) # 0.0495 for win 0.0499 optimize for lose