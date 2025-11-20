from typing import Literal
from seqlearn.hmm import MultinomialHMM
from src.environment.blackjack import BlackjackEnv
import json
import numpy as np
from tqdm import tqdm

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
    return descs.index(special)+21

def description_to_state(desc: Literal['pass', 'hit', 'stand']):
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
        cur_dealer_sum = card_to_index(sample['dealer_hand'][0])
        
        # get turn states and emissions
        for turn in sample['turns']:
            cur_states.append(description_to_state(turn['prev_action']))
            if turn['new_card']:
                cur_player_sum += card_to_index(turn['new_card'])
            cur_emissions.append([cur_player_sum, special_to_index('cont')])

        # get last emission
        special = 'win'
        if sample['outcome'] == 'dealer_win' or sample['outcome'] == 'player_bust':
            special = 'lose'
        elif sample['outcome'] == 'draw':
            special = 'draw'
        cur_emissions[-1][1] = special_to_index(special)

        states.extend(cur_states)
        emissions.extend(cur_emissions)
        lengths.append(len(cur_states))

    states = np.array(states)
    emissions = np.array(emissions)
    return emissions, states, lengths

mhmm = MultinomialHMM()
emissions, states, lengths = process_data(raw_data)
mhmm.fit(emissions, states, lengths=lengths)

env = BlackjackEnv()
predictions = []
for _ in range(1000):
    env.reset()
    cur_emissions = []
    dealer_sum = card_to_index(env.dealer_hand[0])
    player_sum = card_to_index(env.player_hand[0]) + card_to_index(env.player_hand[1])
    cur_emissions.append([player_sum, special_to_index('lose')])
    
    predictions.append(mhmm.predict(np.array(cur_emissions)))

print(np.unique(predictions))