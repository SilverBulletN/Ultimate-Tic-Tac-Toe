from keras.models import load_model
from keras import *
import numpy as np
import copy
import time
from state import State, UltimateTTT_Move
# Hyperparameters
model_path = "model.h5"

nn = load_model(model_path)
def select_move(cur_state: State, remain_time):
    global nn
    valid_moves = cur_state.get_valid_moves
    valid_moves_idx = valid_moves_to_array(valid_moves)
    valids_mask = np.zeros(81)
    np.put(valids_mask, valid_moves_idx,1)
    # print(f"{valid_moves = }")
    if remain_time < 3:
        return np.random.choice(valid_moves) if valid_moves else None
    if len(cur_state.get_valid_moves) == 81:
        return cur_state.get_valid_moves[40]
    nn_input = cur_state.blocks.reshape((1,9,9))
    nn_input = np.where(nn_input == 0, 0.1, nn_input)

    policy, value = nn.predict(nn_input)
    policy = policy.reshape(81) * valids_mask
    policy = policy /(0.01+np.sum(policy))
    action = np.argmax(policy)
    print(f"{action = }, {value = }")
    best_move = to_UTTT_move(action, cur_state.player_to_move)
    
    return best_move if best_move is not None else np.random.choice(cur_state.get_valid_moves)

def valid_moves_to_array(valid_moves):
    arr = []
    for move in valid_moves:
        arr.append(9*move.index_local_board + 3*move.y + move.x)
    return np.array(arr, dtype=np.int32)

def to_UTTT_move(action, player_to_move):
    x = action % 3
    action = (action - x)//3
    y = action % 3
    local_board = (action - y) // 3
    return UltimateTTT_Move(local_board, x, y, player_to_move)