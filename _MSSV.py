import numpy as np
import copy
import time



def calculate_depth(valid_moves, empty_slots):
    base_depth = 2
    early_game_threshold = 40  # Example threshold for early game
    mid_game_threshold = 15    # Example threshold for mid game
    late_game_threshold = 10   # Example threshold for late game

    # Adjust depth based on the stage of the game
    if empty_slots > early_game_threshold:
        depth = base_depth
    elif early_game_threshold >= empty_slots > mid_game_threshold:
        depth = base_depth + 1
    elif mid_game_threshold >= empty_slots > late_game_threshold:
        depth = base_depth + 2
    else:  # Late game
        depth = base_depth + 5

    # Further adjust depth based on the number of valid moves
    if len(valid_moves) > 8:
        depth = max(depth - 1, base_depth)  # Reduce depth if too many choices
    # elif len(valid_moves) < 4:
    #     depth += 1  # Increase depth if fewer choices

    return depth

# Usage in your Minimax function

def count_empty_slots_in_active_boards(cur_state):
    empty_slots = 0
    for idx, local_board in enumerate(cur_state.blocks):
        # Check if the corresponding cell in the global board is still free (indicating active play in the local board)
        if cur_state.global_cells[idx] == 0:
            empty_slots += np.sum(local_board == 0)
    return empty_slots

# Usage




def select_move(cur_state, remain_time):
    if remain_time < 3:
        valid_moves = cur_state.get_valid_moves
        return np.random.choice(valid_moves) if valid_moves else None

    elapsed_time = time.time()
    empty_slots = count_empty_slots_in_active_boards(cur_state)
    
    depth = calculate_depth(cur_state.get_valid_moves, empty_slots)


    best_move = None
    best_score = float('-inf')
    valid_moves = cur_state.get_valid_moves

    print("depth: ", depth)
    for move in valid_moves:
        start_time = time.time()
        new_state = copy.deepcopy(cur_state)
        new_state.act_move(move)
        score = minimax(new_state, depth, False)
        if score > best_score:
            best_score = score
            best_move = move

        # Update elapsed time
        each_move_time = time.time() - start_time
        total_elapsed_time = time.time() - elapsed_time

        # Check if we are out of time
        if total_elapsed_time + each_move_time > 8:  
            print("Out of time")
            break
    print("best_move: ", best_move)
    print("num of valid_moves: ", len(valid_moves))
    print("num of empty_slots: ", empty_slots)

    return best_move if best_move is not None else np.random.choice(valid_moves)

def minimax(state, depth, maximizingPlayer):
    if depth == 0 or state.game_over:
        return evaluate(state)

    if maximizingPlayer:
        maxEval = float('-inf')
        for move in state.get_valid_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            eval = minimax(new_state, depth - 1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = float('inf')
        for move in state.get_valid_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            eval = minimax(new_state, depth - 1, True)
            minEval = min(minEval, eval)
        return minEval


def evaluate(state):
    player = state.player_to_move
    score = 0

    # Evaluate local boards
    for local_board in state.blocks:
        score += evaluate_local_board(local_board, player)

    # Evaluate global board
    score += evaluate_local_board(state.global_cells.reshape(3, 3), player) * 10

    return score

def evaluate_local_board(board, player):
    score = 0
    # Evaluate rows and columns
    for i in range(3):
        score += count_score(board[i, :], player)  # Hàng
        score += count_score(board[:, i], player)  # Cột

    # Evaluate diagonals
    diag1 = [board[i, i] for i in range(3)]
    diag2 = [board[i, 2 - i] for i in range(3)]
    score += count_score(diag1, player)
    score += count_score(diag2, player)

    return score

def count_score(array, player):
    opp_player = -player
    score = 0

    player_count = np.count_nonzero(array == player)
    opp_count = np.count_nonzero(array == opp_player)

    if player_count == 3:
        score += 500  # Increased score for winning a sub-board
    elif player_count == 2 and opp_count == 0:
        score += 50   # Potential win
    elif player_count == 1:
        score += 20   # Occupying a position

    if opp_count == 3:
        score -= 500  # Opponent winning a sub-board
    elif opp_count == 2 and player_count == 0:
        score -= 250  # Blocking opponent's potential win

    return score
