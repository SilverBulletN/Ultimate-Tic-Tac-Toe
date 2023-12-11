import numpy as np
import copy
import time

def select_move(cur_state, remain_time):
    if remain_time < 3:
        valid_moves = cur_state.get_valid_moves
        return np.random.choice(valid_moves) if valid_moves else None

    # In the beginning of the game, choose a move from the center
    if len(cur_state.get_valid_moves) == 81:
        return cur_state.get_valid_moves[40]

    elapsed_time = time.time()
    empty_slots = count_empty_slots_in_active_boards(cur_state)
    depth = calculate_depth(cur_state.get_valid_moves, empty_slots)

    # print("Calculating move with depth:", depth)
    best_move, _ = alphabeta(cur_state, depth, float('-inf'), float('inf'), True, cur_state.player_to_move)

    total_elapsed_time = time.time() - elapsed_time
    # print(f"Selected move: {best_move} (Calculation time: {total_elapsed_time:.2f} seconds)")

    return best_move if best_move is not None else np.random.choice(cur_state.get_valid_moves)

def count_empty_slots_in_active_boards(cur_state):
    empty_slots = 0
    for idx, local_board in enumerate(cur_state.blocks):
        if cur_state.global_cells[idx] == 0:
            empty_slots += np.sum(local_board == 0)
    return empty_slots

def calculate_depth(valid_moves, empty_slots):
    base_depth = 3  # Increase depth for a deeper search
    early_game_threshold = 40
    mid_game_threshold = 20
    late_game_threshold = 10

    # Adjust depth based on game complexity
    if empty_slots > early_game_threshold:
        depth = base_depth
    elif early_game_threshold >= empty_slots > mid_game_threshold:
        depth = base_depth + 1  # Deeper search as the game progresses
    elif mid_game_threshold >= empty_slots > late_game_threshold:
        depth = base_depth + 2
    else:
        depth = base_depth + 3  # Deepest search in late game

    # Further adjustments based on the number of valid moves
    if len(valid_moves) > 8:
        depth = max(depth - 1, base_depth)
    
    return depth

def alphabeta(state, depth, alpha, beta, maximizingPlayer, player):
    if depth == 0 or state.game_over:
        return None, evaluate(state, player)

    if maximizingPlayer:
        maxEval = float('-inf')
        moves = state.get_valid_moves
        sorted_moves = sort_moves(moves, state, player)
        best_move = None

        for move in sorted_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            _, eval = alphabeta(new_state, depth - 1, alpha, beta, False, player)
            if eval > maxEval:
                maxEval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, maxEval
    else:
        minEval = float('inf')
        moves = state.get_valid_moves
        sorted_moves = sort_moves(moves, state, -player)
        best_move = None

        for move in sorted_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            _, eval = alphabeta(new_state, depth - 1, alpha, beta, True, player)
            if eval < minEval:
                minEval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, minEval

def sort_moves(moves, state, player):
    prioritized_moves = []
    other_moves = []

    for move in moves:
        # Create a hypothetical state for each move
        hypothetical_state = copy.deepcopy(state)
        hypothetical_state.act_move(move)
        
        # Prioritize moves that align with Xavier's strategy
        if is_xavier_strategy_move(move, state):
            prioritized_moves.append(move)
        else:
            other_moves.append(move)

    # Return prioritized moves first, followed by other moves
    return prioritized_moves + other_moves

def is_xavier_strategy_move(move, state):
    # Define Xavier's strategy moves based on the current state
    if len(state.get_valid_moves) == 81:
        center_moves = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        return move in center_moves
    elif len(state.get_valid_moves) <= 17:
        return True  # Prioritize strategic moves in early and mid-game
    else:
        return False  # In later stages, prioritize any valid move

# The rest of the code remains the same

def evaluate(state, player):
    score = 0

    # Enhanced evaluation of local boards considering strategic positions
    for idx, local_board in enumerate(state.blocks):
        score += evaluate_local_board(local_board, player, idx, state)

    # Adjust the evaluation of the global board
    global_board_score = evaluate_global_board(state.global_cells.reshape(3, 3), player, state)
    score += global_board_score

    return score

def count_score(array, player):
    opp_player = -player
    score = 0

    player_count = np.count_nonzero(array == player)
    opp_count = np.count_nonzero(array == opp_player)

    # Prioritize control of central squares and strategic positions
    if player_count == 3:
        score += 1000  # Winning a line
    elif player_count == 2 and opp_count == 0:
        score += 200  # Potential win or strategic positioning
    elif player_count == 1 and opp_count == 2:
        score += 500  # Blocking opponent's potential win
    elif player_count == 1 and opp_count == 0:
        score += 50   # Control of a central square or strategic position

    # Penalize opponent's control, especially in critical spots
    if opp_count == 3:
        score -= 1020
    elif opp_count == 2 and player_count == 0:
        score -= 510

    return score


def evaluate_local_board(board, player, board_index, state):
    score = 0
    for i in range(3):
        score += count_score(board[i, :], player)
        score += count_score(board[:, i], player)

    diag1 = [board[i, i] for i in range(3)]
    diag2 = [board[i, 2 - i] for i in range(3)]
    score += count_score(diag1, player)
    score += count_score(diag2, player)

    # Prioritize central squares and strategic positions
    if board_index == 4:  # Center board
        score += 30  # Add more points for controlling the center board
    elif board_index in [0, 2, 6, 8]:  # Corner boards
        score += 20  # Slightly higher score for corner boards
    elif board_index in [1, 3, 5, 7]:  # Edge boards
        score += 25  # Slightly higher score for edge boards

    return score


def evaluate_global_board(board, player, state):
    score = 0

    for i in range(3):
        score += count_score(board[i, :], player)
        score += count_score(board[:, i], player)

    diag1 = [board[i, i] for i in range(3)]
    diag2 = [board[i, 2 - i] for i in range(3)]
    score += count_score(diag1, player)
    score += count_score(diag2, player)

    return score
