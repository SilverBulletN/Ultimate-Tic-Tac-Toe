import numpy as np


def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        # print("valid moves: ",valid_moves)
        # print("cur state: ",cur_state)
        return np.random.choice(valid_moves)
    return None

# import numpy as np
# import copy

# # Giả sử các hàm evaluate và is_terminal đã được định nghĩa ở đâu đó
# # Hàm evaluate đánh giá trạng thái của bảng
# # Hàm is_terminal kiểm tra xem trò chơi đã kết thúc chưa

# def select_move(cur_state, remain_time):
#     empty_slots = np.sum(cur_state.global_cells == 0)
#     print("empty slots: ", empty_slots)
    
#     # Tính toán độ sâu dựa trên số ô trống
#     # Ví dụ: sử dụng công thức đơn giản như 1 + (81 - empty_slots) // 20
#     # Điều này sẽ tăng độ sâu từ 1 lên 2, 3, 4... khi trò chơi tiến triển
#     depth = 1 + (9 - empty_slots)//2

#     best_move = None
#     best_score = float('-inf')

#     # Tìm tất cả các nước đi hợp lệ
#     valid_moves = cur_state.get_valid_moves

#     print("depth: ", depth)
#     # Áp dụng Minimax cho mỗi nước đi hợp lệ
#     for move in valid_moves:
#         new_state = copy.deepcopy(cur_state)
#         new_state.act_move(move)
#         score = minimax(new_state, depth, False)
#         if score > best_score:
#             best_score = score
#             best_move = move
#         print("move: ", move, "score: ", score)
    
#     print("best move: ", best_move, "best score: ", best_score)

#     return best_move

# def minimax(state, depth, maximizingPlayer):
#     if depth == 0 or state.game_over:
#         return evaluate(state)

#     if maximizingPlayer:
#         maxEval = float('-inf')
#         for move in state.get_valid_moves:
#             new_state = copy.deepcopy(state)
#             new_state.act_move(move)
#             eval = minimax(new_state, depth - 1, False)
#             maxEval = max(maxEval, eval)
#         return maxEval
#     else:
#         minEval = float('inf')
#         for move in state.get_valid_moves:
#             new_state = copy.deepcopy(state)
#             new_state.act_move(move)
#             eval = minimax(new_state, depth - 1, True)
#             minEval = min(minEval, eval)
#         return minEval

# # Định nghĩa hàm evaluate dựa trên logic của trò chơi
# def evaluate(state):
#     player = state.player_to_move
#     score = 0

#     # Đánh giá các bảng nhỏ
#     for local_board in state.blocks:
#         score += evaluate_local_board(local_board, player)

#     # Đánh giá bảng lớn
#     score += evaluate_local_board(state.global_cells.reshape(3, 3), player) * 10

#     return score

# def evaluate_local_board(board, player):
#     score = 0
#     # Đánh giá hàng, cột, và đường chéo
#     for i in range(3):
#         score += count_score(board[i, :], player)  # Hàng
#         score += count_score(board[:, i], player)  # Cột

#     # Đánh giá đường chéo
#     diag1 = [board[i, i] for i in range(3)]
#     diag2 = [board[i, 2 - i] for i in range(3)]
#     score += count_score(diag1, player)
#     score += count_score(diag2, player)

#     return score

# def count_score(array, player):
#     opp_player = -player
#     score = 0

#     if np.count_nonzero(array == player) == 3:
#         score += 100
#     elif np.count_nonzero(array == player) == 2:
#         score += 50
#     elif np.count_nonzero(array == player) == 1:
#         score += 20

#     if np.count_nonzero(array == opp_player) == 3:
#         score -= 100
#     elif np.count_nonzero(array == opp_player) == 2:
#         score -= 50

#     if np.count_nonzero(array == player) == 1 and np.count_nonzero(array == opp_player) == 2:
#         score += 10

#     return score


# # Định nghĩa hàm is_terminal để kiểm tra trạng thái kết thúc của trò chơi
# def is_terminal(state):
#     # Kiểm tra xem tất cả các ô trong bảng lớn đã được điền chưa
#     if np.all(state.global_cells != 0):
#         return True

#     # Kiểm tra xem tất cả các ô trong từng bảng nhỏ đã được điền chưa
#     for local_board in state.blocks:
#         if np.all(local_board != 0):
#             return True

#     # Kiểm tra xem có người chơi nào chiến thắng ở bảng lớn chưa
#     if check_winner(state.global_cells.reshape(3, 3)):
#         return True

#     return False

# def check_winner(board):
#     # Kiểm tra hàng, cột, và đường chéo cho bảng
#     for i in range(3):
#         if abs(np.sum(board[i, :])) == 3 or abs(np.sum(board[:, i])) == 3:
#             return True

#     # Kiểm tra đường chéo
#     if abs(np.sum([board[i, i] for i in range(3)])) == 3 or abs(np.sum([board[i, 2 - i] for i in range(3)])) == 3:
#         return True

#     return False

