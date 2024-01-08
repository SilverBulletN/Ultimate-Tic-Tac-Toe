from keras.models import load_model
from keras import *
import numpy as np
import copy
import time
from state import State, UltimateTTT_Move
import math
import random
from importlib import import_module
# Hyperparameters
model_path = "model.h5"
mcts_search = 50
MCTS = True
cpuct = 4

# initializing search tree
Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}   # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state

nn = load_model(model_path)
def select_move(cur_state: State, remain_time):
    global nn
    valid_moves = cur_state.get_valid_moves
    if not valid_moves:
        return None
    valid_moves_idx = valid_moves_to_array(valid_moves)
    valids_mask = np.zeros(81)
    np.put(valids_mask, valid_moves_idx,1)
    # print(f"{valid_moves = }")
    if remain_time < 3:
        return np.random.choice(valid_moves) if valid_moves else None
    if len(cur_state.get_valid_moves) == 81:
        return cur_state.get_valid_moves[40]
    if MCTS:
        board, current_player, mini_board = bridge(cur_state)
        policy = get_action_probs(board, current_player, mini_board)
        policy = policy / np.sum(policy)
    else:
        nn_input = copy.deepcopy(cur_state.blocks).reshape(81)
        for i in list(valid_moves_idx):
            if nn_input[i] == 0:
                nn_input[i] = 0.1
        policy, value = nn.predict(nn_input.reshape((1,9,9)))
        policy = policy.reshape(81) * valids_mask
        policy = policy /(np.sum(policy))
    action = np.argmax(policy)
    # print(f"{action = }, {value = }")
    best_move = to_UTTT_move(action, cur_state)
    
    return best_move if best_move is not None else np.random.choice(cur_state.get_valid_moves)

def valid_moves_to_array(valid_moves):
    arr = []
    for move in valid_moves:
        arr.append(9*move.index_local_board + 3*move.x + move.y)
    return np.array(arr, dtype=np.int32)

######################################################
def bridge(cur_state: State):
    valid_moves = cur_state.get_valid_moves
    if not valid_moves:
        return None
    valid_moves_idx = valid_moves_to_array(valid_moves)
    if len(valid_moves_idx) == 81:
        mini_board = 9
    else:
        mini_board = valid_moves_idx[0]//9
    board = []
    for block_idx in range(9):
        block = []
        for x in range(3):
            row = []
            for y in range(3):
                if cur_state.blocks[block_idx][x][y] == 0:
                    row.append(' ')
                elif cur_state.blocks[block_idx][x][y] == 1:
                    row.append('X')
                else:
                    row.append('O')
            block.append(row)
        board.append(block)
    return board, cur_state.player_to_move, mini_board
######################################################
def possiblePos(board, subBoard):

    if subBoard == 9:
        return range(81)
    
    possible = []


    # otherwise, finds all available spaces in the subBoard
    if board[subBoard][1][1] != 'x' and board[subBoard][1][1] != 'o':
        for row in range(3):
            for coloumn in range(3):
                if board[subBoard][row][coloumn] == " ":
                    possible.append((subBoard * 9) + (row * 3) + coloumn)
        if len(possible) > 0:
            return possible

    # if the subboard has already been won, it finds all available spaces on the entire board
    for mini in range(9):
        if board[mini][1][1] == "x" or board[mini][1][1] == "o":
            continue
        for row in range(3):
            for coloumn in range(3):
                if board[mini][row][coloumn] == " ":
                    possible.append((mini * 9) + (row * 3) + coloumn)

    return possible

def move(board,action, player):

    if player == 1:
        turn = 'X'
    if player == -1:
        turn = "O"
    
    bestPosition = []

    bestPosition.append(int (action / 9))
    remainder = action % 9
    bestPosition.append(int (remainder/3))
    bestPosition.append(remainder%3)

    # place piece at position on board
    board[bestPosition[0]][bestPosition[1]][bestPosition[2]] = turn

    emptyMiniBoard = [[" "," "," "], [" "," "," "], [" "," "," "]]

    wonBoard = False
    win = False
    mini = board[bestPosition[0]]
    subBoard = bestPosition[0]
    x = bestPosition[1]
    y = bestPosition[2]

    #check for win on verticle
    if mini[0][y] == mini[1][y] == mini [2][y]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on horozontal
    if mini[x][0] == mini[x][1] == mini [x][2]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on negative diagonal
    if x == y and mini[0][0] == mini[1][1] == mini [2][2]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #check for win on positive diagonal
    if x + y == 2 and mini[0][2] == mini[1][1] == mini [2][0]:
        board[subBoard] = emptyMiniBoard
        board[subBoard][1][1] = turn.lower()
        wonBoard = True

    #set new subBoard
    newsubBoard = (bestPosition[1] * 3) + bestPosition[2]

    # if the subBoard was won, checking whether the entire board is won as well
    if wonBoard == True:
        win = checkWinner(board, subBoard, turn)
    
    #if win:
    #    print ("won game!")
    #    print_board(board)

    return board, newsubBoard, win

def checkWinner(board,winningSubBoard, turn):

    # getting coordinates of winning subBoard
    for i in range(3):
        if (winningSubBoard - i) % 3 == 0:
            row = int((winningSubBoard - i) /3)
            winningSubBoardCoordinate = [row,i]
            break

    # making winning subBoard using just centre pieces
    winningBoard = [
    [board[0][1][1], board[1][1][1], board[2][1][1]],
    [board[3][1][1], board[4][1][1], board[5][1][1]],
    [board[6][1][1], board[7][1][1], board[8][1][1]]
    ]

    # horozontal wins
    if turn.lower() == winningBoard[winningSubBoardCoordinate[0]][0] == winningBoard[winningSubBoardCoordinate[0]][1] == winningBoard[winningSubBoardCoordinate[0]][2]:
        return True
    # vertical wins
    elif turn.lower() == winningBoard[0][winningSubBoardCoordinate[1]] == winningBoard[1][winningSubBoardCoordinate[1]] == winningBoard[2][winningSubBoardCoordinate[1]]:
        return True
    # top left to bottom right diagonal
    elif turn.lower() == winningBoard[0][0] == winningBoard[1][1] == winningBoard[2][2]:
        return True
    # bottom left to top right diagonal
    elif turn.lower() == winningBoard[2][0] == winningBoard[1][1] == winningBoard[0][2]:
        return True
    else:
        return False

def fill_winning_boards(board):

    # takes in a board in its normal state, and converts all suboards that have been won to be filled with the winning player's piece

    new_board = []
    for suboard in board:
        if suboard[1][1] =='x':
            new_board.append([["X","X","X"],["X","X","X"],["X","X","X"]])
        elif suboard[1][1] =='o':
            new_board.append([["O","O","O"],["O","O","O"],["O","O","O"]])
        else:
            new_board.append(suboard)
    return new_board

def letter_to_int(letter, player):
    # based on the letter in a box in the board, replaces 'X' with 1 and 'O' with -1
    if letter == 'v':
        return 0.1
    elif letter == " ":
        return 0
    elif letter == "X":
        return 1 * player
    elif letter =="O":
        return -1 * player
    
def to_UTTT_move(action, cur_state: State):
    old_agent = import_module('_MSSV')
    best_move = old_agent.select_move(cur_state, 1000)
    return random.choices([best_move, UltimateTTT_Move(int(action/9), int((action % 9)/3), action % 3, cur_state.player_to_move)],weights=(0.7, 0.3) ,k=1)[0]

def board_to_array(boardreal, mini_board, player):
    
    # makes copy of board, so that the original board does not get changed
    board = copy.deepcopy(boardreal)

    # takes a board in its normal state, and returns a 9x9 numpy array, changing 'X' = 1 and 'O' = -1
    # also places a 0.1 in all valid board positions

    board = fill_winning_boards(board)
    tie = True

    # if it is the first turn, then all of the cells are valid moves
    if mini_board == 9:
        return np.full((9,9), 0.1)

    # replacing all valid positions with 'v'
    # checking whether all empty values on the board are valid
    if board[mini_board][1][1] != 'x' or board[mini_board][1][1] != 'o':
        for line in range(3):
            for item in range(3):
                if board[mini_board][line][item] == " ":
                    board[mini_board][line][item] = 'v'
                    tie = False
    # if not, then replacing empty cells in mini board with 'v'
    else:
        for suboard in range (9):
            for line in range(3):
                for item in range(3):
                    if board[suboard][line][item] == " ":
                        board[suboard][line][item] = 'v'

    # if the miniboard ends up being a tie
    if tie:
        for suboard in range (9):
            for line in range(3):
                for item in range(3):
                    if board[suboard][line][item] == " ":
                        board[suboard][line][item] = 'v'


    array = []
    firstline = []
    secondline = []
    thirdline = []
    
    for suboardnum in range(len(board)):
            
        for item in board[suboardnum][0]:
            firstline.append(letter_to_int(item, player))
        
        for item in board[suboardnum][1]:
            secondline.append(letter_to_int(item, player))
        
        for item in board[suboardnum][2]:
            thirdline.append(letter_to_int(item, player))
        
        if (suboardnum + 1) % 3 == 0:
            array.append(firstline)
            array.append(secondline)
            array.append(thirdline)
            firstline = []
            secondline = []
            thirdline = []

    nparray = np.array(array)
    
    return nparray

def mcts(s, current_player, mini_board):

    if mini_board == 9:
        possibleA = range(81)
    else:
        possibleA = possiblePos(s, mini_board)
    sArray = board_to_array(s, mini_board, current_player)
    sTuple = tuple(map(tuple, sArray))
    if len(possibleA) > 0:
        if sTuple not in P.keys():

            policy, v = nn.predict(sArray.reshape(1,9,9))
            v = v[0][0]
            valids = np.zeros(81)
            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)
            P[sTuple] = policy

            Ns[sTuple] = 1

            for a in possibleA:
                Q[(sTuple,a)] = 0
                Nsa[(sTuple,a)] = 0
                W[(sTuple,a)] = 0
            return -v

        best_uct = -100
        for a in possibleA:

            uct_a = Q[(sTuple,a)] + cpuct * P[sTuple][a] * (math.sqrt(Ns[sTuple]) / (1 + Nsa[(sTuple,a)]))

            if uct_a > best_uct:
                best_uct = uct_a
                best_a = a
        
        next_state, mini_board, wonBoard = move(s, best_a, current_player)

        if wonBoard:
            v = 1
        else:
            current_player *= -1
            v = mcts(next_state, current_player, mini_board)
    else:
        return 0

    W[(sTuple,best_a)] += v
    Ns[sTuple] += 1
    Nsa[(sTuple,best_a)] += 1
    Q[(sTuple,best_a)] = W[(sTuple,best_a)] / Nsa[(sTuple,best_a)]
    return -v

def get_action_probs(init_board, current_player, mini_board):

    for _ in range(mcts_search):
        s = copy.deepcopy(init_board)
        value = mcts(s, current_player, mini_board)
    
    print ("done one iteration of MCTS")

    actions_dict = {}

    sArray = board_to_array(init_board, mini_board, current_player)
    sTuple = tuple(map(tuple, sArray))

    for a in possiblePos(init_board, mini_board):
        actions_dict[a] = Nsa[(sTuple,a)] / Ns[sTuple]
    print ("actions dict-", actions_dict)
    action_probs = np.zeros(81)
    
    for a in actions_dict:
        np.put(action_probs, a, actions_dict[a], mode='raise')
    
    return action_probs

