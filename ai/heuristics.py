import numpy as np
from constants import BOARD_SIZE, S1, S2, S3, S4

def heuristic(move, state, player):
    r, c, nr, nc = move
    is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1

    temp_state = state.copy()
    temp_state.make_move(move)
    captured = np.sum(temp_state.board == player) - np.sum(state.board == player) - (1 if is_clone else 0)

    allies = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue

            nnr, nnc = nr + dr, nc + dc
            if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and state.board[nnr][nnc] == player:
                allies += 1

    bonus_clone = 1 if is_clone else 0

    penalty_jump = 0
    if not is_clone:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nnr, nnc = r + dr, c + dc
                if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and state.board[nnr][nnc] == player:
                    penalty_jump += 1

    return S1 * captured + S2 * allies + S3 * bonus_clone - S4 * penalty_jump

def evaluate(state, player):
    if state.is_game_over():
        if state.current_player == player:
            return 1.0
        elif state.current_player == -player:
            return 0
        else:
            return 0.5
        
    own = np.sum(state.board == player)
    opp = np.sum(state.board == -player)
    score = own / (own + opp) if (own + opp) > 0 else 0.5
    return score



