import numpy as np
from constants import BOARD_SIZE, EMPTY, PLAYER_1, PLAYER_2

class AtaxxState:
    def __init__(self, initial_board=None, current_player=PLAYER_1):
        if initial_board is None:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
            self.board[0][0] = self.board[6][6] = PLAYER_1
            self.board[0][6] = self.board[6][0] = PLAYER_2
        else:
            self.board = initial_board.copy()

        self.current_player = current_player
        self.state_history = []

    def copy(self):
        new_state = AtaxxState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.state_history = self.state_history.copy()
        return new_state

    def get_legal_moves(self):
        moves = []
        clone_destinations = set()  

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == self.current_player:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and 
                                self.board[nr][nc] == EMPTY and (nr, nc) not in clone_destinations):
                                moves.append((r, c, nr, nc))
                                clone_destinations.add((nr, nc))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == self.current_player:
                    for dr in [-2, -1, 0, 1, 2]:
                        for dc in [-2, -1, 0, 1, 2]:
                            if abs(dr) <= 1 and abs(dc) <= 1:
                                continue  
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and 
                                self.board[nr][nc] == EMPTY and (nr, nc) not in clone_destinations):
                                moves.append((r, c, nr, nc))
                                clone_destinations.add((nr, nc))  
        return moves

    def get_state_hash(self):
        return hash((self.board.tobytes(), self.current_player))

    def make_move(self, move):
        current_hash = self.get_state_hash()
        self.state_history.append(current_hash)
        
        r, c, nr, nc = move
        is_clone = abs(r - nr) <= 1 and abs(c - nc) <= 1
        self.board[nr][nc] = self.current_player
        if not is_clone:
            self.board[r][c] = EMPTY

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                nnr, nnc = nr + dr, nc + dc
                if 0 <= nnr < BOARD_SIZE and 0 <= nnc < BOARD_SIZE and self.board[nnr][nnc] == -self.current_player:
                    self.board[nnr][nnc] = self.current_player

        self.current_player = -self.current_player

    def is_game_over(self):
        if (np.sum(self.board == EMPTY) == 0 or 
            self.get_player_cells(PLAYER_1) == 0 or 
            self.get_player_cells(PLAYER_2) == 0):
            return True
        
        current_hash = self.get_state_hash()
        repetition_count = self.state_history.count(current_hash)
        
        if repetition_count >= 3:
            return True
            
        return False

    def is_threefold_repetition(self):
        current_hash = self.get_state_hash()
        repetition_count = self.state_history.count(current_hash)
        return repetition_count >= 3

    def get_repetition_count(self):
        current_hash = self.get_state_hash()
        return self.state_history.count(current_hash)

    def get_winner(self):
        if not self.is_game_over():
            return 0
        
        current_hash = self.get_state_hash()
        repetition_count = self.state_history.count(current_hash)
        if repetition_count >= 3:
            return 0  
        
        own = np.sum(self.board == PLAYER_1)
        opp = np.sum(self.board == PLAYER_2)

        if own > opp:
            return PLAYER_1
        
        elif opp > own:
            return PLAYER_2
        
        return 0
    
    def board_full(self):
        return np.sum(self.board == EMPTY) == 0
    
    def count_stones(self, player):
        if player not in (PLAYER_1, PLAYER_2):
            raise ValueError("Invalid player. Use PLAYER_1 or PLAYER_2.")
        return np.sum(self.board == player)
    
    def get_empty_cells(self):
        return np.sum(self.board == EMPTY)
    
    def get_player_cells(self, player):
        if player not in (PLAYER_1, PLAYER_2):
            raise ValueError("Invalid player. Use PLAYER_1 or PLAYER_2.")
        return np.sum(self.board == player)

    def display_board(self):
        COLORS = {
            'X': '\033[91m',  
            'O': '\033[94m',  
            '#': '\033[90m',  
            '.': '\033[0m',   
            'border': '\033[93m',  
            'coord': '\033[92m',   
            'reset': '\033[0m'
        }
        
        print(f"{COLORS['border']}  {COLORS['coord']}", end="")
        
        for c in range(BOARD_SIZE):
            print(f" {c} ", end="")
        print(COLORS['reset'])
        
        for r in range(BOARD_SIZE):
            print(f"{COLORS['border']}{COLORS['coord']}{r} {COLORS['reset']}", end="")
            for c in range(BOARD_SIZE):
                cell = self.board[r][c]
                if cell == 1:
                    print(f"{COLORS['X']} X {COLORS['reset']}", end="")
                elif cell == -1:
                    print(f"{COLORS['O']} O {COLORS['reset']}", end="")
                elif cell == -2:  
                    print(f"{COLORS['#']} # {COLORS['reset']}", end="")
                else:
                    print(f"{COLORS['.']} . {COLORS['reset']}", end="")
            print()
        
        player_color = COLORS['X'] if self.current_player == PLAYER_1 else COLORS['O']
        print(f"Current player: {player_color}{'X' if self.current_player == PLAYER_1 else 'O'}{COLORS['reset']}")
        
        repetition_count = self.get_repetition_count()
        if repetition_count > 0:
            print(f"State repetition count: {repetition_count}/3")
            if repetition_count >= 3:
                print(f"{COLORS['border']}🔄 DRAW: 3-fold repetition detected!{COLORS['reset']}")